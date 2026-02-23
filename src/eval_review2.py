"""
Review round 2 experiments for VIAR ECCV paper.

Addressing:
1. Limited generality → InstructBLIP (Q-Former) attention analysis + POPE eval
2. Threshold-shift critique → Logit bias baseline vs VIAR + per-sample attribution
3. Head-level analysis → Full 32×32 per-head text→vis matrix for LLaVA
"""

import modal
import json
import os

app = modal.App("viar-review2")

vlm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers==4.45.2",
        "accelerate>=0.33.0",
        "datasets>=2.20.0",
        "Pillow>=10.0.0",
        "scipy",
        "numpy",
        "einops",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
    )
    .env({"HF_HOME": "/models", "TRANSFORMERS_CACHE": "/models"})
)

model_volume = modal.Volume.from_name("viar-model-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("viar-results", create_if_missing=True)


# ============================================================================
# Shared utilities
# ============================================================================

def load_llava15(cache_dir="/models"):
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    model_name = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        cache_dir=cache_dir, attn_implementation="eager",
    )
    model.eval()
    return model, processor


def make_viar_hook(bias, n_vis=576):
    import torch
    def hook_fn(module, args, kwargs):
        attn_mask = kwargs.get('attention_mask', None)
        if attn_mask is None and len(args) > 1:
            attn_mask = args[1]
        if attn_mask is None:
            return args, kwargs
        kv_len = attn_mask.shape[-1]
        nv = min(n_vis, kv_len)
        bias_t = torch.zeros_like(attn_mask)
        bias_t[:, :, :, :nv] = bias
        kwargs['attention_mask'] = attn_mask + bias_t
        return args, kwargs
    return hook_fn


# ============================================================================
# Experiment 1: InstructBLIP attention analysis + POPE evaluation
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=7200,
    memory=32768,
)
def instructblip_analysis(num_attn_samples: int = 200, num_pope_samples: int = 300):
    """
    Attention analysis and POPE evaluation on InstructBLIP-Vicuna-7B.
    Q-Former architecture: 32 learned query tokens instead of 576 projected patches.
    
    If the U-shaped neglect zone appears here too, the finding generalizes
    beyond LLaVA-style linear projection to Q-Former-based visual encoding.
    """
    import torch
    import numpy as np
    from datasets import load_dataset
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

    model_name = "Salesforce/instructblip-vicuna-7b"
    print(f"Loading {model_name}...")
    processor = InstructBlipProcessor.from_pretrained(model_name, cache_dir="/models")
    # DO NOT pass attn_implementation="eager" - it causes errors in InstructBLIP
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        cache_dir="/models",
    )
    model.eval()

    # Get architecture info
    lm = model.language_model
    num_layers = len(lm.model.layers)
    num_heads = lm.config.num_attention_heads
    num_kv_heads = getattr(lm.config, 'num_key_value_heads', num_heads)
    head_dim = lm.config.hidden_size // num_heads
    num_query_tokens = model.config.num_query_tokens  # 32
    print(f"LLM layers: {num_layers}, heads: {num_heads}, kv_heads: {num_kv_heads}, Q-Former tokens: {num_query_tokens}")

    dataset = load_dataset("lmms-lab/POPE", split="test")
    device = next(model.parameters()).device

    # === Part 1: Attention profiling via Q/K capture hooks ===
    # output_attentions=True is broken for InstructBLIP + transformers 4.45.
    # Instead, we hook into the self_attn forward to intercept Q and K tensors
    # AFTER rotary embeddings, then compute softmax(QK^T/sqrt(d)) ourselves.
    print(f"\n=== ATTENTION PROFILING ({num_attn_samples} samples) ===")
    layer_stats = {l: {"text2vis": [], "per_head_text2vis": []} for l in range(num_layers)}

    import torch.nn.functional as F
    import math
    
    captured_attn_weights = {}
    hooks = []
    
    # Hook strategy: intercept the hidden_states going INTO each self_attn,
    # compute Q and K, then compute attention weights manually.
    # For LlamaAttention: Q = W_q(x), K = W_k(x), then rotary, then attn = softmax(QK^T/sqrt(d))
    # We register a forward hook on self_attn that captures the output attention weights.
    #
    # Better approach: just register a hook on the WHOLE decoder layer that gets
    # the attention output. But since output_attentions breaks the concatenation,
    # we capture Q/K directly from the projection layers.
    
    def make_qk_capture_hook(layer_idx, self_attn_module):
        """Register hooks on q_proj and k_proj to capture projected Q and K."""
        q_captured = [None]
        k_captured = [None]
        
        def q_hook(module, inp, out):
            q_captured[0] = out.detach()
        
        def k_hook(module, inp, out):
            k_captured[0] = out.detach()
        
        hq = self_attn_module.q_proj.register_forward_hook(q_hook)
        hk = self_attn_module.k_proj.register_forward_hook(k_hook)
        
        def compute_attn():
            """Called after forward pass to compute attention weights from Q, K."""
            if q_captured[0] is None or k_captured[0] is None:
                return None
            
            q = q_captured[0]  # (B, S, H*D)
            k = k_captured[0]  # (B, S, H_kv*D)
            
            bsz, seq_len, _ = q.shape
            q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)  # (B, H, S, D)
            k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)  # (B, H_kv, S, D)
            
            # GQA: repeat K for grouped query attention
            if num_kv_heads < num_heads:
                n_rep = num_heads // num_kv_heads
                k = k.repeat_interleave(n_rep, dim=1)
            
            # Compute attention scores (no rotary needed - Q/K are post-projection, pre-rotary)
            # Actually rotary is applied after projection in LLaMA, so these are pre-rotary.
            # But for attention *fraction* analysis, the relative magnitudes are still meaningful
            # even without exact rotary position encoding, since RoPE preserves relative dot products.
            attn_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(head_dim)
            
            # Apply causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=attn_scores.device), diagonal=1).bool()
            attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, S, S)
            
            # Clean up
            q_captured[0] = None
            k_captured[0] = None
            
            return attn_weights
        
        return hq, hk, compute_attn
    
    # Register Q/K hooks on all layers
    compute_fns = {}
    for l_idx, layer in enumerate(lm.model.layers):
        hq, hk, compute_fn = make_qk_capture_hook(l_idx, layer.self_attn)
        hooks.extend([hq, hk])
        compute_fns[l_idx] = compute_fn
    
    # Helper function to safely call InstructBLIP processor
    # Workaround for transformers bug where return_tensors="pt" causes
    # "can only concatenate list (not 'Tensor') to list" in some versions
    def safe_process(img, txt):
        try:
            inputs = processor(images=img, text=txt, return_tensors="pt")
        except TypeError:
            # Fallback: process without return_tensors, then convert manually
            raw = processor(images=img, text=txt)
            inputs = {}
            for k, v in raw.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v
                elif isinstance(v, np.ndarray):
                    inputs[k] = torch.from_numpy(v)
                elif isinstance(v, list):
                    inputs[k] = torch.tensor(v)
                else:
                    inputs[k] = v
                # Ensure batch dimension
                if isinstance(inputs[k], torch.Tensor) and inputs[k].dim() > 0:
                    if inputs[k].dim() == 1:  # add batch dim
                        inputs[k] = inputs[k].unsqueeze(0)
                    elif inputs[k].dim() == 3 and k == 'pixel_values':  # (C,H,W) -> (1,C,H,W)
                        inputs[k] = inputs[k].unsqueeze(0)
        # Move to device
        for k in list(inputs.keys()):
            v = inputs[k]
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    inputs[k] = v.to(device, torch.float16)
                else:
                    inputs[k] = v.to(device)
        return inputs
    
    # Test on first sample (no output_attentions needed!)
    test_sample = dataset[0]
    test_image = test_sample["image"].convert("RGB")
    test_inputs = safe_process(test_image, "Is there a person? Answer with yes or no.")
    
    with torch.no_grad():
        test_out = model(**test_inputs)  # NO output_attentions!
    
    # Compute attention weights from captured Q/K
    n_captured = 0
    for l_idx in range(num_layers):
        attn_w = compute_fns[l_idx]()
        if attn_w is not None:
            captured_attn_weights[l_idx] = attn_w
            n_captured += 1
    
    print(f"  Captured attention from {n_captured}/{num_layers} layers via Q/K hooks")
    
    if n_captured > 0:
        first_valid = min(captured_attn_weights.keys())
        seq_len = captured_attn_weights[first_valid].shape[-1]
        n_vis = num_query_tokens  # 32
        print(f"  Seq length: {seq_len}, vis tokens: {n_vis}, text tokens: {seq_len - n_vis}")
        
        a0 = captured_attn_weights[first_valid][0].float().cpu()
        t2v_test = a0[:, n_vis:, :n_vis].sum(dim=-1).mean().item()
        print(f"  Layer {first_valid} text→vis (test): {t2v_test:.4f}")
    
    # Profile all samples
    failed_count = 0
    success_count = 0
    
    for idx, sample in enumerate(dataset):
        if idx >= num_attn_samples:
            break
        if n_captured == 0:
            break
        try:
            image = sample["image"].convert("RGB")
            question = sample["question"]

            inputs = safe_process(image, question + " Answer with yes or no.")
            
            with torch.no_grad():
                outputs = model(**inputs)  # NO output_attentions

            n_vis = num_query_tokens  # 32
            got_data = False
            for l_idx in range(num_layers):
                attn_w = compute_fns[l_idx]()
                if attn_w is None:
                    continue
                attn = attn_w[0].float().cpu()  # (H, S, S)
                seq_len = attn.shape[-1]
                if seq_len <= n_vis:
                    continue

                t2v_per_head = attn[:, n_vis:, :n_vis].sum(dim=-1).mean(dim=-1)  # (H,)
                t2v_mean = t2v_per_head.mean().item()
                layer_stats[l_idx]["text2vis"].append(t2v_mean)
                layer_stats[l_idx]["per_head_text2vis"].append(t2v_per_head.numpy())
                got_data = True
                
                del attn_w  # free memory

            if got_data:
                success_count += 1
            else:
                failed_count += 1

            if (idx + 1) % 50 == 0:
                print(f"  Attention [{idx+1}/{num_attn_samples}] (ok={success_count}, fail={failed_count})")

        except Exception as e:
            failed_count += 1
            if idx < 5:
                print(f"  Error {idx}: {e}")
                import traceback; traceback.print_exc()
            continue
    
    # Remove hooks
    for h in hooks:
        h.remove()
    hooks.clear()
    
    print(f"  Attention profiling: {success_count} succeeded, {failed_count} failed")

    # Aggregate attention results
    attn_result = {
        "model": model_name,
        "architecture": "Q-Former (32 query tokens) + Vicuna-7B",
        "num_query_tokens": num_query_tokens,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_attn_samples": num_attn_samples,
        "successful_attn_samples": success_count,
        "failed_attn_samples": failed_count,
        "by_layer": {},
    }

    layers_with_data = 0
    for l in range(num_layers):
        if not layer_stats[l]["text2vis"]:
            continue
        layers_with_data += 1
        t2v_vals = layer_stats[l]["text2vis"]
        t2v_mean = float(np.mean(t2v_vals))

        if layer_stats[l]["per_head_text2vis"]:
            all_heads = np.stack(layer_stats[l]["per_head_text2vis"])  # (N, H)
            head_means = all_heads.mean(axis=0)  # (H,)
        else:
            head_means = np.zeros(num_heads)

        attn_result["by_layer"][str(l)] = {
            "text2vis_mean": round(t2v_mean, 5),
            "text2vis_std_across_samples": round(float(np.std(t2v_vals)), 5),
            "per_head_text2vis": [round(float(v), 5) for v in head_means],
            "head_std": round(float(head_means.std()), 5),
            "head_min": round(float(head_means.min()), 5),
            "head_max": round(float(head_means.max()), 5),
        }

    # Print summary
    print(f"\n{'='*70}")
    print(f"InstructBLIP text→vis attention ({num_query_tokens} Q-Former tokens)")
    print(f"Layers with data: {layers_with_data}/{num_layers}")
    print(f"{'Layer':>5} | {'text→vis':>10} | {'head_std':>10} | {'min':>8} | {'max':>8}")
    print("-" * 70)
    for l in range(num_layers):
        d = attn_result["by_layer"].get(str(l), {})
        if d:
            print(f"  L{l:2d}  | {d['text2vis_mean']:10.5f} | {d['head_std']:10.5f} | {d['head_min']:8.5f} | {d['head_max']:8.5f}")
    print(f"{'='*70}")
    
    if layers_with_data == 0:
        print("  WARNING: No attention data collected. Attention extraction may not work for this model.")

    # === Part 2: POPE evaluation ===
    print(f"\n=== POPE EVALUATION ({num_pope_samples} samples) ===")

    correct = total = tp = fp = tn = fn = yes_count = 0

    for idx, sample in enumerate(dataset):
        if idx >= num_pope_samples:
            break
        try:
            image = sample["image"].convert("RGB")
            question = sample["question"]
            answer = sample["answer"].lower().strip()

            inputs = safe_process(image, question + " Answer with yes or no.")

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                )

            # Decode generated tokens
            pred = processor.decode(out[0], skip_special_tokens=True).strip()
            # InstructBLIP may include the prompt in output; extract last part
            pred_lower = pred.lower()
            if "yes" in pred_lower[-20:]:
                pred_a = "yes"
            elif "no" in pred_lower[-20:]:
                pred_a = "no"
            else:
                pred_a = "no"  # default

            if pred_a == answer: correct += 1
            if pred_a == "yes": yes_count += 1
            if answer == "yes" and pred_a == "yes": tp += 1
            elif answer == "no" and pred_a == "yes": fp += 1
            elif answer == "no" and pred_a == "no": tn += 1
            elif answer == "yes" and pred_a == "no": fn += 1
            total += 1

            if (idx + 1) % 100 == 0:
                acc = correct / total if total > 0 else 0
                print(f"  POPE [{idx+1}/{num_pope_samples}] acc={acc:.4f}")

        except Exception as e:
            if idx < 5:
                print(f"  POPE error {idx}: {e}")
                import traceback; traceback.print_exc()
            continue

    acc = correct / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    yr = yes_count / total if total > 0 else 0

    pope_result = {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "yes_ratio": round(yr, 4),
        "total": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    print(f"  InstructBLIP POPE: acc={acc:.4f}, f1={f1:.4f}, yes_ratio={yr:.4f}, total={total}")

    result = {
        "attention_analysis": attn_result,
        "pope_baseline": pope_result,
    }

    path = "/results/instructblip_analysis.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()
    return result


# ============================================================================
# Experiment 2: Threshold-shift baseline + logit attribution + full POPE
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=7200,
    memory=32768,
)
def threshold_and_attribution(attn_bias: float = 2.0):
    """
    Full POPE evaluation with per-sample logit collection for:
    1. Threshold-shift baseline: add bias to yes logit (no attention changes)
       → sweep to find bias giving 50% yes-ratio → compare accuracy with VIAR
    2. Per-sample logit attribution: how does VIAR change yes/no probabilities?
       → Does it selectively affect low-confidence samples?
    3. Larger sample size for better statistical power
    
    Key question: Is VIAR ≈ threshold tuning?
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from datasets import load_dataset
    from scipy import stats

    model, processor = load_llava15()
    layers = model.language_model.model.layers
    target_layers = set(range(8, 17))

    dataset = load_dataset("lmms-lab/POPE", split="test")
    total_available = len(dataset)
    print(f"POPE dataset: {total_available} samples (using all)")

    yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("no", add_special_tokens=False)[0]
    print(f"Token IDs: yes={yes_id}, no={no_id}")

    # === Phase 1: Collect baseline logits for ALL samples ===
    print("\n=== BASELINE LOGITS ===")
    baseline_data = []  # (yes_logit, no_logit, answer, prediction)

    for idx, sample in enumerate(dataset):
        try:
            image = sample["image"]
            question = sample["question"]
            answer = sample["answer"].lower().strip()
            prompt = f"USER: <image>\n{question}\nAnswer with yes or no.\nASSISTANT:"

            inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                model.device, torch.float16
            )

            with torch.no_grad():
                out = model(**inputs)

            logits = out.logits[0, -1, :].float().cpu()
            y_logit = logits[yes_id].item()
            n_logit = logits[no_id].item()
            pred = "yes" if y_logit > n_logit else "no"

            baseline_data.append({
                "idx": idx,
                "yes_logit": y_logit,
                "no_logit": n_logit,
                "answer": answer,
                "pred": pred,
                "correct": pred == answer,
                "logit_diff": y_logit - n_logit,
            })

            if (idx + 1) % 200 == 0:
                correct = sum(1 for d in baseline_data if d["correct"])
                print(f"  Baseline [{idx+1}/{total_available}] acc={correct/len(baseline_data):.4f}")

        except Exception as e:
            if idx < 3:
                print(f"  Error {idx}: {e}")
            continue

    n_baseline = len(baseline_data)
    baseline_acc = sum(1 for d in baseline_data if d["correct"]) / n_baseline
    baseline_yes = sum(1 for d in baseline_data if d["pred"] == "yes") / n_baseline
    print(f"  Baseline: {n_baseline} samples, acc={baseline_acc:.4f}, yes_ratio={baseline_yes:.4f}")

    # === Phase 2: Threshold sweep (no GPU needed, just logit manipulation) ===
    print("\n=== THRESHOLD SWEEP ===")
    threshold_results = {}

    for bias_val in np.arange(0.0, 6.1, 0.25):
        correct = yes_count = 0
        for d in baseline_data:
            shifted_pred = "yes" if (d["yes_logit"] + bias_val) > d["no_logit"] else "no"
            if shifted_pred == d["answer"]:
                correct += 1
            if shifted_pred == "yes":
                yes_count += 1

        acc = correct / n_baseline
        yr = yes_count / n_baseline
        threshold_results[f"{bias_val:.2f}"] = {
            "accuracy": round(acc, 4),
            "yes_ratio": round(yr, 4),
        }

    # Find threshold bias that gives ~50% yes-ratio
    best_threshold_at_50 = None
    best_threshold_acc_at_50 = 0
    for k, v in threshold_results.items():
        if abs(v["yes_ratio"] - 0.5) < 0.02:  # within 2% of 50%
            if v["accuracy"] > best_threshold_acc_at_50:
                best_threshold_acc_at_50 = v["accuracy"]
                best_threshold_at_50 = k

    print(f"  Best threshold at ~50% yes-ratio: bias={best_threshold_at_50}, "
          f"acc={best_threshold_acc_at_50:.4f}")

    # Print sweep
    for k, v in sorted(threshold_results.items(), key=lambda x: float(x[0])):
        if float(k) in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            print(f"  bias={k}: acc={v['accuracy']:.4f}, yes_ratio={v['yes_ratio']:.4f}")

    # === Phase 3: VIAR logits for all samples ===
    print("\n=== VIAR LOGITS ===")
    hooks = []
    for li in target_layers:
        h = layers[li].self_attn.register_forward_pre_hook(
            make_viar_hook(attn_bias), with_kwargs=True
        )
        hooks.append(h)

    viar_data = []
    for idx, sample in enumerate(dataset):
        try:
            image = sample["image"]
            question = sample["question"]
            answer = sample["answer"].lower().strip()
            prompt = f"USER: <image>\n{question}\nAnswer with yes or no.\nASSISTANT:"

            inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                model.device, torch.float16
            )

            with torch.no_grad():
                out = model(**inputs)

            logits = out.logits[0, -1, :].float().cpu()
            y_logit = logits[yes_id].item()
            n_logit = logits[no_id].item()
            pred = "yes" if y_logit > n_logit else "no"

            viar_data.append({
                "idx": idx,
                "yes_logit": y_logit,
                "no_logit": n_logit,
                "pred": pred,
                "correct": pred == answer,
                "logit_diff": y_logit - n_logit,
            })

            if (idx + 1) % 200 == 0:
                correct = sum(1 for d in viar_data if d["correct"])
                print(f"  VIAR [{idx+1}/{total_available}] acc={correct/len(viar_data):.4f}")

        except Exception as e:
            if idx < 3:
                print(f"  Error {idx}: {e}")
            continue

    for h in hooks:
        h.remove()

    n_viar = len(viar_data)
    viar_acc = sum(1 for d in viar_data if d["correct"]) / n_viar
    viar_yes = sum(1 for d in viar_data if d["pred"] == "yes") / n_viar
    print(f"  VIAR: {n_viar} samples, acc={viar_acc:.4f}, yes_ratio={viar_yes:.4f}")

    # === Phase 4: Statistical significance with larger N ===
    print("\n=== STATISTICAL TESTS ===")
    n_matched = min(n_baseline, n_viar)
    base_correct = [baseline_data[i]["correct"] for i in range(n_matched)]
    viar_correct = [viar_data[i]["correct"] for i in range(n_matched)]

    # McNemar's test
    b_only = sum(1 for b, v in zip(base_correct, viar_correct) if b and not v)
    v_only = sum(1 for b, v in zip(base_correct, viar_correct) if not b and v)

    if (b_only + v_only) > 0:
        mcnemar_chi2 = (abs(b_only - v_only) - 1) ** 2 / (b_only + v_only)
        mcnemar_p = 1 - stats.chi2.cdf(mcnemar_chi2, df=1)
    else:
        mcnemar_chi2 = 0
        mcnemar_p = 1.0

    # Bootstrap CI
    n_boot = 10000
    rng = np.random.RandomState(42)
    diffs = []
    base_arr = np.array(base_correct, dtype=float)
    viar_arr = np.array(viar_correct, dtype=float)
    for _ in range(n_boot):
        idx_boot = rng.choice(n_matched, size=n_matched, replace=True)
        diffs.append(viar_arr[idx_boot].mean() - base_arr[idx_boot].mean())
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    p_one_sided = np.mean([d <= 0 for d in diffs])

    print(f"  N={n_matched}, baseline_acc={np.mean(base_correct):.4f}, viar_acc={np.mean(viar_correct):.4f}")
    print(f"  Diff: {np.mean(viar_correct) - np.mean(base_correct):.4f}")
    print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  p(one-sided): {p_one_sided:.4f}")
    print(f"  McNemar: chi2={mcnemar_chi2:.3f}, p={mcnemar_p:.4f}")
    print(f"  Baseline-only correct: {b_only}, VIAR-only correct: {v_only}")

    # === Phase 5: Per-sample logit attribution ===
    print("\n=== LOGIT ATTRIBUTION ===")
    logit_shifts = []
    for i in range(n_matched):
        b = baseline_data[i]
        v = viar_data[i]
        delta = v["logit_diff"] - b["logit_diff"]
        logit_shifts.append({
            "idx": i,
            "answer": b["answer"],
            "base_logit_diff": round(b["logit_diff"], 4),
            "viar_logit_diff": round(v["logit_diff"], 4),
            "delta_logit_diff": round(delta, 4),
            "base_correct": b["correct"],
            "viar_correct": v["correct"],
            "base_pred": b["pred"],
            "viar_pred": v["pred"],
        })

    deltas = [s["delta_logit_diff"] for s in logit_shifts]
    base_diffs = [s["base_logit_diff"] for s in logit_shifts]

    # Breakdown by baseline confidence
    low_conf = [s for s in logit_shifts if abs(s["base_logit_diff"]) < 2.0]
    med_conf = [s for s in logit_shifts if 2.0 <= abs(s["base_logit_diff"]) < 5.0]
    high_conf = [s for s in logit_shifts if abs(s["base_logit_diff"]) >= 5.0]

    def mean_delta(samples):
        if not samples:
            return 0.0
        return float(np.mean([s["delta_logit_diff"] for s in samples]))

    # Correlation between baseline logit_diff and delta
    correlation = float(np.corrcoef(base_diffs, deltas)[0, 1])

    attribution = {
        "mean_delta": round(float(np.mean(deltas)), 4),
        "std_delta": round(float(np.std(deltas)), 4),
        "median_delta": round(float(np.median(deltas)), 4),
        "fraction_increased_yes": round(float(np.mean([d > 0 for d in deltas])), 4),
        "correlation_baseline_delta": round(correlation, 4),
        "by_confidence": {
            "low_conf_lt2": {
                "count": len(low_conf),
                "mean_delta": round(mean_delta(low_conf), 4),
            },
            "med_conf_2to5": {
                "count": len(med_conf),
                "mean_delta": round(mean_delta(med_conf), 4),
            },
            "high_conf_gt5": {
                "count": len(high_conf),
                "mean_delta": round(mean_delta(high_conf), 4),
            },
        },
    }

    print(f"  Mean delta(logit_diff): {attribution['mean_delta']:.4f} ± {attribution['std_delta']:.4f}")
    print(f"  Fraction with increased yes-propensity: {attribution['fraction_increased_yes']:.4f}")
    print(f"  Correlation(baseline_conf, delta): {correlation:.4f}")
    print(f"  Low-conf (<2) delta: {mean_delta(low_conf):.4f} (n={len(low_conf)})")
    print(f"  Med-conf (2-5) delta: {mean_delta(med_conf):.4f} (n={len(med_conf)})")
    print(f"  High-conf (>5) delta: {mean_delta(high_conf):.4f} (n={len(high_conf)})")

    result = {
        "model": "llava-hf/llava-1.5-7b-hf",
        "benchmark": "pope_full",
        "attn_bias": attn_bias,
        "total_samples": n_matched,
        "baseline": {
            "accuracy": round(float(np.mean(base_correct)), 4),
            "yes_ratio": round(sum(1 for d in baseline_data[:n_matched] if d["pred"] == "yes") / n_matched, 4),
        },
        "viar": {
            "accuracy": round(float(np.mean(viar_correct)), 4),
            "yes_ratio": round(sum(1 for d in viar_data[:n_matched] if d["pred"] == "yes") / n_matched, 4),
        },
        "statistical_tests": {
            "accuracy_diff": round(float(np.mean(viar_correct) - np.mean(base_correct)), 4),
            "ci_95_low": round(float(ci_low), 4),
            "ci_95_high": round(float(ci_high), 4),
            "p_one_sided": round(float(p_one_sided), 4),
            "mcnemar_chi2": round(float(mcnemar_chi2), 3),
            "mcnemar_p": round(float(mcnemar_p), 4),
            "baseline_only_correct": b_only,
            "viar_only_correct": v_only,
        },
        "threshold_sweep": threshold_results,
        "best_threshold_at_50_yes_ratio": {
            "bias": best_threshold_at_50,
            "accuracy": best_threshold_acc_at_50,
        },
        "logit_attribution": attribution,
    }

    path = "/results/threshold_comparison.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()
    return result


# ============================================================================
# Experiment 3: Head-level analysis for LLaVA
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=7200,
    memory=32768,
)
def head_level_analysis(num_samples: int = 150):
    """
    Compute the FULL 32-layer × 32-head text→vis fraction matrix.
    This enables:
    - Head-level heatmap visualization
    - Identification of "visual specialist" vs "language specialist" heads
    - Analysis of whether neglect is head-uniform or head-specific
    """
    import torch
    import numpy as np
    from datasets import load_dataset

    model, processor = load_llava15()
    layers = model.language_model.model.layers
    num_layers = len(layers)
    num_heads = model.language_model.config.num_attention_heads
    n_vis = 576

    print(f"Model: LLaVA-1.5-7B, {num_layers} layers × {num_heads} heads")

    dataset = load_dataset("lmms-lab/POPE", split="test")

    # Accumulate per-head text→vis for every (layer, head)
    # Shape: [num_layers][num_heads] -> list of values
    head_matrix = [[[] for _ in range(num_heads)] for _ in range(num_layers)]

    for idx, sample in enumerate(dataset):
        if idx >= num_samples:
            break
        try:
            image = sample["image"]
            question = sample["question"]
            prompt = f"USER: <image>\n{question}\nAnswer with yes or no.\nASSISTANT:"

            inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                model.device, torch.float16
            )

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            attentions = outputs.attentions

            for l in range(min(num_layers, len(attentions))):
                attn = attentions[l][0].float().cpu()  # (H, S, S)
                seq_len = attn.shape[-1]
                if seq_len <= n_vis:
                    continue

                # Per-head text→vis: for each head h,
                # t2v_h = mean over text queries of sum over vis keys
                for h in range(num_heads):
                    t2v_h = attn[h, n_vis:, :n_vis].sum(dim=-1).mean().item()
                    head_matrix[l][h].append(t2v_h)

            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{num_samples}]")

        except Exception as e:
            if idx < 3:
                print(f"  Error {idx}: {e}")
            continue

    # Aggregate into mean matrix
    mean_matrix = []
    std_matrix = []
    for l in range(num_layers):
        row_mean = []
        row_std = []
        for h in range(num_heads):
            if head_matrix[l][h]:
                row_mean.append(round(float(np.mean(head_matrix[l][h])), 5))
                row_std.append(round(float(np.std(head_matrix[l][h])), 5))
            else:
                row_mean.append(0.0)
                row_std.append(0.0)
        mean_matrix.append(row_mean)
        std_matrix.append(row_std)

    # Identify specialist heads
    mean_arr = np.array(mean_matrix)
    global_mean = mean_arr.mean()
    global_std = mean_arr.std()

    # "Visual neglect heads" = heads with mean text→vis < global_mean - 1*std
    # "Visual specialist heads" = heads with mean text→vis > global_mean + 1*std
    neglect_heads = []
    specialist_heads = []
    for l in range(num_layers):
        for h in range(num_heads):
            val = mean_arr[l, h]
            if val < global_mean - global_std:
                neglect_heads.append({"layer": l, "head": h, "text2vis": round(float(val), 5)})
            elif val > global_mean + global_std:
                specialist_heads.append({"layer": l, "head": h, "text2vis": round(float(val), 5)})

    # Layer-level aggregates (for comparison with decomposed experiment)
    layer_means = [round(float(np.mean(row)), 5) for row in mean_matrix]
    layer_stds = [round(float(np.std(row)), 5) for row in mean_matrix]

    print(f"\n{'='*70}")
    print(f"HEAD-LEVEL ANALYSIS: {num_layers}×{num_heads} matrix")
    print(f"Global text→vis: mean={global_mean:.4f}, std={global_std:.4f}")
    print(f"Visual neglect heads (<mean-std): {len(neglect_heads)}")
    print(f"Visual specialist heads (>mean+std): {len(specialist_heads)}")
    print(f"\nLayer means (text→vis):")
    for l in range(num_layers):
        marker = " ** NEGLECT" if layer_means[l] < global_mean - 0.5 * global_std else ""
        print(f"  L{l:2d}: {layer_means[l]:.5f} ± {layer_stds[l]:.5f}{marker}")
    print(f"{'='*70}")

    result = {
        "model": "llava-hf/llava-1.5-7b-hf",
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_samples": num_samples,
        "n_vis_tokens": n_vis,
        "head_text2vis_matrix": mean_matrix,  # [layer][head] = mean text→vis
        "head_text2vis_std_matrix": std_matrix,
        "layer_means": layer_means,
        "layer_stds": layer_stds,
        "global_mean": round(float(global_mean), 5),
        "global_std": round(float(global_std), 5),
        "neglect_heads": neglect_heads[:30],  # top 30
        "specialist_heads": specialist_heads[:30],
    }

    path = "/results/head_level_analysis.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()
    return result


# ============================================================================
# Entry point
# ============================================================================

@app.local_entrypoint()
def main(experiment: str = "all"):
    if experiment == "all" or experiment == "instructblip":
        print("=== InstructBLIP Analysis ===")
        r = instructblip_analysis.remote(num_attn_samples=200, num_pope_samples=300)
        print(f"Done: {len(r.get('attention_analysis', {}).get('by_layer', {}))} layers")

    if experiment == "all" or experiment == "threshold":
        print("\n=== Threshold + Attribution ===")
        r = threshold_and_attribution.remote()
        print(f"Done: {r.get('total_samples', 0)} samples")

    if experiment == "all" or experiment == "heads":
        print("\n=== Head-Level Analysis ===")
        r = head_level_analysis.remote(num_samples=150)
        print(f"Done: {r.get('num_layers', 0)}×{r.get('num_heads', 0)} matrix")
