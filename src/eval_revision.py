"""
Revision experiments for VIAR ECCV paper — addressing reviewer feedback.

Experiments:
1. Decomposed attention analysis: text→vis, vis→vis, text→text, vis→text + per-head variability
2. POPE with per-sample logits for ECE/Brier calibration metrics + qualitative examples
3. Cross-prompt attention consistency (MMStar prompts vs POPE prompts)
4. HallusionBench evaluation (baseline + VIAR)
"""

import modal
import json
import os

app = modal.App("viar-revision")

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
    """Load LLaVA-1.5-7B with eager attention for hooks/output_attentions."""
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
    """Create a VIAR attention mask hook."""
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
# Experiment 1: Decomposed attention analysis
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=7200,
    memory=32768,
)
def decomposed_attention_analysis(num_samples: int = 200):
    """
    Decompose visual attention fraction into 4 quadrants:
      - text_query → vis_key  (the key metric — do text tokens look at images?)
      - vis_query  → vis_key  (structural in causal models)
      - text_query → text_key
      - vis_query  → text_key (should be ~0 due to causal mask)
    Also: per-head variability (std of vis_frac across heads).
    Also: layer-31 analysis — attention entropy, top-attended positions.
    """
    import torch
    import numpy as np
    from datasets import load_dataset

    model, processor = load_llava15()
    layers = model.language_model.model.layers
    num_layers = len(layers)

    dataset = load_dataset("lmms-lab/POPE", split="test")

    # Per-layer accumulators
    layer_stats = {l: {
        "text2vis": [], "vis2vis": [], "text2text": [], "vis2text": [],
        "per_head_vis_frac": [],  # list of [H]-dim arrays
        "attn_entropy": [],       # entropy of attention distribution
    } for l in range(num_layers)}

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

            attentions = outputs.attentions  # tuple of (B, H, S, S) per layer
            seq_len = attentions[0].shape[-1]
            n_vis = 576  # LLaVA-1.5 visual tokens

            for l_idx in range(num_layers):
                attn = attentions[l_idx][0].float().cpu()  # (H, S, S)
                H = attn.shape[0]

                # Quadrant decomposition
                # text queries: positions [n_vis:], vis queries: positions [:n_vis]
                # vis keys: columns [:n_vis], text keys: columns [n_vis:]

                # text_query → vis_key: attn[h, n_vis:, :n_vis]
                t2v = attn[:, n_vis:, :n_vis].sum(dim=-1).mean().item()
                # vis_query → vis_key: attn[h, :n_vis, :n_vis]
                v2v = attn[:, :n_vis, :n_vis].sum(dim=-1).mean().item()
                # text_query → text_key: attn[h, n_vis:, n_vis:]
                t2t = attn[:, n_vis:, n_vis:].sum(dim=-1).mean().item()
                # vis_query → text_key: attn[h, :n_vis, n_vis:]
                v2t = attn[:, :n_vis, n_vis:].sum(dim=-1).mean().item()

                # Normalize: each query's attention sums to 1, so these are fractions
                # t2v + t2t should ≈ 1 for text queries
                # v2v + v2t should ≈ 1 for visual queries (v2t ≈ 0 due to causal mask)

                layer_stats[l_idx]["text2vis"].append(t2v)
                layer_stats[l_idx]["vis2vis"].append(v2v)
                layer_stats[l_idx]["text2text"].append(t2t)
                layer_stats[l_idx]["vis2text"].append(v2t)

                # Per-head vis_frac (for variability analysis)
                # vis_frac per head = mean attention to visual keys across all queries
                per_head_vf = attn[:, :, :n_vis].sum(dim=-1).mean(dim=-1)  # (H,)
                layer_stats[l_idx]["per_head_vis_frac"].append(per_head_vf.numpy())

                # Attention entropy (averaged over heads and queries)
                # Higher entropy = more distributed attention
                eps = 1e-10
                ent = -(attn * (attn + eps).log()).sum(dim=-1).mean().item()
                layer_stats[l_idx]["attn_entropy"].append(ent)

            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{num_samples}] processed")

        except Exception as e:
            print(f"  Error sample {idx}: {e}")
            import traceback; traceback.print_exc()
            continue

    # Aggregate
    result = {
        "model": "llava-hf/llava-1.5-7b-hf",
        "num_samples": num_samples,
        "num_layers": num_layers,
        "n_vis_tokens": 576,
        "by_layer": {},
    }

    for l in range(num_layers):
        if not layer_stats[l]["text2vis"]:
            continue

        t2v = float(np.mean(layer_stats[l]["text2vis"]))
        v2v = float(np.mean(layer_stats[l]["vis2vis"]))
        t2t = float(np.mean(layer_stats[l]["text2text"]))
        v2t = float(np.mean(layer_stats[l]["vis2text"]))

        # Per-head stats
        all_heads = np.stack(layer_stats[l]["per_head_vis_frac"])  # (N, H)
        head_means = all_heads.mean(axis=0)  # (H,)
        head_std = float(head_means.std())
        head_min = float(head_means.min())
        head_max = float(head_means.max())

        ent = float(np.mean(layer_stats[l]["attn_entropy"]))

        result["by_layer"][str(l)] = {
            "text2vis": round(t2v, 5),
            "vis2vis": round(v2v, 5),
            "text2text": round(t2t, 5),
            "vis2text": round(v2t, 5),
            "head_vis_frac_std": round(head_std, 5),
            "head_vis_frac_min": round(head_min, 5),
            "head_vis_frac_max": round(head_max, 5),
            "attn_entropy": round(ent, 4),
            # Aggregate vis_frac (for comparison with old metric)
            "vis_frac_aggregate": round((t2v * (seq_len - 576) + v2v * 576) / seq_len, 5),
        }

    # Print summary
    print(f"\n{'='*80}")
    print(f"DECOMPOSED ATTENTION ANALYSIS — {num_samples} samples")
    print(f"{'Layer':>5} | {'text→vis':>10} | {'vis→vis':>10} | {'text→text':>10} | {'vis→text':>10} | {'head_std':>10} | {'entropy':>8}")
    print("-" * 80)
    for l in range(num_layers):
        d = result["by_layer"].get(str(l), {})
        if d:
            print(f"  L{l:2d}  | {d['text2vis']:10.5f} | {d['vis2vis']:10.5f} | {d['text2text']:10.5f} | {d['vis2text']:10.5f} | {d['head_vis_frac_std']:10.5f} | {d['attn_entropy']:8.4f}")
    print(f"{'='*80}")

    path = "/results/decomposed_attention.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()
    return result


# ============================================================================
# Experiment 2: POPE with ECE/Brier + qualitative examples
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=7200,
    memory=32768,
)
def pope_calibration_analysis(num_samples: int = 500, attn_bias: float = 2.0):
    """
    Run POPE with baseline and VIAR, saving per-sample logits for yes/no tokens.
    Compute ECE (Expected Calibration Error) and Brier score.
    Also collect qualitative examples (baseline wrong → VIAR right, etc.).
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from datasets import load_dataset

    model, processor = load_llava15()
    layers = model.language_model.model.layers
    target_layers = set(range(8, 17))

    dataset = load_dataset("lmms-lab/POPE", split="test")

    # Find token IDs for "yes" and "no"
    yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("no", add_special_tokens=False)[0]
    print(f"  Token IDs: yes={yes_id}, no={no_id}")

    all_results = {}

    for method in ["baseline", "viar"]:
        print(f"\n--- {method} ---")
        hooks = []
        if method == "viar":
            for li in target_layers:
                h = layers[li].self_attn.register_forward_pre_hook(
                    make_viar_hook(attn_bias), with_kwargs=True
                )
                hooks.append(h)

        per_sample = []
        correct = total = tp = fp = tn = fn = yes_count = 0

        for idx, sample in enumerate(dataset):
            if idx >= num_samples:
                break
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

                # Get logits for the last position (first generated token)
                logits = out.logits[0, -1, :].float()
                probs = F.softmax(logits, dim=-1)

                p_yes = probs[yes_id].item()
                p_no = probs[no_id].item()
                # Normalize to binary
                p_yes_norm = p_yes / (p_yes + p_no) if (p_yes + p_no) > 0 else 0.5

                # Also do full generation for answer extraction
                with torch.no_grad():
                    gen_out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
                pred_text = processor.decode(
                    gen_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
                pred_a = "yes" if "yes" in pred_text.lower()[:10] else "no"

                gt = 1 if answer == "yes" else 0
                pred_bin = 1 if pred_a == "yes" else 0

                sample_info = {
                    "idx": idx,
                    "question": question[:100],
                    "answer": answer,
                    "prediction": pred_a,
                    "p_yes": round(p_yes_norm, 5),
                    "correct": pred_a == answer,
                }
                per_sample.append(sample_info)

                if pred_a == answer: correct += 1
                if pred_a == "yes": yes_count += 1
                if answer == "yes" and pred_a == "yes": tp += 1
                elif answer == "no" and pred_a == "yes": fp += 1
                elif answer == "no" and pred_a == "no": tn += 1
                elif answer == "yes" and pred_a == "no": fn += 1
                total += 1

                if (idx + 1) % 100 == 0:
                    print(f"  [{idx+1}/{num_samples}] {method}: {correct/total:.4f}")

            except Exception as e:
                print(f"  Error sample {idx}: {e}")
                continue

        for h in hooks:
            h.remove()

        # Compute calibration metrics
        confidences = []
        accuracies_bin = []
        for s in per_sample:
            conf = s["p_yes"] if s["prediction"] == "yes" else (1 - s["p_yes"])
            confidences.append(conf)
            accuracies_bin.append(1.0 if s["correct"] else 0.0)

        confidences = np.array(confidences)
        accuracies_bin = np.array(accuracies_bin)

        # ECE (10 bins)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bin_details = []
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = accuracies_bin[mask].mean()
                bin_conf = confidences[mask].mean()
                bin_count = int(mask.sum())
                ece += abs(bin_acc - bin_conf) * bin_count / len(confidences)
                bin_details.append({
                    "bin": f"({bin_boundaries[i]:.1f}, {bin_boundaries[i+1]:.1f}]",
                    "count": bin_count,
                    "avg_conf": round(float(bin_conf), 4),
                    "avg_acc": round(float(bin_acc), 4),
                    "gap": round(abs(float(bin_acc - bin_conf)), 4),
                })

        # Brier score: mean((p_yes - gt)^2) where gt = 1 for yes, 0 for no
        brier = 0.0
        for s in per_sample:
            gt = 1.0 if s["answer"] == "yes" else 0.0
            brier += (s["p_yes"] - gt) ** 2
        brier /= len(per_sample)

        acc = correct / total if total > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        all_results[method] = {
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "yes_ratio": round(yes_count / total, 4) if total > 0 else 0,
            "ece": round(float(ece), 5),
            "brier_score": round(float(brier), 5),
            "total": total,
            "per_sample": per_sample,
            "ece_bins": bin_details,
        }

        print(f"  {method}: acc={acc:.4f}, ECE={ece:.5f}, Brier={brier:.5f}")

    # Identify qualitative examples
    baseline_samples = {s["idx"]: s for s in all_results["baseline"]["per_sample"]}
    viar_samples = {s["idx"]: s for s in all_results["viar"]["per_sample"]}

    qualitative = {"baseline_wrong_viar_right": [], "baseline_right_viar_wrong": []}
    for idx in baseline_samples:
        if idx in viar_samples:
            b = baseline_samples[idx]
            v = viar_samples[idx]
            if not b["correct"] and v["correct"]:
                qualitative["baseline_wrong_viar_right"].append({
                    "idx": idx, "question": b["question"], "answer": b["answer"],
                    "baseline_pred": b["prediction"], "viar_pred": v["prediction"],
                    "baseline_p_yes": b["p_yes"], "viar_p_yes": v["p_yes"],
                })
            elif b["correct"] and not v["correct"]:
                qualitative["baseline_right_viar_wrong"].append({
                    "idx": idx, "question": b["question"], "answer": b["answer"],
                    "baseline_pred": b["prediction"], "viar_pred": v["prediction"],
                    "baseline_p_yes": b["p_yes"], "viar_p_yes": v["p_yes"],
                })

    print(f"\n  Qualitative: {len(qualitative['baseline_wrong_viar_right'])} fixed by VIAR, "
          f"{len(qualitative['baseline_right_viar_wrong'])} broken by VIAR")

    # Strip per_sample from saved results to keep file size manageable
    save_results = {}
    for method in all_results:
        save_results[method] = {k: v for k, v in all_results[method].items() if k != "per_sample"}

    result = {
        "model": "llava-hf/llava-1.5-7b-hf",
        "benchmark": "pope",
        "num_samples": num_samples,
        "attn_bias": attn_bias,
        "results": save_results,
        "qualitative": qualitative,
    }

    path = "/results/pope_calibration.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()
    return result


# ============================================================================
# Experiment 3: Cross-prompt attention consistency
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=7200,
    memory=32768,
)
def cross_prompt_attention(num_samples: int = 100):
    """
    Run the text→vis attention analysis on DIFFERENT prompt types:
    1. POPE-style (binary yes/no)
    2. MMStar-style (multiple choice)
    3. Open caption ("Describe this image briefly.")
    
    If the U-shape persists across all prompt types, the neglect zone
    is prompt-invariant (addressing reviewer question 2).
    """
    import torch
    import numpy as np
    from datasets import load_dataset

    model, processor = load_llava15()
    layers = model.language_model.model.layers
    num_layers = len(layers)

    # Load datasets
    pope_ds = load_dataset("lmms-lab/POPE", split="test")
    mmstar_ds = load_dataset("Lin-Chen/MMStar", split="val", trust_remote_code=True)

    prompt_types = {
        "pope_yesno": [],     # "Is there a X in the image? Answer yes or no."
        "mmstar_mcq": [],     # Multiple choice question
        "open_caption": [],   # "Describe this image briefly."
    }

    def analyze_attention(attentions, n_vis=576):
        """Extract text→vis fraction per layer."""
        result = {}
        for l_idx in range(len(attentions)):
            attn = attentions[l_idx][0].float().cpu()  # (H, S, S)
            seq_len = attn.shape[-1]
            if seq_len <= n_vis:
                continue
            # text_query → vis_key
            t2v = attn[:, n_vis:, :n_vis].sum(dim=-1).mean().item()
            result[l_idx] = t2v
        return result

    # --- POPE prompts ---
    print("--- POPE prompts ---")
    for idx, sample in enumerate(pope_ds):
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
            prompt_types["pope_yesno"].append(analyze_attention(outputs.attentions))
            if (idx + 1) % 25 == 0:
                print(f"  POPE [{idx+1}/{num_samples}]")
        except Exception as e:
            print(f"  POPE error {idx}: {e}")
            continue

    # --- MMStar prompts ---
    print("--- MMStar prompts ---")
    for idx, sample in enumerate(mmstar_ds):
        if idx >= num_samples:
            break
        try:
            image = sample["image"]
            question = sample["question"]
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                model.device, torch.float16
            )
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
            prompt_types["mmstar_mcq"].append(analyze_attention(outputs.attentions))
            if (idx + 1) % 25 == 0:
                print(f"  MMStar [{idx+1}/{num_samples}]")
        except Exception as e:
            print(f"  MMStar error {idx}: {e}")
            continue

    # --- Open caption prompts (reuse POPE images) ---
    print("--- Open caption prompts ---")
    for idx, sample in enumerate(pope_ds):
        if idx >= num_samples:
            break
        try:
            image = sample["image"]
            prompt = "USER: <image>\nDescribe this image briefly.\nASSISTANT:"
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                model.device, torch.float16
            )
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
            prompt_types["open_caption"].append(analyze_attention(outputs.attentions))
            if (idx + 1) % 25 == 0:
                print(f"  Caption [{idx+1}/{num_samples}]")
        except Exception as e:
            print(f"  Caption error {idx}: {e}")
            continue

    # Aggregate
    result = {"model": "llava-hf/llava-1.5-7b-hf", "num_samples_per_type": num_samples}

    for ptype, samples in prompt_types.items():
        if not samples:
            continue
        layer_means = {}
        for l in range(num_layers):
            vals = [s.get(l, None) for s in samples if l in s]
            if vals:
                layer_means[str(l)] = round(float(np.mean(vals)), 5)
        result[ptype] = {"by_layer_text2vis": layer_means, "count": len(samples)}

    # Print comparison
    print(f"\n{'='*80}")
    print(f"CROSS-PROMPT ATTENTION CONSISTENCY (text→vis fraction)")
    print(f"{'Layer':>5} | {'POPE':>10} | {'MMStar':>10} | {'Caption':>10}")
    print("-" * 50)
    for l in range(num_layers):
        pope_val = result.get("pope_yesno", {}).get("by_layer_text2vis", {}).get(str(l), 0)
        mmstar_val = result.get("mmstar_mcq", {}).get("by_layer_text2vis", {}).get(str(l), 0)
        caption_val = result.get("open_caption", {}).get("by_layer_text2vis", {}).get(str(l), 0)
        print(f"  L{l:2d}  | {pope_val:10.5f} | {mmstar_val:10.5f} | {caption_val:10.5f}")
    print(f"{'='*80}")

    path = "/results/cross_prompt_attention.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()
    return result


# ============================================================================
# Experiment 4: HallusionBench evaluation
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=7200,
    memory=32768,
)
def hallusionbench_evaluation(attn_bias: float = 2.0, num_samples: int = 951):
    """
    Evaluate on HallusionBench (binary yes/no questions about visual illusions
    and hallucinations). Reviewer requested an additional hallucination benchmark.
    
    Dataset: lmms-lab/HallusionBench, split='image' (951 samples).
    gt_answer is "0" (no) or "1" (yes).
    """
    import torch
    from datasets import load_dataset

    model, processor = load_llava15()
    layers = model.language_model.model.layers
    target_layers = set(range(8, 17))

    # Load HallusionBench — correct dataset ID and split
    dataset = load_dataset("lmms-lab/HallusionBench", split="image", trust_remote_code=True)
    print(f"  HallusionBench loaded: {len(dataset)} samples")
    print(f"  Columns: {dataset.column_names}")
    print(f"  Sample gt_answers: {[dataset[i]['gt_answer'] for i in range(5)]}")

    all_results = {}

    for method in ["baseline", "viar"]:
        print(f"\n--- {method} ---")
        hooks = []
        if method == "viar":
            for li in target_layers:
                h = layers[li].self_attn.register_forward_pre_hook(
                    make_viar_hook(attn_bias), with_kwargs=True
                )
                hooks.append(h)

        correct = total = tp = fp = tn = fn = yes_count = 0
        skipped = 0

        for idx, sample in enumerate(dataset):
            if idx >= num_samples:
                break
            try:
                image = sample.get("image", None)
                if image is None:
                    skipped += 1
                    continue

                # Convert to RGB if needed
                if hasattr(image, 'convert'):
                    image = image.convert("RGB")

                question = sample.get("question", "")
                gt_raw = str(sample.get("gt_answer", "")).strip()

                if not question:
                    skipped += 1
                    continue

                # gt_answer: "0" = no, "1" = yes
                if gt_raw in ["0", "no"]:
                    answer = "no"
                elif gt_raw in ["1", "yes"]:
                    answer = "yes"
                else:
                    skipped += 1
                    continue

                prompt = f"USER: <image>\n{question}\nAnswer with yes or no.\nASSISTANT:"

                inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                    model.device, torch.float16
                )

                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
                pred = processor.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()

                pred_a = "yes" if "yes" in pred.lower()[:10] else "no"
                if pred_a == answer: correct += 1
                if pred_a == "yes": yes_count += 1
                if answer == "yes" and pred_a == "yes": tp += 1
                elif answer == "no" and pred_a == "yes": fp += 1
                elif answer == "no" and pred_a == "no": tn += 1
                elif answer == "yes" and pred_a == "no": fn += 1
                total += 1

                if (idx + 1) % 100 == 0:
                    acc_so_far = correct / total if total > 0 else 0
                    print(f"  [{idx+1}] {method}: acc={acc_so_far:.4f} ({total} valid, {skipped} skipped)")

            except Exception as e:
                if idx < 5:
                    print(f"  Error sample {idx}: {e}")
                    import traceback; traceback.print_exc()
                skipped += 1
                continue

        for h in hooks:
            h.remove()

        acc = correct / total if total > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        all_results[method] = {
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "yes_ratio": round(yes_count / total, 4) if total > 0 else 0,
            "total": total,
            "skipped": skipped,
        }
        print(f"  {method}: acc={acc:.4f}, f1={f1:.4f}, yes_ratio={yes_count/total:.4f} ({total} valid, {skipped} skipped)")

    result = {
        "model": "llava-hf/llava-1.5-7b-hf",
        "benchmark": "hallusionbench",
        "dataset": "lmms-lab/HallusionBench",
        "split": "image",
        "attn_bias": attn_bias,
        "results": all_results,
    }

    path = "/results/hallusionbench.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()
    return result


# ============================================================================
# Entry point: deploy and spawn all experiments
# ============================================================================

@app.local_entrypoint()
def main(experiment: str = "all"):
    """
    Usage:
      modal run src/eval_revision.py                    # run all
      modal run src/eval_revision.py --experiment decomp
      modal run src/eval_revision.py --experiment calib
      modal run src/eval_revision.py --experiment cross
      modal run src/eval_revision.py --experiment hallusion
    """
    if experiment == "all" or experiment == "decomp":
        print("=== Decomposed Attention Analysis ===")
        r = decomposed_attention_analysis.remote(num_samples=200)
        print(f"\nDone. Layers: {len(r.get('by_layer', {}))}")

    if experiment == "all" or experiment == "calib":
        print("\n=== POPE Calibration Analysis ===")
        r = pope_calibration_analysis.remote(num_samples=500)
        for m, res in r["results"].items():
            print(f"  {m}: acc={res['accuracy']}, ECE={res['ece']}, Brier={res['brier_score']}")

    if experiment == "all" or experiment == "cross":
        print("\n=== Cross-Prompt Attention ===")
        r = cross_prompt_attention.remote(num_samples=100)
        for ptype in ["pope_yesno", "mmstar_mcq", "open_caption"]:
            if ptype in r:
                print(f"  {ptype}: {r[ptype]['count']} samples")

    if experiment == "all" or experiment == "hallusion":
        print("\n=== HallusionBench ===")
        r = hallusionbench_evaluation.remote()
        if "results" in r:
            for m, res in r["results"].items():
                print(f"  {m}: acc={res['accuracy']}")
        else:
            print(f"  Error: {r.get('error', 'unknown')}")
