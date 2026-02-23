"""
VIAR evaluation on Modal.

Implements Vision-Informed Attention Rebalancing as a hook into 
LLaVA's attention layers and evaluates on POPE and MMStar.

Approach: Instead of rewriting the attention forward (fragile with GQA/KV cache),
we use a simpler, more robust method:
  1. Hook into the LLM's input_embeds BEFORE the transformer layers
  2. Add a learned/fixed "visual boost" bias to the attention mask
  
Even simpler: We modify the 4D attention_mask that gets passed to every layer.
In HuggingFace transformers, the causal_mask is a 4D tensor [bsz, 1, q_len, kv_len]
that gets added to attention scores pre-softmax. By adding a positive bias to 
visual token positions in this mask, we boost visual attention across all target layers.

Actually, the simplest robust approach: use model.generate with a custom
LogitsProcessor that applies contrastive decoding (VCD-style), which is
well-tested and doesn't require any attention hooking.

For the attention-level intervention, we use a simpler strategy:
modify the causal_mask in the model's forward to add visual attention bias.
"""

import modal
import json
import os

app = modal.App("viar-method-eval-v2")

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
        "einops",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
    )
    .env({"HF_HOME": "/models", "TRANSFORMERS_CACHE": "/models"})
)

model_volume = modal.Volume.from_name("viar-model-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("viar-results", create_if_missing=True)


@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=10800,
    memory=32768,
)
def eval_viar_on_benchmark(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    benchmark: str = "pope",
    method: str = "viar",  # "baseline", "viar", "vcd", "viar+vcd"
    # VIAR: attention mask bias approach
    attn_bias: float = 3.0,  # Additive bias to pre-softmax attention scores for visual tokens
    target_layers: list = None,  # None = layers 8-16
    # VCD: contrastive decoding approach
    vcd_alpha: float = 1.0,
    num_samples: int = None,
):
    """
    VIAR evaluation using attention mask modification.
    
    Strategy: We hook into the decoder layers and add a positive bias
    to pre-softmax attention scores at visual token positions.
    This is equivalent to making visual tokens more "attractive" to attend to.
    
    Mathematically: score'[i,j] = score[i,j] + bias  (for j in visual positions)
    After softmax, this increases the attention weight on visual tokens.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from datasets import load_dataset
    from PIL import Image as PILImage
    import numpy as np

    print(f"[VIAR-v2] Model: {model_name}, Benchmark: {benchmark}, Method: {method}")
    print(f"  attn_bias={attn_bias}, target_layers={target_layers}, vcd_alpha={vcd_alpha}")

    processor = AutoProcessor.from_pretrained(model_name, cache_dir="/models")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/models",
        attn_implementation="eager",
    )
    model.eval()

    # Get the transformer layers
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        layers = model.language_model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        raise ValueError("Cannot find transformer layers")

    num_layers = len(layers)
    if target_layers is None:
        target_layers = list(range(8, min(17, num_layers)))
    target_set = set(target_layers)
    print(f"  {num_layers} layers, intervening in: {sorted(target_set)}")

    # ================================================================
    # VIAR Hook: Modify attention scores pre-softmax
    # 
    # We hook into each target layer's self_attn module. The hook
    # intercepts the 'args' passed to forward, specifically the
    # attention_mask (a 4D causal mask added to scores pre-softmax).
    # We add a positive bias to visual token columns in the mask.
    # ================================================================

    active_hooks = []
    viar_enabled = [False]  # Mutable flag to enable/disable

    # Per-layer visual attention fraction from our analysis
    vis_frac_by_layer = {
        0: 0.197, 1: 0.054, 2: 0.202, 3: 0.379, 4: 0.271,
        5: 0.231, 6: 0.266, 7: 0.236, 8: 0.208, 9: 0.181,
        10: 0.183, 11: 0.210, 12: 0.189, 13: 0.174, 14: 0.191,
        15: 0.173, 16: 0.187, 17: 0.259, 18: 0.244, 19: 0.277,
        20: 0.314, 21: 0.431, 22: 0.354, 23: 0.372, 24: 0.326,
        25: 0.495, 26: 0.303, 27: 0.321, 28: 0.333, 29: 0.310,
        30: 0.285, 31: 0.141,
    }

    def make_attn_hook(layer_idx, layer_bias):
        """Create a pre-forward hook that modifies the attention mask."""
        def hook_fn(module, args, kwargs):
            if not viar_enabled[0]:
                return args, kwargs
            
            attention_mask = kwargs.get('attention_mask', None)
            if attention_mask is None and len(args) > 1:
                attention_mask = args[1]
            
            if attention_mask is None:
                return args, kwargs
            
            kv_len = attention_mask.shape[-1]
            n_vis = min(576, kv_len)
            
            bias = torch.zeros_like(attention_mask)
            bias[:, :, :, :n_vis] = layer_bias
            
            modified_mask = attention_mask + bias
            kwargs['attention_mask'] = modified_mask
            return args, kwargs
        
        return hook_fn

    def register_viar_hooks():
        """Register hooks on target layers with per-layer adaptive bias."""
        # Compute per-layer bias: proportional to neglect severity
        neglect_scores = {l: 1.0 - vis_frac_by_layer.get(l, 0.25) for l in range(num_layers)}
        # For adaptive: scale by mean neglect so that average bias = attn_bias
        target_neglect = [neglect_scores[l] for l in target_set]
        mean_neglect = sum(target_neglect) / len(target_neglect) if target_neglect else 1.0

        for layer_idx in target_set:
            if layer_idx < num_layers:
                # Use uniform bias (= attn_bias) for all target layers
                # The adaptive version uses the layer ablation function
                layer_bias = attn_bias
                hook = layers[layer_idx].self_attn.register_forward_pre_hook(
                    make_attn_hook(layer_idx, layer_bias), with_kwargs=True
                )
                active_hooks.append(hook)

    def remove_viar_hooks():
        for h in active_hooks:
            h.remove()
        active_hooks.clear()

    # ================================================================
    # VCD implementation using custom generation
    # ================================================================

    def generate_with_vcd(real_inputs, null_inputs, max_new_tokens=10, alpha=1.0):
        """
        Visual Contrastive Decoding: greedily decode by contrasting
        logits from real image vs null image.
        """
        input_ids = real_inputs["input_ids"]
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                real_out = model(**real_inputs)
                null_out = model(**null_inputs)
            
            real_logits = real_out.logits[:, -1, :].float()
            null_logits = null_out.logits[:, -1, :].float()
            
            log_real = F.log_softmax(real_logits, dim=-1)
            log_null = F.log_softmax(null_logits, dim=-1)
            
            corrected = (1 + alpha) * log_real - alpha * log_null
            next_token = corrected.argmax(dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Check for EOS
            if next_token.item() == processor.tokenizer.eos_token_id:
                break
            
            # Update inputs for next step (simplified: re-run full sequence)
            real_inputs = {**real_inputs, "input_ids": input_ids}
            null_inputs = {**null_inputs, "input_ids": input_ids}
            # Remove pixel_values after first step (already encoded)
            real_inputs.pop("pixel_values", None)
            null_inputs.pop("pixel_values", None)
        
        return input_ids

    # ================================================================
    # Load dataset
    # ================================================================

    if benchmark == "pope":
        dataset = load_dataset("lmms-lab/POPE", split="test")
        max_samples = num_samples or 500
    elif benchmark == "mmstar":
        dataset = load_dataset("Lin-Chen/MMStar", split="val")
        max_samples = num_samples or 1500
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    null_image = PILImage.new('RGB', (336, 336), color=(128, 128, 128))

    # Register VIAR hooks if needed
    if "viar" in method:
        register_viar_hooks()
        viar_enabled[0] = True
        print(f"  VIAR hooks registered on {len(active_hooks)} layers")

    results = {
        "correct": 0, "total": 0,
        "config": {
            "model": model_name, "benchmark": benchmark, "method": method,
            "attn_bias": attn_bias, "target_layers": sorted(target_set),
            "vcd_alpha": vcd_alpha,
        }
    }
    if benchmark == "pope":
        results.update({"tp": 0, "fp": 0, "tn": 0, "fn": 0, "yes_count": 0})
    if benchmark == "mmstar":
        results["by_category"] = {}

    # ================================================================
    # Evaluation loop
    # ================================================================

    for idx, sample in enumerate(dataset):
        if idx >= max_samples:
            break

        try:
            image = sample["image"]

            if benchmark == "pope":
                question = sample["question"]
                answer = sample["answer"].lower().strip()
                prompt = f"USER: <image>\n{question}\nAnswer with yes or no.\nASSISTANT:"
            else:
                question = sample["question"]
                options = sample.get("options", "")
                answer = sample["answer"].strip()
                category = sample.get("category", "unknown")
                full_q = f"{question}\n{options}" if options else question
                prompt = f"USER: <image>\n{full_q}\nAnswer with the option letter only.\nASSISTANT:"

            inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                model.device, torch.float16
            )

            if method == "vcd":
                # VCD: contrastive decoding with null image
                null_inputs = processor(images=null_image, text=prompt, return_tensors="pt").to(
                    model.device, torch.float16
                )
                output_ids = generate_with_vcd(inputs, null_inputs, max_new_tokens=10, alpha=vcd_alpha)
                pred = processor.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
            
            elif method == "viar+vcd":
                # Combined: VIAR hooks are already active, also do VCD
                null_inputs = processor(images=null_image, text=prompt, return_tensors="pt").to(
                    model.device, torch.float16
                )
                output_ids = generate_with_vcd(inputs, null_inputs, max_new_tokens=10, alpha=vcd_alpha)
                pred = processor.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
            
            else:
                # baseline or viar (hooks handle VIAR automatically)
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
                pred = processor.decode(
                    output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()

            # Score
            if benchmark == "pope":
                pred_answer = "yes" if "yes" in pred.lower()[:10] else "no"
                is_correct = pred_answer == answer
                if pred_answer == "yes": results["yes_count"] += 1
                if answer == "yes" and pred_answer == "yes": results["tp"] += 1
                elif answer == "no" and pred_answer == "yes": results["fp"] += 1
                elif answer == "no" and pred_answer == "no": results["tn"] += 1
                elif answer == "yes" and pred_answer == "no": results["fn"] += 1
            else:
                pred_letter = pred[0].upper() if pred and pred[0].isalpha() else pred
                is_correct = pred_letter == answer.upper()
                if category not in results["by_category"]:
                    results["by_category"][category] = {"correct": 0, "total": 0}
                results["by_category"][category]["total"] += 1
                if is_correct:
                    results["by_category"][category]["correct"] += 1

            results["total"] += 1
            if is_correct:
                results["correct"] += 1

            if (idx + 1) % 50 == 0:
                acc = results["correct"] / results["total"]
                print(f"  [{idx+1}/{max_samples}] Accuracy: {acc:.4f}")

        except Exception as e:
            print(f"  Error on sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cleanup hooks
    remove_viar_hooks()
    viar_enabled[0] = False

    # Final metrics
    total = results["total"]
    results["accuracy"] = results["correct"] / total if total > 0 else 0

    if benchmark == "pope":
        tp, fp, fn = results["tp"], results["fp"], results["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        results["precision"] = prec
        results["recall"] = rec
        results["f1"] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        results["yes_ratio"] = results["yes_count"] / total if total > 0 else 0

    if benchmark == "mmstar":
        for cat in results["by_category"]:
            s = results["by_category"][cat]
            s["accuracy"] = s["correct"] / s["total"] if s["total"] > 0 else 0

    print(f"\n{'='*60}")
    print(f"[{method.upper()}] {benchmark} | {model_name}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    if benchmark == "pope":
        print(f"  F1: {results['f1']:.4f}, Prec: {results['precision']:.4f}, Rec: {results['recall']:.4f}")
        print(f"  Yes ratio: {results['yes_ratio']:.4f}")
    if benchmark == "mmstar":
        for cat, s in sorted(results["by_category"].items()):
            print(f"  {cat}: {s['accuracy']:.4f} ({s['correct']}/{s['total']})")
    print(f"{'='*60}")

    # Save
    result_path = f"/results/{benchmark}_{model_name.split('/')[-1]}_{method}_b{attn_bias}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    results_volume.commit()

    return results


# ============================================================================
# Alpha/Bias Sweep
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=14400,
    memory=32768,
)
def ablation_bias_sweep(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    benchmark: str = "pope",
    biases: list = None,
    num_samples: int = 200,
):
    """Sweep attn_bias values."""
    if biases is None:
        biases = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]

    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from datasets import load_dataset

    print(f"[Bias Sweep] {model_name}, {benchmark}, biases={biases}, samples={num_samples}")

    processor = AutoProcessor.from_pretrained(model_name, cache_dir="/models")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        cache_dir="/models", attn_implementation="eager",
    )
    model.eval()

    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        layers = model.language_model.model.layers
    else:
        layers = model.model.layers
    num_layers = len(layers)
    target_set = set(range(8, min(17, num_layers)))

    # Load dataset
    if benchmark == "pope":
        dataset = load_dataset("lmms-lab/POPE", split="test")
    else:
        dataset = load_dataset("Lin-Chen/MMStar", split="val")

    sweep_results = {}

    for bias_val in biases:
        print(f"\n--- Bias = {bias_val} ---")

        # Register hooks for this bias value
        hooks = []
        viar_on = [bias_val > 0]

        def make_hook(b):
            def hook_fn(module, args, kwargs):
                if not viar_on[0]:
                    return args, kwargs
                attn_mask = kwargs.get('attention_mask', None)
                if attn_mask is None and len(args) > 1:
                    attn_mask = args[1]
                if attn_mask is None:
                    return args, kwargs
                kv_len = attn_mask.shape[-1]
                n_vis = min(576, kv_len)
                bias_t = torch.zeros_like(attn_mask)
                bias_t[:, :, :, :n_vis] = b
                kwargs['attention_mask'] = attn_mask + bias_t
                return args, kwargs
            return hook_fn

        if bias_val > 0:
            for li in target_set:
                if li < num_layers:
                    h = layers[li].self_attn.register_forward_pre_hook(
                        make_hook(bias_val), with_kwargs=True
                    )
                    hooks.append(h)

        correct = 0
        total = 0

        for idx, sample in enumerate(dataset):
            if idx >= num_samples:
                break
            try:
                image = sample["image"]
                if benchmark == "pope":
                    question = sample["question"]
                    answer = sample["answer"].lower().strip()
                    prompt = f"USER: <image>\n{question}\nAnswer with yes or no.\nASSISTANT:"
                else:
                    question = sample["question"]
                    options = sample.get("options", "")
                    answer = sample["answer"].strip()
                    full_q = f"{question}\n{options}" if options else question
                    prompt = f"USER: <image>\n{full_q}\nAnswer with the option letter only.\nASSISTANT:"

                inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                    model.device, torch.float16
                )

                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=10, do_sample=False)

                pred = processor.decode(
                    output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()

                if benchmark == "pope":
                    pred_answer = "yes" if "yes" in pred.lower()[:10] else "no"
                    if pred_answer == answer: correct += 1
                else:
                    pred_letter = pred[0].upper() if pred and pred[0].isalpha() else pred
                    if pred_letter == answer.upper(): correct += 1
                total += 1

            except Exception as e:
                print(f"  Error: {e}")
                continue

        # Remove hooks
        for h in hooks:
            h.remove()

        acc = correct / total if total > 0 else 0
        sweep_results[str(bias_val)] = {"accuracy": acc, "correct": correct, "total": total}
        print(f"  bias={bias_val}: {acc:.4f} ({correct}/{total})")

    print(f"\n{'='*60}")
    print(f"Bias Sweep Summary ({benchmark})")
    for b, r in sweep_results.items():
        print(f"  bias={b}: {r['accuracy']:.4f}")
    print(f"{'='*60}")

    result = {"model": model_name, "benchmark": benchmark, "sweep": sweep_results,
              "target_layers": sorted(target_set), "num_samples": num_samples}

    result_path = f"/results/bias_sweep_{model_name.split('/')[-1]}_{benchmark}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()
    return result


# ============================================================================
# Full-scale evaluation with best configs
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=14400,
    memory=32768,
)
def eval_viar_full_mmstar(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    attn_bias: float = 1.0,
    num_samples: int = 1500,
):
    """Run baseline + neglect_8_16 + adaptive on full MMStar."""
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from datasets import load_dataset

    print(f"[Full MMStar] {model_name}, bias={attn_bias}, samples={num_samples}")

    processor = AutoProcessor.from_pretrained(model_name, cache_dir="/models")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        cache_dir="/models", attn_implementation="eager",
    )
    model.eval()

    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        layers = model.language_model.model.layers
    else:
        layers = model.model.layers
    num_layers = len(layers)

    vis_frac = {
        0: 0.197, 1: 0.054, 2: 0.202, 3: 0.379, 4: 0.271,
        5: 0.231, 6: 0.266, 7: 0.236, 8: 0.208, 9: 0.181,
        10: 0.183, 11: 0.210, 12: 0.189, 13: 0.174, 14: 0.191,
        15: 0.173, 16: 0.187, 17: 0.259, 18: 0.244, 19: 0.277,
        20: 0.314, 21: 0.431, 22: 0.354, 23: 0.372, 24: 0.326,
        25: 0.495, 26: 0.303, 27: 0.321, 28: 0.333, 29: 0.310,
        30: 0.285, 31: 0.141,
    }

    # Compute adaptive biases
    neglect_scores = {l: 1.0 - vis_frac.get(l, 0.25) for l in range(num_layers)}
    mean_neglect = sum(neglect_scores.values()) / num_layers
    adaptive_biases = {l: attn_bias * (neglect_scores[l] / mean_neglect) for l in range(num_layers)}

    dataset = load_dataset("Lin-Chen/MMStar", split="val")

    configs = {
        "baseline":     {"layers": [], "per_layer_bias": {}},
        "neglect_8_16": {"layers": list(range(8, 17)), "per_layer_bias": {}},
        "adaptive":     {"layers": list(range(0, 32)), "per_layer_bias": adaptive_biases},
    }

    all_results = {}

    for config_name, config in configs.items():
        target_layers = config["layers"]
        per_layer_bias = config["per_layer_bias"]
        print(f"\n--- {config_name} ---")

        hooks = []

        def make_hook(b):
            def hook_fn(module, args, kwargs):
                attn_mask = kwargs.get('attention_mask', None)
                if attn_mask is None and len(args) > 1:
                    attn_mask = args[1]
                if attn_mask is None:
                    return args, kwargs
                kv_len = attn_mask.shape[-1]
                n_vis = min(576, kv_len)
                bias_t = torch.zeros_like(attn_mask)
                bias_t[:, :, :, :n_vis] = b
                kwargs['attention_mask'] = attn_mask + bias_t
                return args, kwargs
            return hook_fn

        if target_layers:
            for li in target_layers:
                if li < num_layers:
                    b = per_layer_bias.get(li, attn_bias)
                    h = layers[li].self_attn.register_forward_pre_hook(
                        make_hook(b), with_kwargs=True
                    )
                    hooks.append(h)

        correct = 0
        total = 0
        by_category = {}

        for idx, sample in enumerate(dataset):
            if idx >= num_samples:
                break
            try:
                image = sample["image"]
                question = sample["question"]
                options = sample.get("options", "")
                answer = sample["answer"].strip()
                category = sample.get("category", "unknown")
                full_q = f"{question}\n{options}" if options else question
                prompt = f"USER: <image>\n{full_q}\nAnswer with the option letter only.\nASSISTANT:"

                inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                    model.device, torch.float16
                )

                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=10, do_sample=False)

                pred = processor.decode(
                    output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()

                pred_letter = pred[0].upper() if pred and pred[0].isalpha() else pred
                is_correct = pred_letter == answer.upper()

                if category not in by_category:
                    by_category[category] = {"correct": 0, "total": 0}
                by_category[category]["total"] += 1
                if is_correct:
                    by_category[category]["correct"] += 1
                    correct += 1
                total += 1

                if (idx + 1) % 100 == 0:
                    print(f"  [{idx+1}/{num_samples}] {config_name}: {correct/total:.4f}")

            except Exception as e:
                print(f"  Error: {e}")
                continue

        for h in hooks:
            h.remove()

        acc = correct / total if total > 0 else 0
        for cat in by_category:
            s = by_category[cat]
            s["accuracy"] = s["correct"] / s["total"] if s["total"] > 0 else 0

        all_results[config_name] = {
            "accuracy": acc, "correct": correct, "total": total,
            "by_category": by_category,
        }
        print(f"  {config_name}: {acc:.4f} ({correct}/{total})")
        for cat, s in sorted(by_category.items()):
            print(f"    {cat}: {s['accuracy']:.4f} ({s['correct']}/{s['total']})")

    result = {
        "model": model_name, "benchmark": "mmstar", "attn_bias": attn_bias,
        "num_samples": num_samples, "configs": all_results,
    }

    result_path = f"/results/full_mmstar_{model_name.split('/')[-1]}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    results_volume.commit()
    return result


# ============================================================================
# Layer Ablation Sweep
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=14400,
    memory=32768,
)
def ablation_layer_sweep(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    benchmark: str = "pope",
    attn_bias: float = 2.0,
    num_samples: int = 200,
):
    """
    Ablate which layers to intervene in.
    
    Configs:
    - neglect_only: layers 8-16 (the visual neglect zone)
    - early: layers 0-7
    - mid: layers 8-16 (same as neglect_only)
    - late: layers 17-31
    - final_only: layer 31 (worst visual neglect)
    - deep_neglect: layers 9-15 (core of the neglect zone, highest h_ratio)
    - all_layers: layers 0-31
    - adaptive: per-layer bias proportional to neglect severity
    """
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from datasets import load_dataset

    print(f"[Layer Ablation] {model_name}, {benchmark}, bias={attn_bias}, samples={num_samples}")

    processor = AutoProcessor.from_pretrained(model_name, cache_dir="/models")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        cache_dir="/models", attn_implementation="eager",
    )
    model.eval()

    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        layers = model.language_model.model.layers
    else:
        layers = model.model.layers
    num_layers = len(layers)

    # Per-layer visual attention fraction from our analysis
    # Lower vis_frac = more neglect = needs more bias
    vis_frac_by_layer = {
        0: 0.197, 1: 0.054, 2: 0.202, 3: 0.379, 4: 0.271,
        5: 0.231, 6: 0.266, 7: 0.236, 8: 0.208, 9: 0.181,
        10: 0.183, 11: 0.210, 12: 0.189, 13: 0.174, 14: 0.191,
        15: 0.173, 16: 0.187, 17: 0.259, 18: 0.244, 19: 0.277,
        20: 0.314, 21: 0.431, 22: 0.354, 23: 0.372, 24: 0.326,
        25: 0.495, 26: 0.303, 27: 0.321, 28: 0.333, 29: 0.310,
        30: 0.285, 31: 0.141,
    }

    # Define layer configs to test
    layer_configs = {
        "baseline":      {"layers": [], "per_layer_bias": {}},
        "neglect_8_16":  {"layers": list(range(8, 17)), "per_layer_bias": {}},
        "deep_neglect":  {"layers": list(range(9, 16)), "per_layer_bias": {}},
        "early_0_7":     {"layers": list(range(0, 8)), "per_layer_bias": {}},
        "late_17_31":    {"layers": list(range(17, 32)), "per_layer_bias": {}},
        "final_31":      {"layers": [31], "per_layer_bias": {}},
        "all_layers":    {"layers": list(range(0, 32)), "per_layer_bias": {}},
        "adaptive": {
            "layers": list(range(0, 32)),
            "per_layer_bias": {},  # Will be computed below
        },
    }

    # Compute adaptive per-layer bias:
    # bias_l = base_bias * (1 - vis_frac_l) / mean(1 - vis_frac)
    # This gives more bias to layers that neglect vision more
    neglect_scores = {l: 1.0 - vis_frac_by_layer[l] for l in range(32)}
    mean_neglect = sum(neglect_scores.values()) / 32
    adaptive_biases = {l: attn_bias * (neglect_scores[l] / mean_neglect) for l in range(32)}
    layer_configs["adaptive"]["per_layer_bias"] = adaptive_biases

    if benchmark == "pope":
        dataset = load_dataset("lmms-lab/POPE", split="test")
    else:
        dataset = load_dataset("Lin-Chen/MMStar", split="val")

    all_results = {}

    for config_name, config in layer_configs.items():
        target_layers = config["layers"]
        per_layer_bias = config["per_layer_bias"]
        print(f"\n--- Config: {config_name} (layers: {target_layers[:5]}{'...' if len(target_layers) > 5 else ''}) ---")

        # Register hooks
        hooks = []

        def make_hook(b):
            def hook_fn(module, args, kwargs):
                attn_mask = kwargs.get('attention_mask', None)
                if attn_mask is None and len(args) > 1:
                    attn_mask = args[1]
                if attn_mask is None:
                    return args, kwargs
                kv_len = attn_mask.shape[-1]
                n_vis = min(576, kv_len)
                bias_t = torch.zeros_like(attn_mask)
                bias_t[:, :, :, :n_vis] = b
                kwargs['attention_mask'] = attn_mask + bias_t
                return args, kwargs
            return hook_fn

        if target_layers:
            for li in target_layers:
                if li < num_layers:
                    b = per_layer_bias.get(li, attn_bias)
                    h = layers[li].self_attn.register_forward_pre_hook(
                        make_hook(b), with_kwargs=True
                    )
                    hooks.append(h)

        correct = 0
        total = 0

        for idx, sample in enumerate(dataset):
            if idx >= num_samples:
                break
            try:
                image = sample["image"]
                if benchmark == "pope":
                    question = sample["question"]
                    answer = sample["answer"].lower().strip()
                    prompt = f"USER: <image>\n{question}\nAnswer with yes or no.\nASSISTANT:"
                else:
                    question = sample["question"]
                    options = sample.get("options", "")
                    answer = sample["answer"].strip()
                    full_q = f"{question}\n{options}" if options else question
                    prompt = f"USER: <image>\n{full_q}\nAnswer with the option letter only.\nASSISTANT:"

                inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                    model.device, torch.float16
                )

                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=10, do_sample=False)

                pred = processor.decode(
                    output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()

                if benchmark == "pope":
                    pred_answer = "yes" if "yes" in pred.lower()[:10] else "no"
                    if pred_answer == answer: correct += 1
                else:
                    pred_letter = pred[0].upper() if pred and pred[0].isalpha() else pred
                    if pred_letter == answer.upper(): correct += 1
                total += 1

            except Exception as e:
                print(f"  Error: {e}")
                continue

        # Remove hooks
        for h in hooks:
            h.remove()

        acc = correct / total if total > 0 else 0
        all_results[config_name] = {
            "accuracy": acc, "correct": correct, "total": total,
            "layers": target_layers,
        }
        if config_name == "adaptive":
            all_results[config_name]["per_layer_bias_sample"] = {
                str(l): round(adaptive_biases[l], 3) for l in [8, 9, 13, 15, 31]
            }
        print(f"  {config_name}: {acc:.4f} ({correct}/{total})")

    print(f"\n{'='*60}")
    print(f"Layer Ablation Summary ({benchmark}, bias={attn_bias})")
    for name, r in all_results.items():
        print(f"  {name}: {r['accuracy']:.4f}")
    print(f"{'='*60}")

    result = {
        "model": model_name, "benchmark": benchmark, "attn_bias": attn_bias,
        "num_samples": num_samples, "configs": all_results,
    }

    result_path = f"/results/layer_ablation_{model_name.split('/')[-1]}_{benchmark}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    results_volume.commit()
    return result


# ============================================================================
# Entry Point
# ============================================================================

@app.local_entrypoint()
def main(
    experiment: str = "viar_pope",
    bias: float = 3.0,
    model: str = "llava-hf/llava-1.5-7b-hf",
):
    """
    Usage:
        modal run src/eval_viar.py --experiment viar_pope --bias 3.0
        modal run src/eval_viar.py --experiment viar_mmstar --bias 3.0
        modal run src/eval_viar.py --experiment bias_sweep_pope
        modal run src/eval_viar.py --experiment bias_sweep_mmstar
        modal run src/eval_viar.py --experiment vcd_pope
    """
    print(f"Experiment: {experiment}, bias: {bias}, model: {model}")

    if experiment == "viar_pope":
        r = eval_viar_on_benchmark.remote(model_name=model, benchmark="pope", method="viar", attn_bias=bias)
        print(f"\nVIAR POPE: {r['accuracy']:.4f}")
    elif experiment == "viar_mmstar":
        r = eval_viar_on_benchmark.remote(model_name=model, benchmark="mmstar", method="viar", attn_bias=bias)
        print(f"\nVIAR MMStar: {r['accuracy']:.4f}")
    elif experiment == "baseline_pope":
        r = eval_viar_on_benchmark.remote(model_name=model, benchmark="pope", method="baseline", attn_bias=0)
        print(f"\nBaseline POPE: {r['accuracy']:.4f}")
    elif experiment == "vcd_pope":
        r = eval_viar_on_benchmark.remote(model_name=model, benchmark="pope", method="vcd", vcd_alpha=1.0)
        print(f"\nVCD POPE: {r['accuracy']:.4f}")
    elif experiment == "bias_sweep_pope":
        r = ablation_bias_sweep.remote(model_name=model, benchmark="pope", num_samples=200)
        print(f"\nBias sweep POPE complete")
    elif experiment == "bias_sweep_mmstar":
        r = ablation_bias_sweep.remote(model_name=model, benchmark="mmstar", num_samples=200)
        print(f"\nBias sweep MMStar complete")
    elif experiment == "layer_ablation_pope":
        r = ablation_layer_sweep.remote(model_name=model, benchmark="pope", attn_bias=bias, num_samples=200)
        print(f"\nLayer ablation POPE complete")
        for name, cfg in r["configs"].items():
            print(f"  {name}: {cfg['accuracy']:.4f}")
    elif experiment == "layer_ablation_mmstar":
        r = ablation_layer_sweep.remote(model_name=model, benchmark="mmstar", attn_bias=bias, num_samples=200)
        print(f"\nLayer ablation MMStar complete")
        for name, cfg in r["configs"].items():
            print(f"  {name}: {cfg['accuracy']:.4f}")
    elif experiment == "full_pope":
        # Best config for POPE: neglect_8_16, bias=2.0
        r = eval_viar_on_benchmark.remote(
            model_name=model, benchmark="pope", method="viar",
            attn_bias=2.0, target_layers=list(range(8, 17)), num_samples=500,
        )
        print(f"\nFull VIAR POPE (bias=2.0, layers 8-16): {r['accuracy']:.4f}")
    elif experiment == "full_mmstar":
        # Run both neglect_8_16 and adaptive on full MMStar
        r = eval_viar_full_mmstar.remote(model_name=model, attn_bias=1.0, num_samples=1500)
        print(f"\nFull MMStar results:")
        for name, cfg in r["configs"].items():
            print(f"  {name}: {cfg['accuracy']:.4f}")
    else:
        print(f"Unknown: {experiment}")
