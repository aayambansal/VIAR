"""
Modal-based evaluation harness for VIAR experiments.

Runs VLM evaluations on cloud GPUs with:
- Baseline (vanilla decoding)
- VCD (Visual Contrastive Decoding)  
- VIAR (our method)
- VIAR + VCD (combined)

Benchmarks: POPE, MME, MMStar, MMMU, MM-Vet
Models: LLaVA-1.5-7B, LLaVA-1.5-13B, InternVL2-8B
"""

import modal
import json
import os

# ============================================================================
# Modal App Setup
# ============================================================================

app = modal.App("viar-eccv-eval")

# Container image with all dependencies
vlm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
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
        "lmms-eval>=0.2.0",
    )
    .env({"HF_HOME": "/models", "TRANSFORMERS_CACHE": "/models"})
)

# Volume for caching models
model_volume = modal.Volume.from_name("viar-model-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("viar-results", create_if_missing=True)


# ============================================================================
# POPE Evaluation (Object Hallucination)
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=7200,
    memory=32768,
)
def eval_pope(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    method: str = "baseline",  # "baseline", "viar", "vcd", "viar+vcd"
    viar_config: dict = None,
    split: str = "random",  # "random", "popular", "adversarial"
):
    """Evaluate on POPE (Polling-based Object Probing Evaluation)."""
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from datasets import load_dataset
    from PIL import Image
    import io
    import requests

    print(f"[POPE] Model: {model_name}, Method: {method}, Split: {split}")

    # Load model
    processor = AutoProcessor.from_pretrained(model_name, cache_dir="/models")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/models",
    )
    model.eval()

    # Load POPE dataset
    # POPE has 3 splits: random, popular, adversarial
    # Each has ~500 yes/no questions about object presence
    dataset = load_dataset("lmms-lab/POPE", split="test")

    results = {"correct": 0, "total": 0, "yes_count": 0, "no_count": 0,
               "tp": 0, "fp": 0, "tn": 0, "fn": 0, "predictions": []}

    for idx, sample in enumerate(dataset):
        if idx >= 500:  # Cap at 500 per split for speed
            break

        try:
            image = sample["image"]
            question = sample["question"]
            answer = sample["answer"].lower().strip()

            # Format prompt for LLaVA
            prompt = f"USER: <image>\n{question}\nAnswer with yes or no.\nASSISTANT:"

            if method == "baseline":
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                    model.device, torch.float16
                )
                with torch.no_grad():
                    output = model.generate(
                        **inputs, max_new_tokens=10, do_sample=False
                    )
                pred_text = processor.decode(
                    output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).lower().strip()

            elif method == "viar":
                # Import VIAR and apply
                from viar import VIARConfig, apply_viar_to_llava
                config = VIARConfig(**(viar_config or {}))
                pred_text, diag = apply_viar_to_llava(
                    model, processor, image, prompt, config=config,
                    max_new_tokens=10
                )
                pred_text = pred_text.lower().strip()

            # Parse yes/no from prediction
            pred_answer = "yes" if "yes" in pred_text[:10] else "no"

            # Update metrics
            results["total"] += 1
            if pred_answer == answer:
                results["correct"] += 1
            if pred_answer == "yes":
                results["yes_count"] += 1
            else:
                results["no_count"] += 1

            # Confusion matrix
            if answer == "yes" and pred_answer == "yes":
                results["tp"] += 1
            elif answer == "no" and pred_answer == "yes":
                results["fp"] += 1
            elif answer == "no" and pred_answer == "no":
                results["tn"] += 1
            elif answer == "yes" and pred_answer == "no":
                results["fn"] += 1

            results["predictions"].append({
                "idx": idx, "question": question, "gt": answer,
                "pred": pred_answer, "raw": pred_text[:50]
            })

            if (idx + 1) % 50 == 0:
                acc = results["correct"] / results["total"]
                print(f"  [{idx+1}/{min(len(dataset), 500)}] Accuracy: {acc:.4f}")

        except Exception as e:
            print(f"  Error on sample {idx}: {e}")
            continue

    # Compute final metrics
    total = results["total"]
    results["accuracy"] = results["correct"] / total if total > 0 else 0
    precision = results["tp"] / (results["tp"] + results["fp"]) if (results["tp"] + results["fp"]) > 0 else 0
    recall = results["tp"] / (results["tp"] + results["fn"]) if (results["tp"] + results["fn"]) > 0 else 0
    results["precision"] = precision
    results["recall"] = recall
    results["f1"] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    results["yes_ratio"] = results["yes_count"] / total if total > 0 else 0

    print(f"\n[POPE Results] Model: {model_name}, Method: {method}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  Yes ratio: {results['yes_ratio']:.4f}")

    # Save results
    result_path = f"/results/pope_{model_name.split('/')[-1]}_{method}_{split}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    results_volume.commit()

    return results


# ============================================================================
# MME Evaluation (Perception + Cognition)
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=7200,
    memory=32768,
)
def eval_mme(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    method: str = "baseline",
    viar_config: dict = None,
):
    """Evaluate on MME benchmark."""
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from datasets import load_dataset

    print(f"[MME] Model: {model_name}, Method: {method}")

    processor = AutoProcessor.from_pretrained(model_name, cache_dir="/models")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/models",
    )
    model.eval()

    # Load MME dataset
    dataset = load_dataset("lmms-lab/MME", split="test")

    category_scores = {}
    total_correct = 0
    total_count = 0

    for idx, sample in enumerate(dataset):
        if idx >= 1000:  # Cap for budget
            break

        try:
            image = sample["image"]
            question = sample["question"]
            answer = sample["answer"].strip()
            category = sample.get("category", "unknown")

            prompt = f"USER: <image>\n{question}\nASSISTANT:"

            inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                model.device, torch.float16
            )

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=50, do_sample=False)

            pred = processor.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            # MME uses exact match for yes/no
            is_correct = pred.lower().startswith(answer.lower())

            if category not in category_scores:
                category_scores[category] = {"correct": 0, "total": 0}
            category_scores[category]["total"] += 1
            if is_correct:
                category_scores[category]["correct"] += 1
                total_correct += 1
            total_count += 1

            if (idx + 1) % 100 == 0:
                print(f"  [{idx+1}] Running accuracy: {total_correct/total_count:.4f}")

        except Exception as e:
            print(f"  Error on sample {idx}: {e}")
            continue

    # Compute per-category and overall scores
    results = {
        "model": model_name,
        "method": method,
        "overall_accuracy": total_correct / total_count if total_count > 0 else 0,
        "total_correct": total_correct,
        "total_count": total_count,
        "category_scores": {
            cat: {**scores, "accuracy": scores["correct"] / scores["total"] if scores["total"] > 0 else 0}
            for cat, scores in category_scores.items()
        },
    }

    # MME computes perception and cognition scores
    perception_cats = ["existence", "count", "position", "color", "posters",
                       "celebrity", "scene", "landmark", "artwork", "OCR"]
    cognition_cats = ["commonsense_reasoning", "numerical_calculation",
                      "text_translation", "code_reasoning"]

    perc_score = sum(
        category_scores.get(c, {}).get("correct", 0)
        for c in perception_cats
    )
    cog_score = sum(
        category_scores.get(c, {}).get("correct", 0)
        for c in cognition_cats
    )
    results["perception_score"] = perc_score
    results["cognition_score"] = cog_score
    results["total_score"] = perc_score + cog_score

    print(f"\n[MME Results] Model: {model_name}, Method: {method}")
    print(f"  Perception: {perc_score}")
    print(f"  Cognition: {cog_score}")
    print(f"  Total: {perc_score + cog_score}")

    result_path = f"/results/mme_{model_name.split('/')[-1]}_{method}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    results_volume.commit()

    return results


# ============================================================================
# MMStar Evaluation (Vision-Indispensable)
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=7200,
    memory=32768,
)
def eval_mmstar(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    method: str = "baseline",
    viar_config: dict = None,
):
    """
    Evaluate on MMStar — the most diagnostic benchmark for our work.
    1500 vision-indispensable samples that prevent text-shortcut reliance.
    """
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from datasets import load_dataset

    print(f"[MMStar] Model: {model_name}, Method: {method}")

    processor = AutoProcessor.from_pretrained(model_name, cache_dir="/models")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/models",
    )
    model.eval()

    # Load MMStar
    dataset = load_dataset("Lin-Chen/MMStar", split="val")

    results = {"correct": 0, "total": 0, "by_category": {}, "predictions": []}

    for idx, sample in enumerate(dataset):
        try:
            image = sample["image"]
            question = sample["question"]
            options = sample.get("options", "")
            answer = sample["answer"].strip()
            category = sample.get("category", "unknown")

            # Format as multiple choice
            full_question = f"{question}\n{options}" if options else question
            prompt = f"USER: <image>\n{full_question}\nAnswer with the option letter only.\nASSISTANT:"

            inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                model.device, torch.float16
            )

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=10, do_sample=False)

            pred = processor.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            # Extract option letter
            pred_letter = pred[0].upper() if pred and pred[0].isalpha() else pred
            is_correct = pred_letter == answer.upper()

            results["total"] += 1
            if is_correct:
                results["correct"] += 1

            if category not in results["by_category"]:
                results["by_category"][category] = {"correct": 0, "total": 0}
            results["by_category"][category]["total"] += 1
            if is_correct:
                results["by_category"][category]["correct"] += 1

            results["predictions"].append({
                "idx": idx, "gt": answer, "pred": pred_letter,
                "category": category
            })

            if (idx + 1) % 100 == 0:
                acc = results["correct"] / results["total"]
                print(f"  [{idx+1}/{len(dataset)}] Accuracy: {acc:.4f}")

        except Exception as e:
            print(f"  Error on sample {idx}: {e}")
            continue

    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    for cat in results["by_category"]:
        s = results["by_category"][cat]
        s["accuracy"] = s["correct"] / s["total"] if s["total"] > 0 else 0

    print(f"\n[MMStar Results] Model: {model_name}, Method: {method}")
    print(f"  Overall Accuracy: {results['accuracy']:.4f}")
    for cat, s in results["by_category"].items():
        print(f"  {cat}: {s['accuracy']:.4f} ({s['correct']}/{s['total']})")

    result_path = f"/results/mmstar_{model_name.split('/')[-1]}_{method}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    results_volume.commit()

    return results


# ============================================================================
# Attention Analysis (for paper figures)
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=3600,
    memory=32768,
)
def analyze_attention_patterns(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    num_samples: int = 50,
):
    """
    Analyze cross-modal attention patterns in VLMs.
    Produces data for Figure 1 of the paper: attention entropy analysis.
    """
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from datasets import load_dataset
    import numpy as np

    print(f"[Attention Analysis] Model: {model_name}, Samples: {num_samples}")

    processor = AutoProcessor.from_pretrained(model_name, cache_dir="/models")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/models",
        attn_implementation="eager",  # Need full attention weights
    )
    model.eval()

    # Use POPE samples for analysis
    dataset = load_dataset("lmms-lab/POPE", split="test")

    all_layer_entropies = []  # List of dicts: {layer, h_vis, h_text, h_ratio}

    for idx, sample in enumerate(dataset):
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
                outputs = model(
                    **inputs,
                    output_attentions=True,
                    return_dict=True,
                )

            # Analyze attention per layer
            for layer_idx, attn_weights in enumerate(outputs.attentions):
                # attn_weights: [batch, heads, seq_len, seq_len]
                seq_len = attn_weights.shape[-1]

                # Estimate visual token positions (first 576 for LLaVA)
                n_vis = min(576, seq_len)
                vis_mask = torch.zeros(seq_len, dtype=torch.bool, device=attn_weights.device)
                txt_mask = torch.zeros(seq_len, dtype=torch.bool, device=attn_weights.device)
                vis_mask[:n_vis] = True
                txt_mask[n_vis:] = True

                # Cast to fp32 for numerical stability in entropy computation
                attn_fp32 = attn_weights.float()

                # Attention FROM text tokens TO visual/text tokens
                text_to_vis_attn = attn_fp32[:, :, txt_mask, :][:, :, :, vis_mask]
                text_to_txt_attn = attn_fp32[:, :, txt_mask, :][:, :, :, txt_mask]

                # Mean attention weight (how much text attends to vision vs text)
                mean_vis_attn = text_to_vis_attn.mean().item()
                mean_txt_attn = text_to_txt_attn.mean().item()

                # Entropy of attention over visual tokens (from text positions)
                vis_attn_norm = text_to_vis_attn / (text_to_vis_attn.sum(dim=-1, keepdim=True) + 1e-10)
                h_vis = -(vis_attn_norm * torch.log(vis_attn_norm + 1e-10)).sum(dim=-1).mean().item()

                # Entropy of attention over text tokens (from text positions)
                txt_attn_norm = text_to_txt_attn / (text_to_txt_attn.sum(dim=-1, keepdim=True) + 1e-10)
                h_text = -(txt_attn_norm * torch.log(txt_attn_norm + 1e-10)).sum(dim=-1).mean().item()

                all_layer_entropies.append({
                    "sample_idx": idx,
                    "layer": layer_idx,
                    "h_vis": h_vis,
                    "h_text": h_text,
                    "h_ratio": h_vis / (h_text + 1e-10),
                    "mean_vis_attn": mean_vis_attn,
                    "mean_txt_attn": mean_txt_attn,
                    "vis_attn_frac": mean_vis_attn / (mean_vis_attn + mean_txt_attn + 1e-10),
                })

            if (idx + 1) % 10 == 0:
                print(f"  Analyzed {idx+1}/{num_samples} samples")

        except Exception as e:
            print(f"  Error on sample {idx}: {e}")
            continue

    # Aggregate by layer
    layer_stats = {}
    for entry in all_layer_entropies:
        l = entry["layer"]
        if l not in layer_stats:
            layer_stats[l] = {"h_vis": [], "h_text": [], "h_ratio": [],
                              "vis_attn_frac": []}
        layer_stats[l]["h_vis"].append(entry["h_vis"])
        layer_stats[l]["h_text"].append(entry["h_text"])
        layer_stats[l]["h_ratio"].append(entry["h_ratio"])
        layer_stats[l]["vis_attn_frac"].append(entry["vis_attn_frac"])

    aggregated = {}
    for l, stats in layer_stats.items():
        aggregated[l] = {
            "h_vis_mean": float(np.mean(stats["h_vis"])),
            "h_vis_std": float(np.std(stats["h_vis"])),
            "h_text_mean": float(np.mean(stats["h_text"])),
            "h_text_std": float(np.std(stats["h_text"])),
            "h_ratio_mean": float(np.mean(stats["h_ratio"])),
            "h_ratio_std": float(np.std(stats["h_ratio"])),
            "vis_attn_frac_mean": float(np.mean(stats["vis_attn_frac"])),
            "vis_attn_frac_std": float(np.std(stats["vis_attn_frac"])),
        }

    print(f"\n[Attention Analysis Summary]")
    for l in sorted(aggregated.keys()):
        s = aggregated[l]
        print(f"  Layer {l:2d}: H_vis={s['h_vis_mean']:.3f}+-{s['h_vis_std']:.3f}, "
              f"H_text={s['h_text_mean']:.3f}+-{s['h_text_std']:.3f}, "
              f"ratio={s['h_ratio_mean']:.3f}, vis_frac={s['vis_attn_frac_mean']:.4f}")

    result = {
        "model": model_name,
        "num_samples": num_samples,
        "per_sample": all_layer_entropies,
        "aggregated_by_layer": {str(k): v for k, v in aggregated.items()},
    }

    result_path = f"/results/attention_analysis_{model_name.split('/')[-1]}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()

    return result


# ============================================================================
# Text-Only Baseline (measures text prior exploitation)
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=3600,
    memory=32768,
)
def eval_text_only_baseline(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    benchmark: str = "pope",
    num_samples: int = 500,
):
    """
    Run VLM with text-only input (no image or gray image).
    Measures how much the model relies on text priors alone.
    Gap between text-only and full performance = visual contribution.
    """
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from datasets import load_dataset
    from PIL import Image as PILImage

    print(f"[Text-Only] Model: {model_name}, Benchmark: {benchmark}")

    processor = AutoProcessor.from_pretrained(model_name, cache_dir="/models")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/models",
    )
    model.eval()

    if benchmark == "pope":
        dataset = load_dataset("lmms-lab/POPE", split="test")
    elif benchmark == "mmstar":
        dataset = load_dataset("Lin-Chen/MMStar", split="val")
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # Create gray null image
    null_image = PILImage.new('RGB', (336, 336), color=(128, 128, 128))

    correct = 0
    total = 0

    for idx, sample in enumerate(dataset):
        if idx >= num_samples:
            break

        try:
            question = sample["question"]
            answer = sample["answer"].strip().lower()

            prompt = f"USER: <image>\n{question}\nASSISTANT:"

            inputs = processor(images=null_image, text=prompt, return_tensors="pt").to(
                model.device, torch.float16
            )

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=10, do_sample=False)

            pred = processor.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).lower().strip()

            if benchmark == "pope":
                pred_answer = "yes" if "yes" in pred[:10] else "no"
                if pred_answer == answer:
                    correct += 1
            elif benchmark == "mmstar":
                pred_letter = pred[0].upper() if pred and pred[0].isalpha() else pred
                if pred_letter == answer.upper():
                    correct += 1

            total += 1

            if (idx + 1) % 100 == 0:
                print(f"  [{idx+1}] Text-only accuracy: {correct/total:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    accuracy = correct / total if total > 0 else 0
    print(f"\n[Text-Only Results] {benchmark}: {accuracy:.4f} ({correct}/{total})")

    result = {
        "model": model_name,
        "benchmark": benchmark,
        "method": "text_only",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }

    result_path = f"/results/text_only_{model_name.split('/')[-1]}_{benchmark}.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()

    return result


# ============================================================================
# Entry Points
# ============================================================================

@app.local_entrypoint()
def main(
    experiment: str = "attention_analysis",
    model: str = "llava-hf/llava-1.5-7b-hf",
    method: str = "baseline",
):
    """
    Run experiments from CLI.
    
    Usage:
        modal run src/eval_modal.py --experiment attention_analysis
        modal run src/eval_modal.py --experiment pope --method baseline
        modal run src/eval_modal.py --experiment pope --method viar
        modal run src/eval_modal.py --experiment mmstar --method baseline
        modal run src/eval_modal.py --experiment text_only
    """
    print(f"Running experiment: {experiment}, model: {model}, method: {method}")

    if experiment == "attention_analysis":
        result = analyze_attention_patterns.remote(model_name=model, num_samples=50)
        print(f"\nAttention analysis complete. Results saved.")

    elif experiment == "pope":
        result = eval_pope.remote(model_name=model, method=method)
        print(f"\nPOPE complete. Accuracy: {result['accuracy']:.4f}")

    elif experiment == "mmstar":
        result = eval_mmstar.remote(model_name=model, method=method)
        print(f"\nMMStar complete. Accuracy: {result['accuracy']:.4f}")

    elif experiment == "mme":
        result = eval_mme.remote(model_name=model, method=method)
        print(f"\nMME complete. Total: {result['total_score']}")

    elif experiment == "text_only":
        for bench in ["pope", "mmstar"]:
            result = eval_text_only_baseline.remote(
                model_name=model, benchmark=bench
            )
            print(f"\nText-only {bench}: {result['accuracy']:.4f}")

    elif experiment == "all_baselines":
        # Run all baseline evaluations
        futures = []
        for bench_fn, bench_name in [
            (eval_pope, "pope"),
            (eval_mmstar, "mmstar"),
            (eval_mme, "mme"),
        ]:
            futures.append(bench_fn.spawn(model_name=model, method="baseline"))
        
        # Also run text-only baselines
        for bench in ["pope", "mmstar"]:
            futures.append(
                eval_text_only_baseline.spawn(model_name=model, benchmark=bench)
            )

        # Collect results
        for f in futures:
            result = f.get()
            print(f"  Completed: {result.get('model', model)}")

    else:
        print(f"Unknown experiment: {experiment}")
        print("Available: attention_analysis, pope, mmstar, mme, text_only, all_baselines")
