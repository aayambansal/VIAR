"""
Multi-model experiments for VIAR paper revision.

Covers:
1. Attention analysis on InstructBLIP-7B and LLaVA-v1.6-Vicuna-7B
2. VIAR evaluation on both models (POPE + MMStar subset)
3. Discovers if the visual neglect zone generalizes across architectures
"""

import modal
import json
import os

app = modal.App("viar-multimodel-v2")

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
    timeout=14400,
    memory=32768,
)
def instructblip_attention_analysis(
    model_name: str = "Salesforce/instructblip-vicuna-7b",
    num_samples: int = 50,
):
    """
    Run per-layer attention analysis on InstructBLIP-Vicuna-7B.
    InstructBLIP uses Q-Former with only 32 query tokens.
    """
    import torch
    import numpy as np
    from datasets import load_dataset
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

    print(f"[Attention Analysis] Model: {model_name}")
    
    processor = InstructBlipProcessor.from_pretrained(model_name, cache_dir="/models")
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        cache_dir="/models",
    )
    model.eval()

    # InstructBLIP: language_model is the LLM (Vicuna), qformer is separate
    layers = model.language_model.model.layers
    num_layers = len(layers)
    n_vis_tokens = 32  # Q-Former produces 32 query tokens
    
    print(f"  Found {num_layers} layers, n_vis_tokens={n_vis_tokens}")

    dataset = load_dataset("lmms-lab/POPE", split="test")

    layer_stats = {l: {"vis_frac": [], "h_vis": [], "h_text": []} for l in range(num_layers)}

    for idx, sample in enumerate(dataset):
        if idx >= num_samples:
            break

        try:
            image = sample["image"]
            question = sample["question"]
            prompt = f"{question} Answer with yes or no."

            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            # InstructBLIP returns language_model_attentions
            attentions = outputs.language_model_attentions
            if attentions is None:
                print(f"  Sample {idx}: no language_model_attentions, trying .attentions")
                attentions = getattr(outputs, 'attentions', None)
            
            if attentions is None:
                print(f"  Sample {idx}: no attentions at all, skipping")
                continue

            seq_len = attentions[0].shape[-1]
            n_vis = min(n_vis_tokens, seq_len)

            for l_idx in range(min(num_layers, len(attentions))):
                attn = attentions[l_idx][0].float().cpu()

                vis_attn = attn[:, :, :n_vis].sum(dim=-1).mean().item()
                total_attn = attn.sum(dim=-1).mean().item()
                vis_frac = vis_attn / total_attn if total_attn > 0 else 0

                vis_dist = attn[:, :, :n_vis].mean(dim=0).mean(dim=0)
                text_dist = attn[:, :, n_vis:].mean(dim=0).mean(dim=0)

                vis_dist = vis_dist / (vis_dist.sum() + 1e-10)
                text_dist = text_dist / (text_dist.sum() + 1e-10)

                h_vis = -(vis_dist * (vis_dist + 1e-10).log()).sum().item()
                h_text = -(text_dist * (text_dist + 1e-10).log()).sum().item()

                layer_stats[l_idx]["vis_frac"].append(vis_frac)
                layer_stats[l_idx]["h_vis"].append(h_vis)
                layer_stats[l_idx]["h_text"].append(h_text)

            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{num_samples}] processed")

        except Exception as e:
            print(f"  Error sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate
    result = {
        "model": model_name,
        "num_samples": num_samples,
        "num_layers": num_layers,
        "n_vis_tokens": n_vis_tokens,
        "by_layer": {},
    }

    for l in range(num_layers):
        if layer_stats[l]["vis_frac"]:
            vf = np.mean(layer_stats[l]["vis_frac"])
            hv = np.mean(layer_stats[l]["h_vis"])
            ht = np.mean(layer_stats[l]["h_text"])
            result["by_layer"][str(l)] = {
                "vis_attn_frac_mean": round(float(vf), 4),
                "h_vis_mean": round(float(hv), 3),
                "h_text_mean": round(float(ht), 3),
                "h_ratio_mean": round(float(hv / ht) if ht > 0 else 0, 3),
            }

    # Identify neglect zone
    sorted_layers = sorted(result["by_layer"].items(), key=lambda x: x[1]["vis_attn_frac_mean"])
    bottom_quarter = sorted_layers[:num_layers // 4]
    neglect_layers = sorted([int(l) for l, _ in bottom_quarter])
    result["neglect_zone"] = neglect_layers

    print(f"\n{'='*60}")
    print(f"Attention Analysis: {model_name}")
    for l in range(num_layers):
        if str(l) in result["by_layer"]:
            d = result["by_layer"][str(l)]
            marker = " <-- NEGLECT" if l in neglect_layers else ""
            print(f"  L{l:2d}: vis_frac={d['vis_attn_frac_mean']:.3f}, h_ratio={d['h_ratio_mean']:.3f}{marker}")
    print(f"  Neglect zone: {neglect_layers}")
    print(f"{'='*60}")

    path = "/results/attn_analysis_instructblip.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()

    return result


@app.local_entrypoint()
def main(experiment: str = "attn_instructblip"):
    """
    Usage:
      modal run src/eval_multimodel.py --experiment attn_instructblip
    """
    if experiment == "attn_instructblip":
        r = instructblip_attention_analysis.remote()
        print(f"\nNeglect zone: {r.get('neglect_zone', 'N/A')}")
    else:
        print(f"Unknown experiment: {experiment}")
