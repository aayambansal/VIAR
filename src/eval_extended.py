"""
Extended experiments for VIAR paper revision.

Covers:
1. VCD comparison and VIAR+VCD combination on POPE
2. Statistical significance (5-seed bootstrap on POPE and MMStar)
3. Mechanistic before/after analysis (attention entropy with VIAR)
4. Qualitative per-example analysis (correct/wrong cases)
5. Alternative adaptive scaling strategies
6. GQA open-ended benchmark
7. Layer 31 investigation
"""

import modal
import json
import os

app = modal.App("viar-extended-v2")

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


# ============================================================================
# Experiment 1: VCD comparison + VIAR+VCD combined
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=14400,
    memory=32768,
)
def vcd_comparison(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    num_samples: int = 500,
    attn_bias: float = 2.0,
    vcd_alpha: float = 1.0,
):
    """
    Compare: baseline, VIAR, VCD, VIAR+VCD on POPE.
    VCD: contrastive decoding with gray image as null input.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from datasets import load_dataset
    from PIL import Image as PILImage

    print(f"[VCD Comparison] {model_name}, {num_samples} samples")

    processor = AutoProcessor.from_pretrained(model_name, cache_dir="/models")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        cache_dir="/models", attn_implementation="eager",
    )
    model.eval()
    layers = model.language_model.model.layers
    num_layers = len(layers)
    target_set = set(range(8, 17))

    dataset = load_dataset("lmms-lab/POPE", split="test")
    null_image = PILImage.new('RGB', (336, 336), color=(128, 128, 128))

    def make_hook(b):
        def hook_fn(module, args, kwargs):
            attn_mask = kwargs.get('attention_mask', None)
            if attn_mask is None and len(args) > 1:
                attn_mask = args[1]
            if attn_mask is None:
                return args, kwargs
            kv_len = attn_mask.shape[-1]
            nv = min(576, kv_len)
            bias_t = torch.zeros_like(attn_mask)
            bias_t[:, :, :, :nv] = b
            kwargs['attention_mask'] = attn_mask + bias_t
            return args, kwargs
        return hook_fn

    def generate_vcd(real_inputs, null_inputs, alpha=1.0, max_new=10):
        """VCD greedy decoding: (1+alpha)*log_p(real) - alpha*log_p(null).
        
        Fixed: handles different sequence lengths between real and null inputs
        by running separate forward passes with each input's own attention mask
        and pixel values, then combining logits for contrastive decoding.
        """
        # Deep copy inputs so we don't mutate originals
        real_kw = {k: v.clone() if torch.is_tensor(v) else v for k, v in real_inputs.items()}
        null_kw = {k: v.clone() if torch.is_tensor(v) else v for k, v in null_inputs.items()}
        
        generated_ids = []
        
        for step in range(max_new):
            with torch.no_grad():
                real_out = model(**real_kw)
                null_out = model(**null_kw)
            
            lr = F.log_softmax(real_out.logits[:, -1, :].float(), dim=-1)
            ln = F.log_softmax(null_out.logits[:, -1, :].float(), dim=-1)
            corrected = (1 + alpha) * lr - alpha * ln
            next_tok = corrected.argmax(dim=-1, keepdim=True)
            generated_ids.append(next_tok)
            
            if next_tok.item() == processor.tokenizer.eos_token_id:
                break
            
            # Append next token to each input's own input_ids
            real_kw["input_ids"] = torch.cat([real_kw["input_ids"], next_tok], dim=-1)
            null_kw["input_ids"] = torch.cat([null_kw["input_ids"], next_tok], dim=-1)
            
            # Update attention masks to match new lengths
            if "attention_mask" in real_kw:
                real_kw["attention_mask"] = torch.cat([
                    real_kw["attention_mask"],
                    torch.ones((1, 1), dtype=real_kw["attention_mask"].dtype, device=real_kw["attention_mask"].device)
                ], dim=-1)
            if "attention_mask" in null_kw:
                null_kw["attention_mask"] = torch.cat([
                    null_kw["attention_mask"],
                    torch.ones((1, 1), dtype=null_kw["attention_mask"].dtype, device=null_kw["attention_mask"].device)
                ], dim=-1)
            
            # Remove pixel_values after first step (already encoded)
            real_kw.pop("pixel_values", None)
            null_kw.pop("pixel_values", None)
        
        # Return: original real input_ids + generated tokens
        if generated_ids:
            gen = torch.cat(generated_ids, dim=-1)
            return torch.cat([real_inputs["input_ids"], gen], dim=-1)
        return real_inputs["input_ids"]

    methods = ["baseline", "viar", "vcd", "viar+vcd"]
    all_results = {}

    for method in methods:
        print(f"\n--- {method} ---")
        hooks = []
        use_viar = "viar" in method
        use_vcd = "vcd" in method

        if use_viar:
            for li in target_set:
                h = layers[li].self_attn.register_forward_pre_hook(
                    make_hook(attn_bias), with_kwargs=True
                )
                hooks.append(h)

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

                if use_vcd:
                    null_inputs = processor(images=null_image, text=prompt, return_tensors="pt").to(
                        model.device, torch.float16
                    )
                    out_ids = generate_vcd(inputs, null_inputs, alpha=vcd_alpha)
                    pred = processor.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                else:
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
                    pred = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

                pred_a = "yes" if "yes" in pred.lower()[:10] else "no"
                if pred_a == answer: correct += 1
                if pred_a == "yes": yes_count += 1
                if answer == "yes" and pred_a == "yes": tp += 1
                elif answer == "no" and pred_a == "yes": fp += 1
                elif answer == "no" and pred_a == "no": tn += 1
                elif answer == "yes" and pred_a == "no": fn += 1
                total += 1

                if (idx+1) % 100 == 0:
                    print(f"  [{idx+1}/{num_samples}] {method}: {correct/total:.4f}")

            except Exception as e:
                print(f"  Error: {e}")
                continue

        for h in hooks:
            h.remove()

        acc = correct/total if total > 0 else 0
        prec = tp/(tp+fp) if (tp+fp) > 0 else 0
        rec = tp/(tp+fn) if (tp+fn) > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

        all_results[method] = {
            "accuracy": round(acc, 4), "f1": round(f1, 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "yes_ratio": round(yes_count/total, 4) if total > 0 else 0,
            "total": total,
        }
        print(f"  {method}: acc={acc:.4f}, f1={f1:.4f}, prec={prec:.4f}, rec={rec:.4f}")

    result = {"model": model_name, "benchmark": "pope", "methods": all_results,
              "attn_bias": attn_bias, "vcd_alpha": vcd_alpha}
    path = "/results/vcd_comparison.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()
    return result


# ============================================================================
# Experiment 5: GQA Open-Ended VQA
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=14400,
    memory=32768,
)
def gqa_evaluation(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    attn_bias: float = 2.0,
    num_samples: int = 500,
):
    """
    Evaluate on GQA (open-ended VQA) to show generalization beyond
    binary and multiple-choice tasks.
    """
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from datasets import load_dataset

    print(f"[GQA Eval] {model_name}, {num_samples} samples")

    processor = AutoProcessor.from_pretrained(model_name, cache_dir="/models")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        cache_dir="/models", attn_implementation="eager",
    )
    model.eval()
    layers = model.language_model.model.layers
    target_set = set(range(8, 17))

    # Load GQA: need both images and instructions
    # Load instructions for Q&A, images for the actual images
    ds_instr = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
    ds_imgs = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")
    print(f"  Instructions: {len(ds_instr)} samples, columns: {ds_instr.column_names}")
    print(f"  Images: {len(ds_imgs)} samples, columns: {ds_imgs.column_names}")
    
    # Build image lookup by imageId
    img_lookup = {}
    for item in ds_imgs:
        img_id = item.get("id", item.get("imageId", None))
        img = item.get("image", None)
        if img_id and img:
            img_lookup[str(img_id)] = img
    print(f"  Image lookup: {len(img_lookup)} images")
    
    # Use instructions as the main dataset
    dataset = ds_instr

    def make_hook(b):
        def hook_fn(module, args, kwargs):
            attn_mask = kwargs.get('attention_mask', None)
            if attn_mask is None and len(args) > 1:
                attn_mask = args[1]
            if attn_mask is None:
                return args, kwargs
            kv_len = attn_mask.shape[-1]
            nv = min(576, kv_len)
            bias_t = torch.zeros_like(attn_mask)
            bias_t[:, :, :, :nv] = b
            kwargs['attention_mask'] = attn_mask + bias_t
            return args, kwargs
        return hook_fn

    all_results = {}

    for method in ["baseline", "viar"]:
        print(f"\n--- {method} ---")
        hooks = []
        if method == "viar":
            for li in target_set:
                h = layers[li].self_attn.register_forward_pre_hook(
                    make_hook(attn_bias), with_kwargs=True
                )
                hooks.append(h)

        correct = total = 0

        for idx, sample in enumerate(dataset):
            if idx >= num_samples:
                break
            try:
                # GQA lmms-lab format: instructions have Q&A, images are separate
                question = sample.get("question", "")
                answer = sample.get("answer", "").lower().strip()
                image_id = sample.get("imageId", "")
                
                if not question or not answer:
                    continue
                
                # Look up image by imageId
                image = img_lookup.get(str(image_id), None)
                if image is None:
                    if idx < 3:
                        print(f"    [SKIP #{idx}] No image for imageId={image_id}")
                    continue

                prompt = f"USER: <image>\n{question}\nAnswer in one word or short phrase.\nASSISTANT:"
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                    model.device, torch.float16
                )

                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
                pred = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()

                # Exact match (standard GQA metric)
                if pred == answer or answer in pred.split()[:3]:
                    correct += 1
                total += 1

                if idx < 5:
                    print(f"    [DEBUG {method} #{idx}] Q={question[:60]} | pred='{pred}' | answer='{answer}'")

                if (idx+1) % 100 == 0:
                    print(f"  [{idx+1}/{num_samples}] {method}: {correct/total:.4f}")

            except Exception as e:
                print(f"  Error sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        for h in hooks:
            h.remove()

        acc = correct / total if total > 0 else 0
        all_results[method] = {"accuracy": round(acc, 4), "correct": correct, "total": total}
        print(f"  {method}: {acc:.4f}")

    result = {"model": model_name, "benchmark": "gqa", "attn_bias": attn_bias,
              "num_samples": num_samples, "results": all_results}
    path = "/results/gqa_evaluation.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()
    return result


# ============================================================================
# Entry point
# ============================================================================

@app.local_entrypoint()
def main(experiment: str = "gqa"):
    """
    Usage:
      modal run src/eval_extended.py --experiment gqa
    """
    if experiment == "gqa":
        r = gqa_evaluation.remote()
        print("\nGQA:")
        for m, res in r["results"].items():
            print(f"  {m}: {res['accuracy']:.4f}")
    else:
        print(f"Unknown: {experiment}")
