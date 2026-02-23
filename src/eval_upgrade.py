"""
VIAR Upgrade Experiments — pushing from weak accept to strong accept.

Experiments:
1. Language Prior Collapse: Compare hidden states real-image vs black-image at text positions
   → If cosine sim peaks in neglect zone, proves model ignores visual info there
2. Per-Sample Neglect-Hallucination Correlation: Does deeper neglect predict hallucination?
   → If yes, this is the "surprising result" that pushes to 9+
3. Gradient Attribution: Gradient-based visual importance across layers
   → Independent validation of attention-based findings
4. Cross-Architecture: LLaVA-NeXT-Mistral-7B (different LM backbone, same architecture)
5. Cross-Architecture: Qwen2-VL-7B-Instruct (different architecture entirely)
"""

import modal
import json
import os

app = modal.App("viar-upgrade")

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

qwen_image = (
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
        "qwen-vl-utils",
    )
    .env({"HF_HOME": "/models", "TRANSFORMERS_CACHE": "/models"})
)

model_volume = modal.Volume.from_name("viar-model-cache", create_if_missing=True)


# ============================================================================
# Function 1: LLaVA-1.5-7B Mechanistic Experiments
# Language prior collapse + per-sample correlation + gradient attribution
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume},
    timeout=10800,  # 3 hours
)
def llava_mechanistic():
    """
    Three mechanistic experiments on LLaVA-1.5-7B:
    1. Language prior collapse (real vs black image hidden states)
    2. Per-sample neglect-hallucination correlation
    3. Gradient-based visual importance
    """
    import torch
    import numpy as np
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from datasets import load_dataset
    from PIL import Image

    print("Loading LLaVA-1.5-7B...")
    model_name = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_name, cache_dir="/models")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        cache_dir="/models", attn_implementation="eager",
    )
    model.eval()
    device = model.device

    print("Loading POPE dataset...")
    dataset = load_dataset("lmms-lab/POPE", split="test")
    samples = list(dataset.select(range(500)))

    n_layers = 32
    n_vis = 576
    results = {}

    # ==================================================================
    # Experiment 1: Language Prior Collapse
    # Compare hidden states at TEXT positions: real image vs black image
    # ==================================================================
    print("\n=== Experiment 1: Language Prior Collapse ===")
    n_collapse = 200

    # Store per-layer metrics
    raw_cosine_sims = {l: [] for l in range(n_layers)}
    residual_cosine_sims = {l: [] for l in range(n_layers)}
    vis_token_norms_real = {l: [] for l in range(n_layers)}
    vis_token_norms_black = {l: [] for l in range(n_layers)}

    for idx in range(n_collapse):
        sample = samples[idx]
        image = sample['image'].convert('RGB')
        question = sample['question']
        prompt = f"USER: <image>\n{question}\nASSISTANT:"

        black_image = Image.new('RGB', image.size, (0, 0, 0))

        inputs_real = processor(text=prompt, images=image, return_tensors="pt")
        inputs_real = {k: v.to(device) for k, v in inputs_real.items()}

        inputs_black = processor(text=prompt, images=black_image, return_tensors="pt")
        inputs_black = {k: v.to(device) for k, v in inputs_black.items()}

        with torch.no_grad():
            out_real = model(**inputs_real, output_hidden_states=True)
            out_black = model(**inputs_black, output_hidden_states=True)

        # hidden_states: tuple of (n_layers+1) tensors, each (batch, seq_len, hidden_dim)
        # Index 0 = embedding layer output, index l+1 = layer l output
        for l in range(n_layers):
            h_real = out_real.hidden_states[l + 1][0]  # (seq_len, hidden_dim)
            h_black = out_black.hidden_states[l + 1][0]

            # Text positions (after visual tokens)
            h_real_text = h_real[n_vis:, :]
            h_black_text = h_black[n_vis:, :]

            # 1a. Raw cosine similarity at text positions
            cos = torch.nn.functional.cosine_similarity(
                h_real_text, h_black_text, dim=-1
            ).mean().item()
            raw_cosine_sims[l].append(cos)

            # 1b. Residual update cosine similarity
            if l > 0:
                h_real_prev = out_real.hidden_states[l][0][n_vis:, :]
                h_black_prev = out_black.hidden_states[l][0][n_vis:, :]
                update_real = h_real_text - h_real_prev
                update_black = h_black_text - h_black_prev
                # Avoid division by zero for very small updates
                norms = update_real.norm(dim=-1) * update_black.norm(dim=-1)
                valid = norms > 1e-8
                if valid.any():
                    res_cos = torch.nn.functional.cosine_similarity(
                        update_real[valid], update_black[valid], dim=-1
                    ).mean().item()
                else:
                    res_cos = 1.0
                residual_cosine_sims[l].append(res_cos)
            else:
                residual_cosine_sims[l].append(float('nan'))

            # 1c. Visual token norms
            vis_norm_real = h_real[:n_vis, :].norm(dim=-1).mean().item()
            vis_norm_black = h_black[:n_vis, :].norm(dim=-1).mean().item()
            vis_token_norms_real[l].append(vis_norm_real)
            vis_token_norms_black[l].append(vis_norm_black)

        del out_real, out_black, inputs_real, inputs_black
        torch.cuda.empty_cache()

        if idx % 50 == 0:
            print(f"  Collapse: {idx}/{n_collapse}")

    # Aggregate collapse results
    collapse_by_layer = {}
    for l in range(n_layers):
        collapse_by_layer[l] = {
            'raw_cosine_sim_mean': float(np.mean(raw_cosine_sims[l])),
            'raw_cosine_sim_std': float(np.std(raw_cosine_sims[l])),
            'residual_cosine_sim_mean': float(np.nanmean(residual_cosine_sims[l])),
            'residual_cosine_sim_std': float(np.nanstd(residual_cosine_sims[l])),
            'vis_norm_real_mean': float(np.mean(vis_token_norms_real[l])),
            'vis_norm_black_mean': float(np.mean(vis_token_norms_black[l])),
            'vis_norm_difference': float(np.mean(vis_token_norms_real[l]) - np.mean(vis_token_norms_black[l])),
        }

    results['language_prior_collapse'] = {
        'description': 'Cosine similarity of text-token hidden states: real image vs black image',
        'interpretation': 'Higher similarity in neglect zone means model ignores visual input there',
        'n_samples': n_collapse,
        'n_layers': n_layers,
        'n_vis_tokens': n_vis,
        'by_layer': collapse_by_layer,
    }
    print("  Collapse experiment complete.")

    # ==================================================================
    # Experiment 2: Per-Sample Neglect-Hallucination Correlation
    # ==================================================================
    print("\n=== Experiment 2: Per-Sample Correlation ===")
    n_corr = 500

    per_sample_data = []

    for idx in range(n_corr):
        sample = samples[idx]
        image = sample['image'].convert('RGB')
        question = sample['question']
        answer_gt = sample['answer'].strip().lower()
        prompt = f"USER: <image>\n{question}\nASSISTANT:"

        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs, output_attentions=True)

        # Get model prediction
        logits = out.logits[0, -1, :]
        yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_id = processor.tokenizer.encode("no", add_special_tokens=False)[0]
        p_yes = torch.softmax(logits[torch.tensor([yes_id, no_id])], dim=0)[0].item()
        pred = "yes" if p_yes > 0.5 else "no"
        correct = (pred == answer_gt)
        hallucinated = (pred == "yes" and answer_gt == "no")  # false positive

        # Compute per-sample text→vis attention at each layer
        layer_text2vis = []
        for l in range(n_layers):
            attn = out.attentions[l][0]  # (n_heads, seq_len, seq_len)
            # Text queries attending to visual keys
            text2vis = attn[:, n_vis:, :n_vis].mean().item()
            layer_text2vis.append(text2vis)

        # Neglect zone depth (mean text→vis in L8-16)
        neglect_depth = float(np.mean(layer_text2vis[8:17]))
        # Early layers (L2-7) for comparison
        early_depth = float(np.mean(layer_text2vis[2:8]))
        # Late layers (L17-30)
        late_depth = float(np.mean(layer_text2vis[17:31]))

        per_sample_data.append({
            'idx': idx,
            'correct': correct,
            'hallucinated': hallucinated,
            'pred': pred,
            'answer_gt': answer_gt,
            'p_yes': float(p_yes),
            'neglect_depth': neglect_depth,
            'early_depth': early_depth,
            'late_depth': late_depth,
            'layer_text2vis': [float(x) for x in layer_text2vis],
        })

        del out, inputs
        torch.cuda.empty_cache()

        if idx % 100 == 0:
            print(f"  Correlation: {idx}/{n_corr}")

    # Compute correlation statistics
    correct_samples = [s for s in per_sample_data if s['correct']]
    incorrect_samples = [s for s in per_sample_data if not s['correct']]
    hallucinated_samples = [s for s in per_sample_data if s['hallucinated']]

    neglect_correct = [s['neglect_depth'] for s in correct_samples]
    neglect_incorrect = [s['neglect_depth'] for s in incorrect_samples]
    neglect_hallucinated = [s['neglect_depth'] for s in hallucinated_samples]

    from scipy import stats

    # Point-biserial correlation: neglect depth vs correctness
    all_neglect = [s['neglect_depth'] for s in per_sample_data]
    all_correct = [1 if s['correct'] else 0 for s in per_sample_data]
    pb_corr, pb_pval = stats.pointbiserialr(all_correct, all_neglect)

    # t-test: neglect depth for correct vs incorrect
    if len(neglect_incorrect) > 1:
        t_stat, t_pval = stats.ttest_ind(neglect_correct, neglect_incorrect)
        cohens_d = (np.mean(neglect_correct) - np.mean(neglect_incorrect)) / np.sqrt(
            (np.std(neglect_correct)**2 + np.std(neglect_incorrect)**2) / 2
        )
    else:
        t_stat, t_pval, cohens_d = float('nan'), float('nan'), float('nan')

    # Per-layer correlation: text2vis vs correctness
    layer_correlations = []
    for l in range(n_layers):
        layer_vals = [s['layer_text2vis'][l] for s in per_sample_data]
        r, p = stats.pointbiserialr(all_correct, layer_vals)
        layer_correlations.append({'layer': l, 'correlation': float(r), 'p_value': float(p)})

    results['persample_correlation'] = {
        'description': 'Per-sample neglect depth vs hallucination',
        'n_samples': n_corr,
        'n_correct': len(correct_samples),
        'n_incorrect': len(incorrect_samples),
        'n_hallucinated': len(hallucinated_samples),
        'neglect_depth_correct_mean': float(np.mean(neglect_correct)) if neglect_correct else None,
        'neglect_depth_correct_std': float(np.std(neglect_correct)) if neglect_correct else None,
        'neglect_depth_incorrect_mean': float(np.mean(neglect_incorrect)) if neglect_incorrect else None,
        'neglect_depth_incorrect_std': float(np.std(neglect_incorrect)) if neglect_incorrect else None,
        'neglect_depth_hallucinated_mean': float(np.mean(neglect_hallucinated)) if neglect_hallucinated else None,
        'point_biserial_r': float(pb_corr),
        'point_biserial_p': float(pb_pval),
        'ttest_t': float(t_stat),
        'ttest_p': float(t_pval),
        'cohens_d': float(cohens_d),
        'layer_correlations': layer_correlations,
        'per_sample_data': per_sample_data,  # full data for figure generation
    }
    print(f"  Correlation: pb_r={pb_corr:.4f}, p={pb_pval:.4f}, d={cohens_d:.4f}")

    # ==================================================================
    # Experiment 3: Gradient-Based Visual Importance
    # ==================================================================
    print("\n=== Experiment 3: Gradient Attribution ===")
    n_grad = 100

    # Enable gradients
    for param in model.parameters():
        param.requires_grad_(True)

    grad_vis_importance = {l: [] for l in range(n_layers)}

    for idx in range(n_grad):
        sample = samples[idx]
        image = sample['image'].convert('RGB')
        question = sample['question']
        prompt = f"USER: <image>\n{question}\nASSISTANT:"

        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Register backward hooks to capture gradients at each layer
        layer_grads = {}

        def make_hook(layer_idx):
            def hook_fn(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    layer_grads[layer_idx] = grad_output[0].detach()
            return hook_fn

        hooks = []
        for i, layer in enumerate(model.language_model.model.layers):
            h = layer.register_full_backward_hook(make_hook(i))
            hooks.append(h)

        # Forward pass
        model.zero_grad()
        outputs = model(**inputs)

        # Get yes-token logit
        logits = outputs.logits[0, -1, :]
        yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
        yes_logit = logits[yes_id]

        # Backward
        yes_logit.backward()

        # Extract gradient norms at visual positions
        for l in range(n_layers):
            if l in layer_grads:
                grad = layer_grads[l][0]  # (seq_len, hidden_dim)
                vis_grad_norm = grad[:n_vis, :].norm(dim=-1).mean().item()
                grad_vis_importance[l].append(vis_grad_norm)
            else:
                grad_vis_importance[l].append(0.0)

        # Remove hooks
        for h in hooks:
            h.remove()

        del outputs, inputs, layer_grads
        torch.cuda.empty_cache()

        if idx % 25 == 0:
            print(f"  Gradient: {idx}/{n_grad}")

    # Disable gradients again
    for param in model.parameters():
        param.requires_grad_(False)

    gradient_by_layer = {}
    for l in range(n_layers):
        gradient_by_layer[l] = {
            'mean_vis_grad_norm': float(np.mean(grad_vis_importance[l])),
            'std_vis_grad_norm': float(np.std(grad_vis_importance[l])),
        }

    results['gradient_attribution'] = {
        'description': 'Gradient-based visual importance: mean gradient norm at visual positions per layer',
        'interpretation': 'Higher gradient norm = model output more sensitive to visual tokens at that layer',
        'n_samples': n_grad,
        'by_layer': gradient_by_layer,
    }
    print("  Gradient attribution complete.")

    print("\n=== All LLaVA mechanistic experiments complete ===")
    return results


# ============================================================================
# Function 2: LLaVA-NeXT-Mistral-7B Cross-Architecture
# ============================================================================

@app.function(
    gpu="A100",
    image=vlm_image,
    volumes={"/models": model_volume},
    timeout=7200,  # 2 hours
)
def llava_mistral_profile():
    """Attention profile for LLaVA-NeXT-Mistral-7B (Mistral backbone)."""
    import torch
    import numpy as np
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    from datasets import load_dataset

    print("Loading LLaVA-NeXT-Mistral-7B...")
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    processor = LlavaNextProcessor.from_pretrained(model_name, cache_dir="/models")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        cache_dir="/models", attn_implementation="eager",
    )
    model.eval()
    device = model.device

    print("Loading POPE dataset...")
    dataset = load_dataset("lmms-lab/POPE", split="test")
    samples = list(dataset.select(range(200)))

    n_layers = 32  # Mistral-7B has 32 layers

    layer_text2vis = {l: [] for l in range(n_layers)}
    layer_head_text2vis = {l: [] for l in range(n_layers)}  # per-head data

    successful = 0
    failed = 0

    for idx in range(len(samples)):
        try:
            sample = samples[idx]
            image = sample['image'].convert('RGB')
            question = sample['question']
            prompt = f"[INST] <image>\n{question} [/INST]"

            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs, output_attentions=True)

            # Determine visual token count from attention shape
            # input_ids had one <image> token, model expanded it to n_vis visual features
            seq_len = out.attentions[0].shape[-1]
            orig_len = inputs['input_ids'].shape[1]
            n_vis = seq_len - orig_len + 1  # +1 for the removed <image> token

            if n_vis < 1 or n_vis >= seq_len:
                failed += 1
                continue

            for l in range(n_layers):
                attn = out.attentions[l][0]  # (n_heads, seq_len, seq_len)
                # Text→vis: text query positions attending to visual key positions
                text2vis_per_head = attn[:, n_vis:, :n_vis].mean(dim=(1, 2))  # (n_heads,)
                mean_t2v = text2vis_per_head.mean().item()
                layer_text2vis[l].append(mean_t2v)
                layer_head_text2vis[l].append(text2vis_per_head.cpu().numpy().tolist())

            successful += 1

            del out, inputs
            torch.cuda.empty_cache()

            if idx % 50 == 0:
                print(f"  Mistral: {idx}/{len(samples)}, n_vis={n_vis}")

        except Exception as e:
            failed += 1
            print(f"  Sample {idx} failed: {e}")
            torch.cuda.empty_cache()

    # Aggregate
    by_layer = {}
    for l in range(n_layers):
        vals = layer_text2vis[l]
        head_vals = layer_head_text2vis[l]
        if vals:
            # Average per-head across samples
            mean_per_head = np.mean(head_vals, axis=0).tolist() if head_vals else []
            by_layer[l] = {
                'text2vis_mean': float(np.mean(vals)),
                'text2vis_std': float(np.std(vals)),
                'per_head_text2vis': mean_per_head,
                'head_std': float(np.std(mean_per_head)) if mean_per_head else 0.0,
            }

    return {
        'model': model_name,
        'architecture': 'LLaVA-NeXT (Mistral-7B backbone, direct projection)',
        'n_layers': n_layers,
        'successful_samples': successful,
        'failed_samples': failed,
        'by_layer': by_layer,
    }


# ============================================================================
# Function 3: Qwen2-VL-7B Cross-Architecture
# ============================================================================

@app.function(
    gpu="A100",
    image=qwen_image,
    volumes={"/models": model_volume},
    timeout=7200,  # 2 hours
)
def qwen2vl_profile():
    """Attention profile for Qwen2-VL-7B-Instruct (Q-Former-free, different architecture)."""
    import torch
    import numpy as np
    from datasets import load_dataset

    results = {}

    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        print("Loading Qwen2-VL-7B-Instruct...")
        model_name = "Qwen/Qwen2-VL-7B-Instruct"
        processor = AutoProcessor.from_pretrained(model_name, cache_dir="/models")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
            cache_dir="/models", attn_implementation="eager",
        )
        model.eval()
        device = next(model.parameters()).device

        print("Loading POPE dataset...")
        dataset = load_dataset("lmms-lab/POPE", split="test")
        samples = list(dataset.select(range(200)))

        n_layers = 28  # Qwen2-7B has 28 layers
        n_heads = 28

        layer_text2vis = {l: [] for l in range(n_layers)}
        layer_head_text2vis = {l: [] for l in range(n_layers)}

        # Find image pad token ID
        image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        successful = 0
        failed = 0

        for idx in range(len(samples)):
            try:
                sample = samples[idx]
                image = sample['image'].convert('RGB')
                question = sample['question']

                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ]}
                ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Identify visual token positions
                input_ids = inputs['input_ids'][0]
                vis_mask = (input_ids == image_pad_id)
                n_vis = vis_mask.sum().item()
                text_mask = ~vis_mask

                if n_vis < 1:
                    failed += 1
                    continue

                with torch.no_grad():
                    out = model(**inputs, output_attentions=True)

                for l in range(min(n_layers, len(out.attentions))):
                    attn = out.attentions[l][0]  # (n_heads, seq_len, seq_len)
                    # Text queries → visual keys
                    text_indices = text_mask.nonzero().squeeze(-1)
                    vis_indices = vis_mask.nonzero().squeeze(-1)

                    if len(text_indices.shape) == 0 or len(vis_indices.shape) == 0:
                        continue

                    # Extract text→vis attention block
                    t2v = attn[:, text_indices][:, :, vis_indices]  # (heads, n_text, n_vis)
                    t2v_per_head = t2v.mean(dim=(1, 2))  # (heads,)
                    mean_t2v = t2v_per_head.mean().item()
                    layer_text2vis[l].append(mean_t2v)
                    layer_head_text2vis[l].append(t2v_per_head.cpu().numpy().tolist())

                successful += 1
                del out, inputs
                torch.cuda.empty_cache()

                if idx % 50 == 0:
                    print(f"  Qwen2-VL: {idx}/{len(samples)}, n_vis={n_vis}")

            except Exception as e:
                failed += 1
                if idx < 5:
                    print(f"  Sample {idx} failed: {e}")
                torch.cuda.empty_cache()

        by_layer = {}
        for l in range(n_layers):
            vals = layer_text2vis[l]
            head_vals = layer_head_text2vis[l]
            if vals:
                mean_per_head = np.mean(head_vals, axis=0).tolist() if head_vals else []
                by_layer[l] = {
                    'text2vis_mean': float(np.mean(vals)),
                    'text2vis_std': float(np.std(vals)),
                    'per_head_text2vis': mean_per_head,
                    'head_std': float(np.std(mean_per_head)) if mean_per_head else 0.0,
                }

        results = {
            'model': model_name,
            'architecture': 'Qwen2-VL (ViT + direct projection + Qwen2-7B, 28 layers)',
            'n_layers': n_layers,
            'n_heads': n_heads,
            'successful_samples': successful,
            'failed_samples': failed,
            'by_layer': by_layer,
        }

    except Exception as e:
        results = {
            'error': str(e),
            'model': 'Qwen/Qwen2-VL-7B-Instruct',
            'status': 'FAILED',
        }
        print(f"Qwen2-VL experiment failed: {e}")

    return results
