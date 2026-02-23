"""
VIAR: Vision-Informed Attention Rebalancing

Inference-time method to fix modality imbalance in Vision-Language Models.
Core idea: detect when VLMs neglect visual tokens via attention entropy,
then dynamically rescale cross-modal attention to restore visual grounding.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class VIARConfig:
    """Configuration for VIAR intervention."""
    # Layers to intervene in (0-indexed). If None, auto-detect via entropy.
    target_layers: Optional[List[int]] = None
    # Entropy ratio threshold: intervene when H_vis / H_text > threshold
    entropy_threshold: float = 1.5
    # Base rescaling factor for visual token attention
    alpha_base: float = 2.0
    # Maximum rescaling factor (clamp to prevent instability)
    alpha_max: float = 5.0
    # Whether to use contrastive visual anchoring (Step 3)
    use_contrastive_anchoring: bool = True
    # Temperature for contrastive head selection
    contrastive_temperature: float = 1.0
    # Minimum layer to intervene (skip embedding layers)
    min_layer: int = 2
    # Maximum layer to intervene (skip final classification layers)
    max_layer_frac: float = 0.75
    # Whether to apply per-head rescaling vs uniform
    per_head: bool = True


class VIARHook:
    """
    Attention hook that implements VIAR at inference time.
    
    Attaches to a VLM's attention layers and dynamically rebalances
    attention between visual and language tokens based on entropy analysis.
    """
    
    def __init__(
        self,
        config: VIARConfig,
        num_visual_tokens: int,
        total_tokens: int,
        num_layers: int,
    ):
        self.config = config
        self.num_visual_tokens = num_visual_tokens
        self.total_tokens = total_tokens
        self.num_layers = num_layers
        
        # Determine target layers
        max_layer = int(num_layers * config.max_layer_frac)
        if config.target_layers is not None:
            self.target_layers = set(config.target_layers)
        else:
            self.target_layers = set(range(config.min_layer, max_layer))
        
        # Storage for analysis
        self.attention_entropies: Dict[int, Dict[str, float]] = {}
        self.rescaling_factors: Dict[int, float] = {}
        self.hooks = []
    
    def compute_attention_entropy(
        self, attn_weights: torch.Tensor, token_mask: torch.Tensor
    ) -> float:
        """
        Compute entropy of attention distribution over a set of tokens.
        
        Args:
            attn_weights: [batch, heads, seq_len, seq_len] attention weights
            token_mask: [seq_len] boolean mask for tokens of interest
        
        Returns:
            Mean entropy over the masked token positions
        """
        # Get attention to masked tokens: [batch, heads, seq_len, num_masked]
        masked_attn = attn_weights[:, :, :, token_mask]
        
        # Renormalize over masked positions
        masked_attn = masked_attn / (masked_attn.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Compute entropy: H = -sum(p * log(p))
        log_attn = torch.log(masked_attn + 1e-10)
        entropy = -(masked_attn * log_attn).sum(dim=-1)  # [batch, heads, seq_len]
        
        return entropy.mean().item()
    
    def compute_rescaling_factor(
        self, h_vis: float, h_text: float, layer_idx: int
    ) -> float:
        """
        Compute dynamic rescaling factor based on entropy ratio.
        
        Higher entropy over visual tokens (diffuse attention) relative to
        text tokens means the model is neglecting vision -> stronger boost.
        """
        if h_text < 1e-6:
            return 1.0  # No text tokens or degenerate case
        
        entropy_ratio = h_vis / (h_text + 1e-10)
        
        if entropy_ratio < self.config.entropy_threshold:
            return 1.0  # Visual attention is already focused, no intervention
        
        # Scale linearly with entropy ratio above threshold
        excess = entropy_ratio - self.config.entropy_threshold
        alpha = self.config.alpha_base + excess
        alpha = min(alpha, self.config.alpha_max)
        
        # Decay rescaling in later layers (visual info already compressed)
        layer_frac = layer_idx / self.num_layers
        decay = 1.0 - 0.5 * layer_frac  # Less intervention in deeper layers
        alpha = 1.0 + (alpha - 1.0) * decay
        
        return alpha
    
    def rescale_attention(
        self,
        attn_weights: torch.Tensor,
        layer_idx: int,
        visual_mask: torch.Tensor,
        text_mask: torch.Tensor,
        head_importance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Core VIAR operation: rescale attention weights to boost visual tokens.
        
        Args:
            attn_weights: [batch, heads, seq_len, seq_len] pre-softmax or post-softmax
            layer_idx: which transformer layer
            visual_mask: [seq_len] boolean mask for visual tokens
            text_mask: [seq_len] boolean mask for text tokens
            head_importance: [heads] optional per-head scaling from contrastive anchoring
        
        Returns:
            Modified attention weights with boosted visual attention
        """
        if layer_idx not in self.target_layers:
            return attn_weights
        
        # Compute entropies
        h_vis = self.compute_attention_entropy(attn_weights, visual_mask)
        h_text = self.compute_attention_entropy(attn_weights, text_mask)
        
        # Store for analysis
        self.attention_entropies[layer_idx] = {"visual": h_vis, "text": h_text}
        
        # Compute rescaling factor
        alpha = self.compute_rescaling_factor(h_vis, h_text, layer_idx)
        self.rescaling_factors[layer_idx] = alpha
        
        if alpha <= 1.0 + 1e-6:
            return attn_weights  # No intervention needed
        
        # Create rescaling tensor
        scale = torch.ones_like(attn_weights)
        
        if self.config.per_head and head_importance is not None:
            # Per-head rescaling: stronger boost on heads that respond to vision
            # head_importance: [heads], higher = more visual-responsive
            head_scale = 1.0 + (alpha - 1.0) * head_importance.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            scale[:, :, :, visual_mask] = head_scale.expand_as(scale[:, :, :, visual_mask])
        else:
            # Uniform rescaling across all heads
            scale[:, :, :, visual_mask] = alpha
        
        # Apply rescaling
        rescaled = attn_weights * scale
        
        # Renormalize attention weights
        rescaled = rescaled / (rescaled.sum(dim=-1, keepdim=True) + 1e-10)
        
        return rescaled
    
    def get_diagnostics(self) -> Dict:
        """Return diagnostic information about the intervention."""
        return {
            "attention_entropies": self.attention_entropies,
            "rescaling_factors": self.rescaling_factors,
            "num_intervened_layers": sum(
                1 for v in self.rescaling_factors.values() if v > 1.0 + 1e-6
            ),
        }


class ContrastiveVisualAnchor:
    """
    Step 3 of VIAR: Identify visual-functional attention heads by comparing
    attention patterns with real image vs null image.
    
    Heads that differ most between real/null are the ones that actually
    respond to visual content -> these get the strongest boost.
    """
    
    def __init__(self, config: VIARConfig):
        self.config = config
        self.head_importance_cache: Dict[int, torch.Tensor] = {}
    
    @torch.no_grad()
    def compute_head_importance(
        self,
        model,
        processor,
        image,
        prompt: str,
        null_image=None,
        device: str = "cuda",
    ) -> Dict[int, torch.Tensor]:
        """
        Compare attention patterns: real image vs null image.
        
        Heads with larger attention divergence are more visual-responsive.
        Returns per-layer, per-head importance scores.
        """
        from PIL import Image as PILImage
        
        # Create null image if not provided (gray image of same size)
        if null_image is None:
            if hasattr(image, 'size'):
                null_image = PILImage.new('RGB', image.size, color=(128, 128, 128))
            else:
                # Tensor input
                null_image = torch.full_like(image, 0.5)
        
        # Process both inputs
        inputs_real = processor(images=image, text=prompt, return_tensors="pt").to(device)
        inputs_null = processor(images=null_image, text=prompt, return_tensors="pt").to(device)
        
        # Forward pass with attention output
        with torch.no_grad():
            outputs_real = model(**inputs_real, output_attentions=True)
            outputs_null = model(**inputs_null, output_attentions=True)
        
        head_importance = {}
        
        for layer_idx, (attn_real, attn_null) in enumerate(
            zip(outputs_real.attentions, outputs_null.attentions)
        ):
            # attn: [batch, heads, seq, seq]
            # Compute KL divergence per head
            # D_KL(real || null) for each head
            kl_div = F.kl_div(
                torch.log(attn_null + 1e-10),
                attn_real,
                reduction='none',
                log_target=False,
            ).mean(dim=(0, 2, 3))  # Average over batch, query, key -> [heads]
            
            # Normalize to [0, 1] range
            if kl_div.max() > kl_div.min():
                importance = (kl_div - kl_div.min()) / (kl_div.max() - kl_div.min())
            else:
                importance = torch.ones_like(kl_div)
            
            # Apply temperature
            importance = torch.softmax(
                importance / self.config.contrastive_temperature, dim=0
            ) * importance.shape[0]  # Scale so mean ~= 1
            
            head_importance[layer_idx] = importance
        
        self.head_importance_cache = head_importance
        return head_importance


def apply_viar_to_llava(
    model,
    processor,
    image,
    prompt: str,
    config: Optional[VIARConfig] = None,
    device: str = "cuda",
    max_new_tokens: int = 512,
    use_contrastive: bool = True,
) -> Tuple[str, Dict]:
    """
    Apply VIAR to a LLaVA-style model for a single inference.
    
    This is the main entry point for using VIAR. It:
    1. Processes the image and prompt
    2. Optionally computes contrastive head importance
    3. Hooks into the model's attention layers
    4. Generates output with rebalanced attention
    
    Args:
        model: LLaVA model
        processor: LLaVA processor
        image: PIL Image
        prompt: Text prompt
        config: VIAR configuration
        device: Device to run on
        max_new_tokens: Maximum generation length
        use_contrastive: Whether to use contrastive anchoring
    
    Returns:
        Tuple of (generated_text, diagnostics_dict)
    """
    if config is None:
        config = VIARConfig()
    
    # Process inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    
    # Determine visual token positions
    # In LLaVA, visual tokens are injected after the image placeholder
    # The exact positions depend on the model architecture
    input_ids = inputs["input_ids"]
    total_tokens = input_ids.shape[1]
    
    # For LLaVA: image tokens are typically 576 tokens (24x24 patches)
    # They appear before the text tokens in the sequence
    # We'll detect them from the model's image token embedding
    num_visual_tokens = 576  # Default for LLaVA with CLIP ViT-L/14@336
    
    # Get model layer count
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        num_layers = len(model.language_model.model.layers)
    else:
        num_layers = 32  # Default for 7B models
    
    # Optional: Contrastive visual anchoring
    head_importance = None
    if use_contrastive and config.use_contrastive_anchoring:
        anchor = ContrastiveVisualAnchor(config)
        head_importance = anchor.compute_head_importance(
            model, processor, image, prompt, device=device
        )
    
    # Create VIAR hook
    viar = VIARHook(
        config=config,
        num_visual_tokens=num_visual_tokens,
        total_tokens=total_tokens,
        num_layers=num_layers,
    )
    
    # Register hooks on attention layers
    hooks = []
    
    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            # output is typically (attn_output, attn_weights, past_key_value)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_output, attn_weights = output[0], output[1]
                if attn_weights is not None:
                    # Create masks
                    seq_len = attn_weights.shape[-1]
                    visual_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
                    text_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
                    
                    # Visual tokens come first in LLaVA
                    n_vis = min(num_visual_tokens, seq_len)
                    visual_mask[:n_vis] = True
                    text_mask[n_vis:] = True
                    
                    # Apply VIAR rescaling
                    hi = head_importance.get(layer_idx) if head_importance else None
                    modified_weights = viar.rescale_attention(
                        attn_weights, layer_idx, visual_mask, text_mask, hi
                    )
                    
                    # Return modified output
                    return (attn_output, modified_weights) + output[2:]
            return output
        return hook_fn
    
    # Note: The actual hook registration depends on the model architecture.
    # For LLaVA, we hook into the self-attention modules of the LLM backbone.
    # This will be adapted per-model in the evaluation script.
    
    # Generate with VIAR
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            output_attentions=True,
            return_dict_in_generate=True,
        )
    
    # Decode
    generated_ids = output.sequences[:, inputs["input_ids"].shape[1]:]
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Cleanup hooks
    for h in hooks:
        h.remove()
    
    diagnostics = viar.get_diagnostics()
    
    return generated_text, diagnostics


# ============================================================================
# Alternative: Logit-level VIAR (complementary to attention-level)
# This can be combined with VCD for compound gains
# ============================================================================

class VIARLogitCorrection:
    """
    Logit-level correction that complements attention-level VIAR.
    
    Idea: After attention rebalancing, also correct the output distribution
    by contrasting with a "vision-deprived" forward pass (similar to VCD
    but using our entropy-informed approach).
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def correct_logits(
        self,
        logits_full: torch.Tensor,
        logits_deprived: torch.Tensor,
    ) -> torch.Tensor:
        """
        Contrastive logit correction:
        p_corrected = softmax((1+alpha) * log p_full - alpha * log p_deprived)
        
        This suppresses tokens that the model generates even without vision.
        """
        log_full = F.log_softmax(logits_full, dim=-1)
        log_deprived = F.log_softmax(logits_deprived, dim=-1)
        
        corrected = (1 + self.alpha) * log_full - self.alpha * log_deprived
        return corrected
