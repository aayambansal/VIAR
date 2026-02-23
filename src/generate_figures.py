"""
Generate all publication-quality figures for the VIAR paper.
Target: ECCV 2026 (Springer LNCS format)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json
import os

# ============================================================================
# Publication Style Setup
# ============================================================================

# ECCV uses Springer LNCS: single column 122mm, text width ~122mm
# Double column not applicable (single column format)
# Figures can be up to full text width

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
})

# Okabe-Ito colorblind-safe palette
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'red': '#D55E00',
    'purple': '#CC79A7',
    'cyan': '#56B4E9',
    'yellow': '#F0E442',
    'black': '#000000',
    'gray': '#999999',
}

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
RESULTDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(OUTDIR, exist_ok=True)


def load_json(name):
    with open(os.path.join(RESULTDIR, name)) as f:
        return json.load(f)


# ============================================================================
# Figure 1: Visual Attention Analysis (U-shaped pattern)
# This is the key diagnostic figure showing the visual neglect zone
# ============================================================================

def figure1_attention_analysis():
    """
    Figure 1: Layer-wise visual attention analysis in LLaVA-1.5-7B.
    Shows U-shaped pattern with visual neglect zone (layers 8-16).
    """
    data = load_json('attention_analysis_summary.json')
    layers_data = data['aggregated_by_layer']

    layer_ids = sorted([int(k) for k in layers_data.keys()])
    vis_frac = [layers_data[str(l)]['vis_attn_frac_mean'] for l in layer_ids]
    h_ratio = [layers_data[str(l)]['h_ratio_mean'] for l in layer_ids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.4))

    # Panel A: Visual attention fraction per layer
    ax1.bar(layer_ids, [v * 100 for v in vis_frac], color=COLORS['blue'], alpha=0.7, width=0.8)
    # Highlight neglect zone
    for l in range(8, 17):
        ax1.bar(l, vis_frac[l] * 100, color=COLORS['red'], alpha=0.7, width=0.8)
    # Highlight layer 31
    ax1.bar(31, vis_frac[31] * 100, color=COLORS['orange'], alpha=0.7, width=0.8)

    ax1.axhspan(17, 21, alpha=0.1, color=COLORS['red'], label='Neglect zone')
    ax1.set_xlabel('Transformer Layer')
    ax1.set_ylabel('Visual Attention Fraction (%)')
    ax1.set_title('(a) Fraction of attention on visual tokens')
    ax1.set_xlim(-0.5, 31.5)
    ax1.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])

    # Add annotations
    ax1.annotate('Neglect\nzone', xy=(12, 18), fontsize=6.5,
                 color=COLORS['red'], ha='center', fontweight='bold')
    ax1.annotate('Peak\nengagement', xy=(3.5, 38), fontsize=6,
                 color=COLORS['green'], ha='center')

    # Panel B: Entropy ratio (visual / text)
    ax2.plot(layer_ids, h_ratio, color=COLORS['blue'], marker='o', markersize=3, linewidth=1.2)
    ax2.axhline(y=1.0, color=COLORS['gray'], linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.fill_between(range(8, 17), 0, [h_ratio[l] for l in range(8, 17)],
                     alpha=0.15, color=COLORS['red'])

    ax2.set_xlabel('Transformer Layer')
    ax2.set_ylabel('Entropy Ratio (Visual / Text)')
    ax2.set_title('(b) Visual attention entropy ratio')
    ax2.set_xlim(-0.5, 31.5)
    ax2.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])

    ax2.annotate('H_ratio > 1:\nDiffuse visual\nattention', xy=(12, 1.15),
                 fontsize=6, color=COLORS['red'], ha='center')
    ax2.annotate('H_ratio < 1:\nFocused visual\nattention', xy=(3.5, 0.3),
                 fontsize=6, color=COLORS['green'], ha='center')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig1_attention_analysis.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'fig1_attention_analysis.png'), dpi=300)
    plt.close(fig)
    print("Figure 1: Attention analysis saved.")


# ============================================================================
# Figure 2: VIAR Method Overview + Bias Sweep
# ============================================================================

def figure2_bias_sweep():
    """
    Figure 2: Bias sweep results showing inverted-U curve.
    """
    pope_data = load_json('bias_sweep_pope.json')
    mmstar_data = load_json('bias_sweep_mmstar.json')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.4))

    # Panel A: POPE bias sweep
    biases = [float(b) for b in pope_data['sweep'].keys()]
    pope_acc = [pope_data['sweep'][str(b)]['accuracy'] * 100 for b in biases]

    ax1.plot(biases, pope_acc, color=COLORS['blue'], marker='o', linewidth=1.5, markersize=5)
    ax1.axhline(y=pope_data['sweep']['0.0']['accuracy'] * 100,
                color=COLORS['gray'], linestyle='--', linewidth=0.8, label='Baseline')

    # Highlight optimal
    opt_idx = np.argmax(pope_acc)
    ax1.scatter([biases[opt_idx]], [pope_acc[opt_idx]], color=COLORS['red'],
                s=80, zorder=5, marker='*', label=f'Best: {pope_acc[opt_idx]:.1f}%')

    ax1.set_xlabel('Attention Bias Strength')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('(a) POPE (200 samples)')
    ax1.legend(frameon=False, loc='lower left', fontsize=6.5)
    ax1.set_ylim(45, 90)

    # Panel B: MMStar bias sweep
    biases_m = [float(b) for b in mmstar_data['sweep'].keys()]
    mmstar_acc = [mmstar_data['sweep'][str(b)]['accuracy'] * 100 for b in biases_m]

    ax2.plot(biases_m, mmstar_acc, color=COLORS['orange'], marker='s', linewidth=1.5, markersize=5)
    ax2.axhline(y=mmstar_data['sweep']['0.0']['accuracy'] * 100,
                color=COLORS['gray'], linestyle='--', linewidth=0.8, label='Baseline')

    opt_idx_m = np.argmax(mmstar_acc)
    ax2.scatter([biases_m[opt_idx_m]], [mmstar_acc[opt_idx_m]], color=COLORS['red'],
                s=80, zorder=5, marker='*', label=f'Best: {mmstar_acc[opt_idx_m]:.1f}%')

    ax2.set_xlabel('Attention Bias Strength')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('(b) MMStar (200 samples)')
    ax2.legend(frameon=False, loc='lower left', fontsize=6.5)
    ax2.set_ylim(25, 55)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig2_bias_sweep.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'fig2_bias_sweep.png'), dpi=300)
    plt.close(fig)
    print("Figure 2: Bias sweep saved.")


# ============================================================================
# Figure 3: Main Results — MMStar Category Breakdown
# ============================================================================

def figure3_mmstar_results():
    """
    Figure 3: Full MMStar results with per-category breakdown.
    """
    data = load_json('full_mmstar.json')

    categories = ['coarse perception', 'fine-grained perception', 'instance reasoning',
                  'logical reasoning', 'math', 'science & technology']
    short_names = ['Coarse\nPercep.', 'Fine-gr.\nPercep.', 'Instance\nReason.',
                   'Logical\nReason.', 'Math', 'Sci &\nTech']

    baseline_acc = [data['configs']['baseline']['by_category'][c]['accuracy'] * 100 for c in categories]
    neglect_acc = [data['configs']['neglect_8_16']['by_category'][c]['accuracy'] * 100 for c in categories]
    adaptive_acc = [data['configs']['adaptive']['by_category'][c]['accuracy'] * 100 for c in categories]

    fig, ax = plt.subplots(figsize=(6.5, 3.0))

    x = np.arange(len(categories))
    width = 0.25

    bars1 = ax.bar(x - width, baseline_acc, width, label='Baseline',
                   color=COLORS['gray'], alpha=0.7)
    bars2 = ax.bar(x, neglect_acc, width, label='VIAR (layers 8-16)',
                   color=COLORS['blue'], alpha=0.8)
    bars3 = ax.bar(x + width, adaptive_acc, width, label='VIAR-Adaptive (all layers)',
                   color=COLORS['orange'], alpha=0.8)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('MMStar Per-Category Performance (n=250 per category)')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=7)
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    ax.set_ylim(0, 65)

    # Add random chance line
    ax.axhline(y=25, color=COLORS['gray'], linestyle=':', linewidth=0.6, alpha=0.5)
    ax.text(5.3, 25.5, 'Random\nchance', fontsize=5.5, color=COLORS['gray'], va='bottom')

    # Add delta annotations for adaptive
    for i, (b, a) in enumerate(zip(baseline_acc, adaptive_acc)):
        delta = a - b
        color = COLORS['green'] if delta > 0 else COLORS['red']
        sign = '+' if delta > 0 else ''
        ax.text(x[i] + width, a + 0.8, f'{sign}{delta:.1f}', ha='center',
                va='bottom', fontsize=5.5, color=color, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig3_mmstar_categories.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'fig3_mmstar_categories.png'), dpi=300)
    plt.close(fig)
    print("Figure 3: MMStar categories saved.")


# ============================================================================
# Figure 4: Layer Ablation Results
# ============================================================================

def figure4_layer_ablation():
    """
    Figure 4: Layer ablation showing which layers matter for VIAR.
    """
    pope_data = load_json('layer_ablation_pope.json')
    mmstar_data = load_json('layer_ablation_mmstar.json')

    configs = ['baseline', 'early_0_7', 'neglect_8_16', 'deep_neglect',
               'late_17_31', 'final_31', 'all_layers', 'adaptive']
    config_labels = ['Baseline', 'Early\n(0-7)', 'Neglect\n(8-16)', 'Deep\n(9-15)',
                     'Late\n(17-31)', 'Final\n(31)', 'All\n(0-31)', 'Adaptive']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # POPE
    pope_acc = [pope_data['configs'][c]['accuracy'] * 100 for c in configs]
    colors_pope = [COLORS['gray'] if c == 'baseline' else
                   COLORS['red'] if c in ['neglect_8_16'] else
                   COLORS['blue'] for c in configs]

    bars1 = ax1.bar(range(len(configs)), pope_acc, color=colors_pope, alpha=0.8)
    ax1.axhline(y=pope_data['configs']['baseline']['accuracy'] * 100,
                color=COLORS['gray'], linestyle='--', linewidth=0.8)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(f'(a) POPE (bias=2.0, n=200)')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(config_labels, fontsize=5.5)
    ax1.set_ylim(78, 86)

    # MMStar
    mmstar_acc = [mmstar_data['configs'][c]['accuracy'] * 100 for c in configs]
    colors_mm = [COLORS['gray'] if c == 'baseline' else
                 COLORS['orange'] if c == 'adaptive' else
                 COLORS['blue'] for c in configs]

    bars2 = ax2.bar(range(len(configs)), mmstar_acc, color=colors_mm, alpha=0.8)
    ax2.axhline(y=mmstar_data['configs']['baseline']['accuracy'] * 100,
                color=COLORS['gray'], linestyle='--', linewidth=0.8)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'(b) MMStar (bias=1.0, n=200)')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(config_labels, fontsize=5.5)
    ax2.set_ylim(46, 52)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig4_layer_ablation.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'fig4_layer_ablation.png'), dpi=300)
    plt.close(fig)
    print("Figure 4: Layer ablation saved.")


# ============================================================================
# Figure 5: Summary comparison table (as figure)
# ============================================================================

def figure5_summary():
    """
    Figure 5: Summary comparison — overall results on both benchmarks.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.2))

    # POPE comparison
    methods = ['Baseline', 'VIAR\n(b=2.0)']
    pope_metrics = {
        'Accuracy': [83.6, 84.4],
        'F1': [82.3, 84.4],
        'Precision': [89.3, 84.4],
        'Recall': [76.4, 84.4],
    }

    x = np.arange(len(pope_metrics))
    width = 0.35

    baseline_vals = [v[0] for v in pope_metrics.values()]
    viar_vals = [v[1] for v in pope_metrics.values()]

    ax1.bar(x - width/2, baseline_vals, width, label='Baseline',
            color=COLORS['gray'], alpha=0.7)
    ax1.bar(x + width/2, viar_vals, width, label='VIAR',
            color=COLORS['blue'], alpha=0.8)

    ax1.set_ylabel('Score (%)')
    ax1.set_title('(a) POPE (n=500)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pope_metrics.keys(), fontsize=7)
    ax1.legend(frameon=False, fontsize=6.5)
    ax1.set_ylim(70, 95)

    # MMStar comparison
    methods_mm = ['Baseline', 'VIAR\n(8-16)', 'VIAR-\nAdaptive']
    mmstar_overall = [32.3, 33.4, 34.1]

    colors_mm = [COLORS['gray'], COLORS['blue'], COLORS['orange']]
    ax2.bar(range(3), mmstar_overall, color=colors_mm, alpha=0.8)
    ax2.axhline(y=25, color=COLORS['gray'], linestyle=':', linewidth=0.6)

    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('(b) MMStar Overall (n=1500)')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['Baseline', 'VIAR\n(layers 8-16)', 'VIAR-\nAdaptive'], fontsize=6.5)
    ax2.set_ylim(20, 38)

    # Add delta annotations
    for i, (val, base) in enumerate(zip(mmstar_overall, [32.3]*3)):
        if i > 0:
            delta = val - base
            ax2.text(i, val + 0.3, f'+{delta:.1f}%', ha='center',
                     fontsize=6.5, color=COLORS['green'], fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig5_summary.pdf'))
    fig.savefig(os.path.join(OUTDIR, 'fig5_summary.png'), dpi=300)
    plt.close(fig)
    print("Figure 5: Summary saved.")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Generating all figures...")
    figure1_attention_analysis()
    figure2_bias_sweep()
    figure3_mmstar_results()
    figure4_layer_ablation()
    figure5_summary()
    print(f"\nAll figures saved to {OUTDIR}/")
    print("Figures: fig1_attention_analysis, fig2_bias_sweep, fig3_mmstar_categories, fig4_layer_ablation, fig5_summary")
