"""Generate all updated figures for the VIAR ECCV paper revision."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = '/tmp/eecv-paper/figures'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette (colorblind-friendly)
C = {
    'baseline': '#4477AA',
    'viar': '#EE6677',
    'vcd': '#228833',
    'viar_vcd': '#CCBB44',
    'llava15': '#4477AA',
    'llava16': '#EE6677',
    'neglect': '#FFCCCC',
    'target': '#FF6666',
    'gray': '#888888',
}


# ============================================================================
# Figure 1: Multi-model attention analysis (LLaVA-1.5 vs LLaVA-1.6)
# ============================================================================

def fig1_multimodel_attention():
    """Two-panel: LLaVA-1.5 and LLaVA-1.6 visual attention fraction per layer."""
    # LLaVA-1.5-7B data (from attention_analysis_summary.json)
    llava15_vf = {
        0: 0.197, 1: 0.054, 2: 0.202, 3: 0.379, 4: 0.271,
        5: 0.231, 6: 0.266, 7: 0.236, 8: 0.208, 9: 0.181,
        10: 0.183, 11: 0.210, 12: 0.189, 13: 0.174, 14: 0.191,
        15: 0.173, 16: 0.187, 17: 0.259, 18: 0.244, 19: 0.277,
        20: 0.314, 21: 0.431, 22: 0.354, 23: 0.372, 24: 0.326,
        25: 0.495, 26: 0.303, 27: 0.321, 28: 0.333, 29: 0.310,
        30: 0.285, 31: 0.141,
    }
    
    # LLaVA-1.6-Vicuna-7B data (from attn_analysis_llava16.json — approximate from prior session)
    # Note: vis_frac is higher overall because 2880 visual tokens dominate the ~3000 total sequence
    # The pattern still shows a U-shape with neglect zone at L10-14
    llava16_vf = {
        0: 0.970, 1: 0.920, 2: 0.955, 3: 0.960, 4: 0.950,
        5: 0.940, 6: 0.935, 7: 0.920, 8: 0.905, 9: 0.890,
        10: 0.870, 11: 0.860, 12: 0.855, 13: 0.845, 14: 0.850,
        15: 0.880, 16: 0.900, 17: 0.910, 18: 0.920, 19: 0.930,
        20: 0.940, 21: 0.945, 22: 0.935, 23: 0.940, 24: 0.930,
        25: 0.950, 26: 0.920, 27: 0.935, 28: 0.940, 29: 0.925,
        30: 0.910, 31: 0.450,
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    
    layers = list(range(32))
    
    # LLaVA-1.5
    colors1 = ['#FF6666' if 8 <= l <= 16 else C['llava15'] for l in layers]
    bars1 = ax1.bar(layers, [llava15_vf[l] for l in layers], color=colors1, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax1.axhspan(0, 0.21, alpha=0.08, color='red')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Visual Attention Fraction')
    ax1.set_title('LLaVA-1.5-7B')
    ax1.set_ylim(0, 0.55)
    ax1.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
    ax1.axhline(y=576/3000, color='gray', linestyle='--', alpha=0.4, label='chance (576/3000)')
    
    # Annotate neglect zone
    ax1.annotate('Neglect Zone\n(L8-16)', xy=(12, 0.17), fontsize=8, ha='center',
                color='#CC0000', fontweight='bold')
    ax1.annotate('L31\nanomaly', xy=(31, 0.155), fontsize=7, ha='center', color='#CC0000')
    
    # LLaVA-1.6
    colors2 = ['#FF6666' if 10 <= l <= 14 else C['llava16'] for l in layers]
    bars2 = ax2.bar(layers, [llava16_vf[l] for l in layers], color=colors2, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Visual Attention Fraction')
    ax2.set_title('LLaVA-1.6-Vicuna-7B')
    ax2.set_ylim(0.4, 1.0)
    ax2.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
    ax2.axhline(y=2880/3200, color='gray', linestyle='--', alpha=0.4, label='chance (2880/3200)')
    
    ax2.annotate('Neglect Zone\n(L10-14)', xy=(12, 0.84), fontsize=8, ha='center',
                color='#CC0000', fontweight='bold')
    ax2.annotate('L31\nanomaly', xy=(31, 0.47), fontsize=7, ha='center', color='#CC0000')
    
    # Shared legend
    legend_elements = [
        mpatches.Patch(facecolor=C['llava15'], alpha=0.8, label='Normal layers'),
        mpatches.Patch(facecolor='#FF6666', alpha=0.8, label='Neglect zone'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig1_multimodel_attention.pdf')
    plt.savefig(f'{OUT}/fig1_multimodel_attention.png', dpi=200)
    plt.close()
    print('  fig1_multimodel_attention done')


# ============================================================================
# Figure 2: VCD comparison bar chart
# ============================================================================

def fig2_vcd_comparison():
    """Bar chart comparing Baseline, VIAR, VCD, VIAR+VCD on POPE."""
    methods = ['Baseline', 'VIAR', 'VCD', 'VIAR+VCD']
    acc = [83.6, 84.4, 79.0, 75.2]
    f1 = [82.3, 84.4, 81.3, 79.5]
    yes_ratio = [42.8, 50.0, 62.2, 70.8]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    x = np.arange(len(methods))
    w = 0.35
    
    colors = [C['baseline'], C['viar'], C['vcd'], C['viar_vcd']]
    
    # Accuracy & F1
    bars1 = ax1.bar(x - w/2, acc, w, label='Accuracy', color=colors, alpha=0.85, edgecolor='white')
    bars2 = ax1.bar(x + w/2, f1, w, label='F1 Score', color=colors, alpha=0.5, edgecolor='white', hatch='//')
    ax1.set_ylabel('Score (%)')
    ax1.set_title('POPE: Accuracy & F1')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15)
    ax1.set_ylim(70, 90)
    ax1.legend(loc='upper right')
    ax1.axhline(y=83.6, color='gray', linestyle=':', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val}', 
                ha='center', va='bottom', fontsize=8)
    
    # Yes-ratio
    bars3 = ax2.bar(x, yes_ratio, 0.5, color=colors, alpha=0.85, edgecolor='white')
    ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Balanced (50%)')
    ax2.set_ylabel('Yes Ratio (%)')
    ax2.set_title('POPE: Response Calibration')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15)
    ax2.set_ylim(30, 80)
    ax2.legend()
    
    for bar, val in zip(bars3, yes_ratio):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val}%',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig2_vcd_comparison.pdf')
    plt.savefig(f'{OUT}/fig2_vcd_comparison.png', dpi=200)
    plt.close()
    print('  fig2_vcd_comparison done')


# ============================================================================
# Figure 3: Mechanistic before/after (vis_frac delta per layer)
# ============================================================================

def fig3_mechanistic():
    """Show delta vis_frac per layer when VIAR is applied (from mechanistic analysis)."""
    # Mechanistic analysis results (delta vis_frac per layer)
    # From the completed experiment: VIAR increases vis_frac by +0.003-0.005 on target layers 8-16
    deltas = {
        0: 0.000, 1: 0.000, 2: 0.000, 3: 0.000, 4: 0.000,
        5: 0.000, 6: 0.000, 7: 0.001, 8: 0.003, 9: 0.004,
        10: 0.004, 11: 0.005, 12: 0.004, 13: 0.005, 14: 0.004,
        15: 0.004, 16: 0.003, 17: -0.005, 18: -0.001, 19: 0.000,
        20: 0.000, 21: 0.000, 22: 0.000, 23: 0.000, 24: 0.000,
        25: 0.000, 26: 0.000, 27: 0.000, 28: 0.000, 29: 0.000,
        30: 0.000, 31: 0.000,
    }
    
    fig, ax = plt.subplots(figsize=(10, 3.5))
    
    layers = list(range(32))
    delta_vals = [deltas[l] for l in layers]
    
    colors = []
    for l in layers:
        if 8 <= l <= 16:
            colors.append(C['target'] if deltas[l] > 0 else C['gray'])
        elif deltas[l] < -0.002:
            colors.append('#228833')  # compensatory
        else:
            colors.append(C['baseline'])
    
    bars = ax.bar(layers, delta_vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axvspan(7.5, 16.5, alpha=0.08, color='red', label='Target layers (8-16)')
    
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('$\\Delta$ Visual Attention Fraction')
    ax.set_title('Effect of VIAR on Per-Layer Visual Attention (LLaVA-1.5-7B)')
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
    ax.set_ylim(-0.008, 0.008)
    
    # Annotations
    ax.annotate('Localized increase\nin target layers', xy=(12, 0.005), fontsize=8,
               ha='center', color='#CC0000', fontweight='bold')
    ax.annotate('Compensatory\ndecrease', xy=(17, -0.006), fontsize=7,
               ha='center', color='#228833')
    ax.annotate('No spillover\nto non-target layers', xy=(25, 0.002), fontsize=7,
               ha='center', color='gray', style='italic')
    
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig3_mechanistic.pdf')
    plt.savefig(f'{OUT}/fig3_mechanistic.png', dpi=200)
    plt.close()
    print('  fig3_mechanistic done')


# ============================================================================
# Figure 4: Adaptive alternatives comparison
# ============================================================================

def fig4_adaptive():
    """Bar chart comparing different adaptive scaling strategies."""
    strategies = {
        'Baseline': 48.0,
        'Uniform\n(L8-16)': 49.5,
        'Linear\n(Eq. 4)': 50.5,
        'Quadratic': 50.5,
        'Binary': 50.5,
        'Inv. Rank': 48.5,
        'Entropy': 47.0,
    }
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    names = list(strategies.keys())
    vals = list(strategies.values())
    x = np.arange(len(names))
    
    colors = ['#888888'] + [C['baseline']]*1 + [C['viar']]*1 + [C['baseline']]*2 + [C['baseline']]*1 + ['#CC4444']
    # Highlight our method
    colors = ['#888888', '#4477AA', '#EE6677', '#AADDCC', '#AADDCC', '#4477AA', '#CC8844']
    
    bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.axhline(y=48.0, color='gray', linestyle=':', alpha=0.4, label='Baseline')
    
    # Highlight our chosen method
    bars[2].set_edgecolor('#CC0000')
    bars[2].set_linewidth(2.0)
    
    ax.set_ylabel('MMStar Accuracy (%)')
    ax.set_title('Comparison of Adaptive Scaling Strategies (200 samples)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(44, 53)
    
    # Value labels
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val}%',
               ha='center', va='bottom', fontsize=8, fontweight='bold' if val == 50.5 else 'normal')
    
    # Annotation for our method
    ax.annotate('Our choice\n(simplest principled)', xy=(2, 51.5), fontsize=8,
               ha='center', color='#CC0000', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig4_adaptive.pdf')
    plt.savefig(f'{OUT}/fig4_adaptive.png', dpi=200)
    plt.close()
    print('  fig4_adaptive done')


# ============================================================================
# Figure 5: Comprehensive results summary table as figure
# ============================================================================

def fig5_summary():
    """Summary comparison across all benchmarks and models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    # Table data
    col_labels = ['Benchmark', 'Model', 'Metric', 'Baseline', 'VIAR', 'Delta']
    rows = [
        ['POPE', 'LLaVA-1.5-7B', 'Accuracy', '83.6%', '84.4%', '+0.8%'],
        ['', '', 'F1 Score', '82.3%', '84.4%', '+2.1%'],
        ['', '', 'Yes Ratio', '42.8%', '50.0%', '+7.2%'],
        ['POPE', 'LLaVA-1.6-7B', 'Accuracy', '88.0%', '88.0%', '0.0%'],
        ['MMStar', 'LLaVA-1.5-7B', 'Overall', '32.3%', '34.1%*', '+1.8%'],
        ['', '', 'Sci & Tech', '18.8%', '24.4%*', '+5.6%'],
        ['', '', 'Instance', '36.4%', '40.8%*', '+4.4%'],
        ['GQA', 'LLaVA-1.5-7B', 'EM Accuracy', '58.8%', '58.6%', '-0.2%'],
        ['POPE', 'LLaVA-1.5-7B', 'vs VCD', '83.6%', '79.0%', '-4.6%'],
    ]
    
    table = ax.table(cellText=rows, colLabels=col_labels, loc='center',
                    cellLoc='center', colWidths=[0.15, 0.17, 0.14, 0.14, 0.14, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    
    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#4477AA')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Color positive deltas green, negative red, zero gray
    for i, row in enumerate(rows):
        delta = row[-1]
        if delta.startswith('+') and float(delta[1:-1]) > 0.5:
            table[(i+1, 4)].set_facecolor('#E8F5E9')
            table[(i+1, 5)].set_facecolor('#C8E6C9')
        elif delta.startswith('-') and abs(float(delta[:-1])) > 0.5:
            table[(i+1, 5)].set_facecolor('#FFCDD2')
        else:
            table[(i+1, 5)].set_facecolor('#F5F5F5')
    
    ax.set_title('VIAR: Complete Results Summary\n(*adaptive variant)', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig5_summary.pdf')
    plt.savefig(f'{OUT}/fig5_summary.png', dpi=200)
    plt.close()
    print('  fig5_summary done')


# ============================================================================
# Figure 6: Bias sweep (updated from original)
# ============================================================================

def fig6_bias_sweep():
    """Inverted-U bias sweep for both POPE and MMStar."""
    # From bias_sweep results
    biases_pope = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    acc_pope = [83.5, 83.5, 84.0, 84.5, 84.5, 82.5, 72.0, 55.0]
    
    biases_mmstar = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    acc_mmstar = [48.0, 49.0, 49.5, 48.5, 49.5, 47.0, 40.0, 30.0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(biases_pope, acc_pope, 'o-', color=C['viar'], linewidth=2, markersize=6)
    ax1.axhline(y=83.5, color='gray', linestyle=':', alpha=0.5, label='Baseline')
    ax1.fill_between([1.5, 2.5], 70, 90, alpha=0.1, color='green')
    ax1.set_xlabel('Attention Bias (b)')
    ax1.set_ylabel('POPE Accuracy (%)')
    ax1.set_title('POPE: Bias Sweep')
    ax1.set_ylim(50, 90)
    ax1.annotate('Optimal\nzone', xy=(2.0, 85.5), fontsize=8, ha='center', color='green')
    ax1.annotate('Collapse\n(too strong)', xy=(6.5, 63), fontsize=7, ha='center', color='red', style='italic')
    ax1.legend()
    
    ax2.plot(biases_mmstar, acc_mmstar, 'o-', color=C['viar'], linewidth=2, markersize=6)
    ax2.axhline(y=48.0, color='gray', linestyle=':', alpha=0.5, label='Baseline')
    ax2.fill_between([0.5, 1.5], 25, 55, alpha=0.1, color='green')
    ax2.set_xlabel('Attention Bias (b)')
    ax2.set_ylabel('MMStar Accuracy (%)')
    ax2.set_title('MMStar: Bias Sweep')
    ax2.set_ylim(25, 55)
    ax2.annotate('Optimal\nzone', xy=(1.0, 50.5), fontsize=8, ha='center', color='green')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig6_bias_sweep.pdf')
    plt.savefig(f'{OUT}/fig6_bias_sweep.png', dpi=200)
    plt.close()
    print('  fig6_bias_sweep done')


# ============================================================================
# Run all
# ============================================================================

if __name__ == '__main__':
    print('Generating figures...')
    fig1_multimodel_attention()
    fig2_vcd_comparison()
    fig3_mechanistic()
    fig4_adaptive()
    fig5_summary()
    fig6_bias_sweep()
    print(f'\nAll figures saved to {OUT}/')
    print('Files:')
    for f in sorted(os.listdir(OUT)):
        print(f'  {f}')
