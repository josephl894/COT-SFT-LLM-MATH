"""
Training-metric plots — Experiment 3-5 (GRPO-only) — LARGE
Font setup: large (posters and large screens)

Font sizes:
- Main title: 26pt
- Panel title: 20pt
- XYAxis label: 17pt
- Tick labels: 15pt
- Legend: 15pt
- Footer note: 12pt
- Font: Arial
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set the style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 11)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15

# Config
CSV_FILE = "training_metrics.csv"
OUTPUT_PNG = "training_metrics_visualization_LARGE.svg"
OUTPUT_DIR = "../../output_archive"

def plot_with_smooth(ax, x, y, color, label, window=50, alpha_raw=0.15):
    """Plot the raw series and a smoothed curve"""
    # Raw data (translucent)
    ax.plot(x, y, color=color, alpha=alpha_raw, linewidth=0.8)
    
    # Smoothed data (rolling average)
    y_smooth = pd.Series(y).rolling(window=window, min_periods=1, center=False).mean()
    ax.plot(x, y_smooth, color=color, linewidth=2.5, label=label)
    
    return y_smooth

def main():
    print("="*70)
    print("📊 Experiment 3-5 (GRPO-only) training plot — LARGE")
    print("="*70)
    
    # Read the CSV
    if not Path(CSV_FILE).exists():
        print(f"❌ Error: not found {CSV_FILE}")
        print("   Run extract_training_metrics.py first")
        return
    
    df = pd.read_csv(CSV_FILE)
    print(f"\n✅ Rows read: {len(df)} records")
    
    # Create the figure
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1], hspace=0.35, wspace=0.45)
    
    # Colors
    colors = sns.color_palette("husl", 5)
    
    # ==========================================================================
    # Main panel: accuracy
    # ==========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    reward_col = 'rewards/reward_func/mean'
    if reward_col in df.columns:
        accuracy = df[reward_col] * 100
        plot_with_smooth(ax1, df['step'], accuracy, colors[0], 'Smoothed trend (rolling avg.)', window=50)
        
        ax1.set_xlabel('Training Step', fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontweight='bold')
        ax1.set_title('(a) Model Performance During GRPO Training (GRPO-only, no SFT)', 
                     fontsize=26, fontweight='bold', pad=15)
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Hide the top and right spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
    
    # ==========================================================================
    # Panel (b): training loss
    # ==========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    if 'loss' in df.columns:
        plot_with_smooth(ax2, df['step'], df['loss'], colors[1], 'Smoothed', window=50)
        ax2.set_xlabel('Training Step', fontweight='bold')
        ax2.set_ylabel('Loss', fontweight='bold')
        ax2.set_title('(b) Training Loss', fontsize=18, fontweight='bold', pad=12)
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    
    # ==========================================================================
    # Panel (c): KL divergence
    # ==========================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    if 'kl' in df.columns:
        plot_with_smooth(ax3, df['step'], df['kl'], colors[2], 'Smoothed', window=50)
        ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.6, linewidth=1.5, 
                   label='Warning threshold (0.05)')
        ax3.set_xlabel('Training Step', fontweight='bold')
        ax3.set_ylabel('KL Divergence', fontweight='bold')
        ax3.set_title('(c) KL Divergence from Base Model', fontsize=18, fontweight='bold', pad=12)
        ax3.grid(True, alpha=0.3)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
    
    # ==========================================================================
    # Panel (d): policy entropy
    # ==========================================================================
    ax4 = fig.add_subplot(gs[1, 2])
    if 'entropy' in df.columns:
        plot_with_smooth(ax4, df['step'], df['entropy'], colors[3], 'Smoothed', window=50)
        ax4.set_xlabel('Training Step', fontweight='bold')
        ax4.set_ylabel('Policy Entropy', fontweight='bold')
        ax4.set_title('(d) Policy Entropy', fontsize=18, fontweight='bold', pad=12)
        ax4.grid(True, alpha=0.3)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
    
    # ==========================================================================
    # Panel (e): gradient norm
    # ==========================================================================
    ax5 = fig.add_subplot(gs[1, 3])
    if 'grad_norm' in df.columns:
        plot_with_smooth(ax5, df['step'], df['grad_norm'], colors[4], 'Smoothed', window=50)
        ax5.axhline(y=1.0, color='orange', linestyle='--', alpha=0.6, linewidth=1.5,
                   label='High value warning (1.0)')
        ax5.set_xlabel('Training Step', fontweight='bold')
        ax5.set_ylabel('Gradient Norm', fontweight='bold')
        ax5.set_title('(e) Gradient Norm', fontsize=18, fontweight='bold', pad=12)
        ax5.grid(True, alpha=0.3)
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
    
    # Save the figure
    plt.savefig(OUTPUT_PNG, format='svg', bbox_inches='tight', pad_inches=0.2)
    print(f"\n✅ Figure saved to: {OUTPUT_PNG}")
    print(f"   Fonts: LARGE (main title 26pt, panel title 18pt, axis label 17pt, figure 18x11)")
    
    # Also save a copy under output_archive
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    archive_path = Path(OUTPUT_DIR) / OUTPUT_PNG
    plt.savefig(archive_path, format='svg', bbox_inches='tight', pad_inches=0.2)
    print(f"📦 Backup saved to: {archive_path}")
    
    # Print the summary
    print("\n" + "="*70)
    print("📈 GRPO-only Training Performance Summary")
    print("="*70)
    
    if reward_col in df.columns:
        initial_acc = df[reward_col].iloc[0] * 100
        final_acc = df[reward_col].iloc[-1] * 100
        peak_acc = df[reward_col].max() * 100
        mean_acc = df[reward_col].mean() * 100
        
        print(f"\n   Initial Accuracy:    {initial_acc:.1f}%")
        print(f"   Final Accuracy:      {final_acc:.1f}%")
        print(f"   Peak Accuracy:       {peak_acc:.1f}%")
        print(f"   Improvement:         +{final_acc-initial_acc:.1f}%")
    
    print("="*70)

if __name__ == "__main__":
    main()
