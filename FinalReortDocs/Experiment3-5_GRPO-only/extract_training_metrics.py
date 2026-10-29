"""
Extract training metrics — Experiment 3-5 (GRPO-only)
Read training metrics from a SLURM log and save them as CSV

Log file: slurm-4871595-3-5-grpo-train-full-set.out
Experiment: GRPO-only (no SFT pretraining)
"""

import re
import json
import pandas as pd
from pathlib import Path

# Config
LOG_FILE = "../../output_archive/slurm-4871595-3-5-grpo-train-full-set.out"
OUTPUT_CSV = "training_metrics.csv"

def extract_metrics_from_log(log_file_path):
    """
    Extract training metrics from the log file
    
    Example log line:
    {'loss': -0.0626, 'grad_norm': 0.4672642946243286, 
     'learning_rate': 1.3368983957219251e-07, ...}
    """
    metrics_list = []
    
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Match dictionary lines that contain metrics
    pattern = r"\{'loss':.*?\}"
    matches = re.findall(pattern, content)
    
    print(f"Found {len(matches)} training records")
    
    for match in matches:
        try:
            # Turn the string into a dict
            # Replace single quotes with double quotes for JSON
            json_str = match.replace("'", '"')
            metrics = json.loads(json_str)
            metrics_list.append(metrics)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try eval
            try:
                metrics = eval(match)
                metrics_list.append(metrics)
            except Exception as e:
                print(f"Skipping an invalid record: {e}")
                continue
    
    return metrics_list

def main():
    print("="*70)
    print("🔍 Experiment 3-5 (GRPO-only) metric extraction")
    print("="*70)
    
    # Check that the log file exists
    log_path = Path(LOG_FILE)
    if not log_path.exists():
        print(f"❌ Error: log file not found {LOG_FILE}")
        return
    
    print(f"\n📂 Reading log file: {LOG_FILE}")
    
    # Extract metrics
    metrics_list = extract_metrics_from_log(log_path)
    
    if not metrics_list:
        print("❌ No training-metric data found")
        return
    
    # Convert to a DataFrame
    df = pd.DataFrame(metrics_list)
    
    # Add a step column if it is missing
    if 'step' not in df.columns:
        df.insert(0, 'step', range(len(df)))
    
    print(f"\n📊 Metrics extracted: {len(df)} records")
    print(f"📋 Metric columns: {len(df.columns)} columns")
    
    # Show the first rows
    print("\nPreview of the first 5 rows:")
    print(df.head())
    
    # Save as CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Data saved to: {OUTPUT_CSV}")
    
    # Print the summary
    print("\n" + "="*70)
    print("📈 Training summary")
    print("="*70)
    
    if 'loss' in df.columns:
        print(f"Loss: {df['loss'].iloc[0]:.4f} → {df['loss'].iloc[-1]:.4f}")
    
    if 'learning_rate' in df.columns:
        print(f"Learning Rate: {df['learning_rate'].iloc[0]:.2e} → {df['learning_rate'].iloc[-1]:.2e}")
    
    if 'rewards/reward_func/mean' in df.columns:
        initial_acc = df['rewards/reward_func/mean'].iloc[0] * 100
        final_acc = df['rewards/reward_func/mean'].iloc[-1] * 100
        peak_acc = df['rewards/reward_func/mean'].max() * 100
        mean_acc = df['rewards/reward_func/mean'].mean() * 100
        print(f"Accuracy (reward mean):")
        print(f"  Initial: {initial_acc:.1f}%")
        print(f"  Final: {final_acc:.1f}%")
        print(f"  Peak: {peak_acc:.1f}%")
        print(f"  Mean: {mean_acc:.1f}%")
    
    if 'kl' in df.columns:
        print(f"KL Divergence: {df['kl'].iloc[0]:.6f} → {df['kl'].iloc[-1]:.6f}")
        print(f"  Max KL: {df['kl'].max():.6f}")
    
    if 'entropy' in df.columns:
        print(f"Entropy: {df['entropy'].iloc[0]:.4f} → {df['entropy'].iloc[-1]:.4f}")
    
    if 'grad_norm' in df.columns:
        print(f"Gradient Norm: Mean={df['grad_norm'].mean():.4f}, Max={df['grad_norm'].max():.4f}")
    
    print("="*70)

if __name__ == "__main__":
    main()
