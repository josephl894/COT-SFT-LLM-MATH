"""
GRPO training reward statistics

What it does:
1. Load the reward-statistics JSON log recorded during training
2. Plot training trends
3. Print the main statistics

Usage:
    # Basic use (find the log automatically)
    python utilities_snippet/analyze_reward_logs.py
    
    # Pass a log path
    python utilities_snippet/analyze_reward_logs.py --log_file qwen_grpo_lora_multigpu/reward_statistics.json
    
    # Save the figure to a given path
    python utilities_snippet/analyze_reward_logs.py --output reward_analysis.png
    
    # Print statistics only, no plot
    python utilities_snippet/analyze_reward_logs.py --no_plot

Output:
    - Terminal: print a training summary
    - Figure: mean_reward, accuracy, and std_reward versus training step
    - File: saved as reward_analysis.png by default

Dependencies:
    pip install matplotlib numpy
"""

import json
import argparse
import os
from pathlib import Path

def load_reward_logs(log_file):
    """Load the reward-statistics JSON log"""
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    if not logs:
        raise ValueError("Log file is empty!")
    
    print(f"✅ Loaded log: {log_file}")
    print(f"📊 Log entries: {len(logs)}")
    return logs


def print_summary_statistics(logs):
    """Print the training summary"""
    import numpy as np
    
    # Extract data
    mean_rewards = [log["mean_reward"] for log in logs]
    accuracies = [log["accuracy"] for log in logs]
    std_rewards = [log["std_reward"] for log in logs]
    
    # Compute overall statistics
    print("\n" + "="*70)
    print("📈 Training summary")
    print("="*70)
    
    print(f"\n🔢 Reward-function calls:")
    print(f"   Total calls: {logs[-1]['call_count']}")
    print(f"   Logged entries: {len(logs)}")
    
    print(f"\n📊 Mean reward:")
    print(f"   Initial: {mean_rewards[0]:.4f}")
    print(f"   Final: {mean_rewards[-1]:.4f}")
    print(f"   Max: {max(mean_rewards):.4f} (record {mean_rewards.index(max(mean_rewards))+1})")
    print(f"   Min: {min(mean_rewards):.4f} (record {mean_rewards.index(min(mean_rewards))+1})")
    print(f"   Change: {mean_rewards[-1] - mean_rewards[0]:+.4f}")
    
    print(f"\n🎯 Accuracy:")
    print(f"   Initial: {accuracies[0]:.2%}")
    print(f"   Final: {accuracies[-1]:.2%}")
    print(f"   Highest: {max(accuracies):.2%} (record {accuracies.index(max(accuracies))+1})")
    print(f"   Lowest: {min(accuracies):.2%} (record {accuracies.index(min(accuracies))+1})")
    print(f"   Change: {(accuracies[-1] - accuracies[0])*100:+.2f}%")
    
    print(f"\n📉 Std reward:")
    print(f"   Initial: {std_rewards[0]:.4f}")
    print(f"   Final: {std_rewards[-1]:.4f}")
    print(f"   Mean: {np.mean(std_rewards):.4f}")
    
    # Health check
    print(f"\n🏥 Training health check:")
    
    # Check 1: did accuracy rise
    if accuracies[-1] > accuracies[0]:
        print("   ✅ Accuracy rose")
    else:
        print("   ❌ Accuracy fell or stayed flat")
    
    # Check 2: did mean reward rise
    if mean_rewards[-1] > mean_rewards[0]:
        print("   ✅ Mean reward rose")
    else:
        print("   ❌ Mean reward fell or stayed flat")
    
    # Check 3: is there a gradient signal (std not 0)
    if np.mean(std_rewards) > 0.1:
        print("   ✅ Useful gradient signal (std > 0.1)")
    else:
        print("   ⚠️  Weak gradient signal (std < 0.1)")
    
    # Check 4: is the reward spread reasonable (not all 0 or all 1)
    avg_acc = np.mean(accuracies)
    if 0.1 < avg_acc < 0.9:
        print("   ✅ Reward spread looks reasonable (10%-90%)")
    elif avg_acc < 0.1:
        print("   ❌ Reward too low (almost all wrong)")
    else:
        print("   ⚠️  Reward too high (almost all correct; possible overfitting)")
    
    print("="*70 + "\n")


def plot_reward_trends(logs, output_file="reward_analysis.png"):
    """Plot reward trends"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("❌ matplotlib is missing, cannot plot")
        print("   Install it: pip install matplotlib")
        return
    
    # Extract data
    calls = [log["call_count"] for log in logs]
    mean_rewards = [log["mean_reward"] for log in logs]
    accuracies = [log["accuracy"] for log in logs]
    std_rewards = [log["std_reward"] for log in logs]
    
    # Create the figure (3 panels)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Panel 1: mean reward
    axes[0].plot(calls, mean_rewards, linewidth=2, color='blue', marker='o', markersize=3)
    axes[0].set_title("Mean Reward over Training", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Reward Function Calls")
    axes[0].set_ylabel("Mean Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Neutral (0.5)')
    axes[0].legend()
    
    # Panel 2: accuracy
    axes[1].plot(calls, [acc * 100 for acc in accuracies], linewidth=2, color='green', marker='s', markersize=3)
    axes[1].set_title("Accuracy (% Correct Answers) over Training", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Reward Function Calls")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 105])
    
    # Panel 3: standard deviation
    axes[2].plot(calls, std_rewards, linewidth=2, color='orange', marker='^', markersize=3)
    axes[2].set_title("Standard Deviation of Rewards over Training", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Reward Function Calls")
    axes[2].set_ylabel("Std Reward")
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Weak Signal (0.1)')
    axes[2].legend()
    
    # Tighten the layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Figure saved: {output_file}")
    
    # Show the figure
    try:
        plt.show()
    except:
        print("⚠️  Cannot display the figure (maybe an SSH session), but it was saved")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the GRPO reward-statistics log",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (find the log automatically)
  python utilities_snippet/analyze_reward_logs.py
  
  # Pass a log file
  python utilities_snippet/analyze_reward_logs.py --log_file path/to/reward_statistics.json
  
  # Statistics only, no plot
  python utilities_snippet/analyze_reward_logs.py --no_plot
  
  # Custom figure path
  python utilities_snippet/analyze_reward_logs.py --output my_analysis.png
        """
    )
    
    parser.add_argument(
        "--log_file", 
        type=str, 
        default="qwen_grpo_lora_multigpu/reward_statistics.json",
        help="Reward-statistics JSON log path (default: qwen_grpo_lora_multigpu/reward_statistics.json)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="reward_analysis.png",
        help="Output figure path (default: reward_analysis.png)"
    )
    
    parser.add_argument(
        "--no_plot", 
        action="store_true",
        help="Do not plot; print statistics only"
    )
    
    args = parser.parse_args()
    
    # Load the log
    try:
        logs = load_reward_logs(args.log_file)
    except Exception as e:
        print(f"❌ Failed to load the log: {e}")
        return
    
    # Print the summary
    print_summary_statistics(logs)
    
    # Plot trends (unless disabled)
    if not args.no_plot:
        plot_reward_trends(logs, args.output)


if __name__ == "__main__":
    main()
