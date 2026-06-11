import os
import sys
import glob
import json
import subprocess
import matplotlib.pyplot as plt
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    runs_dir = os.path.join(base_dir, "result", "train_pose_17kpt_merged")
    eval_script = os.path.join(base_dir, "src", "pred", "evaluate_dataset.py")
    python_exe = sys.executable

    # Find all run directories
    run_dirs = glob.glob(os.path.join(runs_dir, "run*"))
    valid_runs = []

    for run_dir in run_dirs:
        model_path = os.path.join(run_dir, "weights", "best.pt")
        if os.path.exists(model_path):
            run_name = os.path.basename(run_dir)
            valid_runs.append((run_name, model_path))

    # Sort runs logically (e.g., run, run-2, run-3...)
    def get_run_number(run_name):
        if run_name == "run": return 1
        return int(run_name.split("-")[1])
    
    valid_runs.sort(key=lambda x: get_run_number(x[0]))

    print(f"Found {len(valid_runs)} valid runs to evaluate.")

    results_data = []

    for run_name, model_path in valid_runs:
        print(f"\n{'='*40}")
        print(f"Evaluating {run_name}...")
        print(f"{'='*40}")
        
        out_dir = os.path.join(base_dir, "result", "pred", run_name)
        
        # Run evaluation script
        cmd = [
            python_exe, eval_script,
            "--model_path", model_path,
            "--out_dir", out_dir
        ]
        
        subprocess.run(cmd, check=True)
        
        # Read the generated report
        report_path = os.path.join(out_dir, "evaluation_report.json")
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results_data.append({
                    "run_name": run_name,
                    "avg_iou": data.get("avg_iou", 0),
                    "avg_corner_mae": data.get("avg_corner_mae", 0)
                })
        else:
            print(f"Warning: {report_path} not found.")

    if not results_data:
        print("No evaluation results were gathered.")
        return

    # Plotting
    run_names = [d["run_name"] for d in results_data]
    ious = [d["avg_iou"] for d in results_data]
    maes = [d["avg_corner_mae"] for d in results_data]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Training Run')
    ax1.set_ylabel('Average IoU', color=color)
    ax1.bar(run_names, ious, color=color, alpha=0.6, label='Avg IoU')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 1.0])

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Average Corner MAE (pixels)', color=color)  
    ax2.plot(run_names, maes, color=color, marker='o', linewidth=2, label='Avg MAE')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, max(maes) * 1.2 if maes else 100])

    fig.tight_layout()  
    plt.title('YOLO Pose Evaluation Metrics Across Runs')
    
    chart_path = os.path.join(base_dir, "result", "pred", "comparison_chart.png")
    plt.savefig(chart_path)
    plt.close()
    
    print(f"\nBatch evaluation complete. Chart saved to {chart_path}")
    
    # Save a summary JSON
    summary_path = os.path.join(base_dir, "result", "pred", "batch_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4)
        
if __name__ == "__main__":
    main()
