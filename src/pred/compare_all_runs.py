import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pred_dir = os.path.join(base_dir, "result", "pred")
    
    # 1. 收集所有的 run 資料夾中的 evaluation_report.json
    run_reports = glob.glob(os.path.join(pred_dir, "run*", "evaluation_report.json"))
    
    # 2. 收集根目錄的 evaluation_report.json (這應該是 run-10)
    root_report = os.path.join(pred_dir, "evaluation_report.json")
    if os.path.exists(root_report):
        run_reports.append(root_report)
        
    results_data = []

    for report_path in run_reports:
        with open(report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 從路徑判斷 run 的名稱
            parent_dir = os.path.basename(os.path.dirname(report_path))
            if parent_dir == "pred":
                run_name = "run-10 (Latest)"
            else:
                run_name = parent_dir
                
            results_data.append({
                "run_name": run_name,
                "avg_iou": data.get("avg_iou", 0),
                "avg_corner_mae": data.get("avg_corner_mae", 0),
                "valid_predictions": data.get("valid_predictions", 0),
                "total_images": data.get("total_images", 600)
            })

    if not results_data:
        print("沒有找到任何評估報告！")
        return

    # 根據 run 名稱進行排序，確保順序合理
    def get_run_num(name):
        if name == "run": return 1
        if "Latest" in name: return 999
        try:
            return int(name.split("-")[1])
        except:
            return 99
            
    results_data.sort(key=lambda x: get_run_num(x["run_name"]))

    run_names = [d["run_name"] for d in results_data]
    ious = [d["avg_iou"] for d in results_data]
    maes = [d["avg_corner_mae"] for d in results_data]
    valids = [d["valid_predictions"] for d in results_data]

    # --- 圖表 1: Average IoU 比較 ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(run_names, ious, color='skyblue', edgecolor='black')
    plt.title('Average IoU Comparison Across Runs', fontsize=14)
    plt.ylabel('Average IoU', fontsize=12)
    plt.ylim(0.85, 1.0) # 放大数据差异
    plt.xticks(rotation=45)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.3f}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(pred_dir, "chart_iou.png"))
    plt.close()

    # --- 圖表 2: Average Corner MAE 比較 ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(run_names, maes, color='salmon', edgecolor='black')
    plt.title('Average Corner MAE (Pixels) Comparison', fontsize=14)
    plt.ylabel('MAE (lower is better)', fontsize=12)
    plt.xticks(rotation=45)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(pred_dir, "chart_mae.png"))
    plt.close()

    # --- 圖表 3: Valid Predictions 比較 ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(run_names, valids, color='lightgreen', edgecolor='black')
    plt.title('Valid Predictions (Out of 600) Comparison', fontsize=14)
    plt.ylabel('Valid Count', fontsize=12)
    plt.ylim(500, 610)
    plt.xticks(rotation=45)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{int(yval)}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(pred_dir, "chart_valid.png"))
    plt.close()
    
    # --- 圖表 4: IoU vs MAE 散佈圖 (Pareto Frontier) ---
    plt.figure(figsize=(10, 8))
    plt.scatter(maes, ious, color='purple', s=100)
    for i, txt in enumerate(run_names):
        plt.annotate(txt, (maes[i], ious[i]), xytext=(5, 5), textcoords='offset points', fontsize=10)
    plt.title('IoU vs MAE (Top-Left is Better)', fontsize=14)
    plt.xlabel('Average Corner MAE (pixels)', fontsize=12)
    plt.ylabel('Average IoU', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(pred_dir, "chart_scatter.png"))
    plt.close()

    print(f"成功生成 4 張對比圖表，儲存於 {pred_dir} 中：")
    print("  1. chart_iou.png")
    print("  2. chart_mae.png")
    print("  3. chart_valid.png")
    print("  4. chart_scatter.png")

if __name__ == "__main__":
    main()
