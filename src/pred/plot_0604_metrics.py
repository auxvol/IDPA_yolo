import json
import os
import sys
import matplotlib.pyplot as plt

# 設定路徑
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
report_path = os.path.join(base_dir, "data", "0604_eval_output", "evaluation_report.json")
out_dir = os.path.join(base_dir, "data", "0604_eval_output")

def main():
    if not os.path.exists(report_path):
        print(f"錯誤: 找不到報表檔案 {report_path}")
        return

    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get("results", [])
    
    # 提取有效的 MAE 與 Accuracy 數值
    maes = []
    accs = []
    files = []
    for r in results:
        mae = r.get("mae")
        acc = r.get("accuracy")
        if mae is not None:
            maes.append(mae)
            if acc is not None:
                accs.append(acc)
            # 將副檔名去掉以節省空間
            files.append(os.path.splitext(r.get("file", ""))[0])

    if not maes:
        print("沒有找到任何有效的 MAE 數據可以繪製。")
        return

    # 1. 繪製 MAE 分佈直方圖 (Histogram)
    plt.figure(figsize=(10, 6))
    plt.hist(maes, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(sum(maes)/len(maes), color='red', linestyle='dashed', linewidth=2, label=f'Average MAE ({sum(maes)/len(maes):.1f})')
    plt.title('Distribution of 12-Point MAE (Pixels)', fontsize=16)
    plt.xlabel('MAE (Pixels)', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    hist_out = os.path.join(out_dir, "mae_histogram.png")
    plt.savefig(hist_out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"已生成直方圖: {hist_out}")

    # 2. 繪製各圖片 MAE 長條圖 (Bar Chart)
    plt.figure(figsize=(14, 6))
    x_pos = range(len(maes))
    plt.bar(x_pos, maes, color='lightcoral')
    plt.axhline(sum(maes)/len(maes), color='blue', linestyle='dashed', linewidth=2, label=f'Average MAE ({sum(maes)/len(maes):.1f})')
    plt.title('12-Point MAE per Image', fontsize=16)
    plt.xlabel('Image Index', fontsize=14)
    plt.ylabel('MAE (Pixels)', fontsize=14)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    bar_out = os.path.join(out_dir, "mae_per_image.png")
    plt.savefig(bar_out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"已生成各圖片 MAE 長條圖: {bar_out}")
    
    # 3. 繪製各圖片 Accuracy 長條圖 (Bar Chart)
    if accs:
        plt.figure(figsize=(14, 6))
        plt.bar(x_pos, accs, color='mediumseagreen')
        plt.axhline(sum(accs)/len(accs), color='darkgreen', linestyle='dashed', linewidth=2, label=f'Average Acc ({sum(accs)/len(accs):.1f}%)')
        plt.title('PCK Accuracy per Image', fontsize=16)
        plt.xlabel('Image Index', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.ylim(0, 110)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        acc_out = os.path.join(out_dir, "accuracy_per_image.png")
        plt.savefig(acc_out, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"已生成各圖片 Accuracy 長條圖: {acc_out}")

if __name__ == "__main__":
    main()
