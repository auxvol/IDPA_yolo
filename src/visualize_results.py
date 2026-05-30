import os
import json
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_polygon(img, points, color, thickness=2, label=""):
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    if label:
        # Put text near the top-left point
        tl = np.min(pts, axis=0)[0]
        cv2.putText(img, label, (tl[0], max(30, tl[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_file = os.path.join(base_dir, "result", "pred", "evaluation_report.json")
    images_dir = os.path.join(base_dir, "data", "dataset_v1", "images")
    vis_dir = os.path.join(base_dir, "result", "pred", "visualizations")
    
    os.makedirs(vis_dir, exist_ok=True)
    
    with open(report_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
        
    results = metrics.get("results", [])
    valid_results = [r for r in results if r.get("iou") is not None and r.get("iou") > 0]
    
    if not valid_results:
        print("No valid results found in report.")
        return
        
    # 1. Plot distributions
    ious = [r["iou"] for r in valid_results]
    maes = [r["mae"] for r in valid_results if r.get("mae") is not None]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(ious, bins=20, color='skyblue', edgecolor='black')
    plt.title("IoU Distribution")
    plt.xlabel("IoU")
    plt.ylabel("Count")
    
    plt.subplot(1, 2, 2)
    plt.hist(maes, bins=20, color='salmon', edgecolor='black')
    plt.title("Corner MAE Distribution (pixels)")
    plt.xlabel("MAE")
    plt.ylabel("Count")
    
    plot_path = os.path.join(base_dir, "result", "pred", "metrics_distribution.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Saved metric distributions to {plot_path}")
    
    # 2. Randomly select 20 images and draw
    sample_size = min(20, len(valid_results))
    sampled_results = random.sample(valid_results, sample_size)
    
    for r in sampled_results:
        img_name = r["file"]
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        gt_corners = r["gt_corners"]
        pred_corners = r["pred_corners"]
        
        # Draw GT in Green
        draw_polygon(img, gt_corners, color=(0, 255, 0), thickness=3, label="GT")
        
        # Draw Pred in Red
        draw_polygon(img, pred_corners, color=(0, 0, 255), thickness=3, label="Pred")
        
        # Add IoU and MAE text
        iou = r["iou"]
        mae = r["mae"]
        text = f"IoU: {iou:.3f} | MAE: {mae:.1f}px"
        cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        out_path = os.path.join(vis_dir, img_name)
        cv2.imwrite(out_path, img)
        
    print(f"Saved {sample_size} visualized images to {vis_dir}")

if __name__ == '__main__':
    main()
