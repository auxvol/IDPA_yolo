import os
import sys
import glob
import json
import argparse
import random
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# 加入根目錄以便載入 config
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from config.config import settings
from ultralytics import YOLO

# 註冊 HEIF 支援以讀取 .HEIC
import pillow_heif
from PIL import Image
pillow_heif.register_heif_opener()

def calculate_point_mae(pts_pred, pts_gt):
    """
    計算預測點與真實點之間的平均絕對誤差 (MAE)
    pts_pred: (N, 2) array
    pts_gt: (N, 2) array
    """
    # 計算每個點的歐幾里得距離
    distances = np.linalg.norm(pts_pred - pts_gt, axis=1)
    return np.mean(distances)

def parse_yolo_label(txt_path, img_w, img_h):
    """
    解析 YOLO Pose 標註檔，回傳前 12 個關鍵點的 (x, y) 座標
    """
    if not os.path.exists(txt_path):
        return None
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
            
        # YOLO 格式: class cx cy bw bh [kpx kpy kpv ...]
        kpts = []
        for i in range(5, len(parts), 3):
            if i + 2 < len(parts):
                kpx = float(parts[i])
                kpy = float(parts[i+1])
                kpv = float(parts[i+2])
                kpts.append([kpx * img_w, kpy * img_h, kpv])
                
        # 我們只取前 12 個點作為輪廓點進行評估
        if len(kpts) >= 12:
            pts_12 = np.array(kpts[:12], dtype=np.float32)
            # 檢查這 12 個點是否都有被標註 (v > 0)
            if np.all(pts_12[:, 2] > 0):
                bbox_w = float(parts[3]) * img_w
                bbox_h = float(parts[4]) * img_h
                return pts_12[:, :2], bbox_w, bbox_h # 回傳 (12, 2) 座標與 bbox 尺寸
    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO Pose on 0604.yolo26 (12-Point MAE)")
    parser.add_argument("--model_path", type=str, default="", help="Path to the YOLO model weights")
    parser.add_argument("--dataset_dir", type=str, default="", help="Path to 0604.yolo26/train")
    parser.add_argument("--out_dir", type=str, default="", help="Directory to save evaluation results")
    args = parser.parse_args()

    # 預設路徑處理
    dataset_dir = args.dataset_dir if args.dataset_dir else settings.get_path("data/0604.yolo26/train")
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    
    # 預設模型 (你可以替換成實際訓練出的模型)
    model_path = args.model_path if args.model_path else os.path.join(base_dir, "result", "train_pose_17kpt_merged", "run-10", "weights", "best.pt")
    
    out_dir = args.out_dir if args.out_dir else settings.get_path("data/0604_eval_output")
    vis_dir = os.path.join(out_dir, "visualizations")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"錯誤: 找不到模型權重檔案 {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # 取得圖片列表
    valid_exts = ('.heic', '.jpg', '.jpeg', '.png')
    img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]
    
    if not img_files:
        print(f"找不到任何圖片於 {images_dir}")
        return
        
    print(f"Found {len(img_files)} images in {images_dir}")
    
    # 為了避免評估全部太久，我們可以隨機抽樣 50 張或全部評估
    # 這裡預設全跑，但如果是為了展示，先抽樣
    sample_size = min(50, len(img_files))
    img_files = random.sample(img_files, sample_size)
    print(f"Randomly evaluating {sample_size} images...")

    metrics = {
        "total_evaluated": 0,
        "valid_predictions": 0,
        "avg_point_mae": 0.0,
        "avg_accuracy_pck": 0.0,
        "results": []
    }
    
    total_mae = 0.0
    total_accuracy = 0.0

    for i, img_file in enumerate(img_files):
        img_path = os.path.join(images_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        txt_path = os.path.join(labels_dir, base_name + ".txt")
        
        # 讀取圖片
        if img_file.lower().endswith('.heic'):
            try:
                pil_img = Image.open(img_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Error reading HEIC {img_file}: {e}")
                continue
        else:
            img = cv2.imread(img_path)
            
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # 讀取 GT 12 points 與 BBox
        gt_data = parse_yolo_label(txt_path, w, h)
        if gt_data is None:
            continue # 跳過沒有正確 12 個點標註的圖
        gt_pts_12, bbox_w, bbox_h = gt_data
            
        metrics["total_evaluated"] += 1
        
        # 模型推論
        results = model.predict(img, conf=0.5, imgsz=640, device=0, verbose=False)
        
        success = False
        mae = 0.0
        
        if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            # 取得預測的點座標
            pred_kpts = results[0].keypoints.xy[0].cpu().numpy()
            
            # 確保模型至少預測出了 12 個點
            if len(pred_kpts) >= 12:
                pred_pts_12 = pred_kpts[:12]
                
                # 自動校正: 使用 Hungarian Algorithm 將預測點與真實點配對
                dist_matrix = cdist(pred_pts_12, gt_pts_12)
                row_ind, col_ind = linear_sum_assignment(dist_matrix)
                
                # 依據 GT 的順序重新排列預測點
                ordered_pred_pts_12 = np.zeros_like(pred_pts_12)
                for p_idx, g_idx in zip(row_ind, col_ind):
                    ordered_pred_pts_12[g_idx] = pred_pts_12[p_idx]
                
                pred_pts_12 = ordered_pred_pts_12
                
                # 計算 MAE
                mae = calculate_point_mae(pred_pts_12, gt_pts_12)
                
                # 計算 PCK 準確度 (Threshold: 10% of max bbox dimension)
                threshold = 0.1 * max(bbox_w, bbox_h)
                distances = np.linalg.norm(pred_pts_12 - gt_pts_12, axis=1)
                correct_points = np.sum(distances < threshold)
                accuracy = (correct_points / 12.0) * 100.0
                
                total_mae += mae
                total_accuracy += accuracy
                metrics["valid_predictions"] += 1
                success = True
                
                # 視覺化輸出
                # 畫 GT (綠色)
                for pt in gt_pts_12:
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), -1)
                pts_gt_int = np.array(gt_pts_12, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts_gt_int], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # 畫 Pred (紅色)
                for pt in pred_pts_12:
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)
                pts_pred_int = np.array(pred_pts_12, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts_pred_int], isClosed=True, color=(0, 0, 255), thickness=2)
                
                # 印出字樣
                text = f"MAE: {mae:.2f} px | Acc: {accuracy:.1f}%"
                cv2.putText(img, text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 5)
                
                cv2.imwrite(os.path.join(vis_dir, base_name + ".jpg"), img)
                
                metrics["results"].append({
                    "file": img_file,
                    "mae": float(mae),
                    "accuracy": float(accuracy),
                    "pred_12_pts": pred_pts_12.tolist(),
                    "gt_12_pts": gt_pts_12.tolist()
                })
                
        if not success:
            metrics["results"].append({
                "file": img_file,
                "mae": None,
                "accuracy": None,
                "error": "Model did not predict 12 keypoints or no detection"
            })
            
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{sample_size} images...")
            
    if metrics["valid_predictions"] > 0:
        metrics["avg_point_mae"] = float(total_mae / metrics["valid_predictions"])
        metrics["avg_accuracy_pck"] = float(total_accuracy / metrics["valid_predictions"])
        
    out_json = os.path.join(out_dir, "evaluation_report.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
        
    print(f"\nEvaluation Complete! Results saved to {out_dir}")
    print(f"Total Evaluated: {metrics['total_evaluated']}")
    print(f"Valid Predictions: {metrics['valid_predictions']}")
    if metrics["valid_predictions"] > 0:
        print(f"Average 12-Point MAE: {metrics['avg_point_mae']:.2f} pixels")
        print(f"Average Accuracy (PCK@0.1): {metrics['avg_accuracy_pck']:.2f}%")

if __name__ == "__main__":
    main()
