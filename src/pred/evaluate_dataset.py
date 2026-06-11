import cv2
import json
import numpy as np
import os
import glob
import random
from ultralytics import YOLO

def expand_polygon(points, margin_x=48, margin_y=24):
    """
    Expand a 4-point polygon outwards by margin_y (top/bottom) and margin_x (left/right).
    points: 4x2 array [TL, TR, BR, BL]
    """
    expanded = []
    lines = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i+1)%4]
        v = p2 - p1
        length = np.linalg.norm(v)
        if length == 0:
            v_norm = np.array([1.0, 0.0])
        else:
            v_norm = v / length
        # Outward normal (assuming clockwise TL->TR->BR->BL)
        n = np.array([v_norm[1], -v_norm[0]])
        
        margin = margin_y if i % 2 == 0 else margin_x
        q = p1 + n * margin
        lines.append((q, v_norm))
        
    for i in range(4):
        q1, v1 = lines[(i-1)%4]
        q2, v2 = lines[i]
        A = np.column_stack((v1, -v2))
        b = q2 - q1
        try:
            t = np.linalg.solve(A, b)
            intersect = q1 + t[0] * v1
            expanded.append(intersect)
        except np.linalg.LinAlgError:
            expanded.append(points[i])
            
    return np.array(expanded)

def compute_iou(poly1, poly2, img_size=(3840, 2160)):
    # Calculate IoU using masks
    mask1 = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    mask2 = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    
    cv2.fillPoly(mask1, [np.int32(poly1)], 1)
    cv2.fillPoly(mask2, [np.int32(poly2)], 1)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    return intersection / union

import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO Pose on dataset")
    parser.add_argument("--model_path", type=str, default="", help="Path to the YOLO model weights")
    parser.add_argument("--out_dir", type=str, default="", help="Directory to save evaluation results")
    args = parser.parse_args()

    # evaluate_dataset.py 現在位於 src/pred/ 下，所以需要往上推三層才能到達專案根目錄
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    images_dir = os.path.join(base_dir, "data", "dataset_v1", "images")
    ann_dir = os.path.join(base_dir, "data", "dataset_v1", "annotations")
    
    model_path = args.model_path if args.model_path else os.path.join(base_dir, "result", "train_pose_17kpt_merged", "run-10", "weights", "best.pt")
    out_dir = args.out_dir if args.out_dir else os.path.join(base_dir, "result", "pred")
    vis_dir = os.path.join(out_dir, "visualizations")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    model = YOLO(model_path)
    
    NORMALIZED_POINTS_17 = np.array([
        [0.329828, 0.203805], [0.329828, 0.000000], [0.679026, 0.000000], 
        [0.679026, 0.203805], [0.892086, 0.208972], [1.000000, 0.278788], 
        [0.999447, 0.810260], [0.827892, 0.996325], [0.173215, 1.000000], 
        [0.000000, 0.810260], [0.008301, 0.278788], [0.116215, 0.206634], 
        [0.000000, 0.544524], [0.504427, 0.000000], [0.504427, 1.000000], 
        [0.504427, 0.544524], [1.000000, 0.544524],
    ], dtype=np.float32)
    OUTLINE_INDICES = list(range(12))
    
    target_w, target_h = 285, 468
    ideal_pts_17 = NORMALIZED_POINTS_17 * [target_w - 1, target_h - 1]
    ideal_pts_12 = ideal_pts_17[OUTLINE_INDICES]
    std_corners = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype=np.float32).reshape(-1, 1, 2)

    metrics = {
        "total_images": 0,
        "valid_predictions": 0,
        "avg_iou": 0.0,
        "avg_corner_mae": 0.0,
        "results": []
    }
    
    img_files = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg"))
    sample_size = min(20, len(img_files))
    sample_indices = set(random.sample(range(len(img_files)), sample_size))
    
    total_iou = 0.0
    total_mae = 0.0
    
    print(f"Found {len(img_files)} images in {images_dir}")
    
    for i, img_file in enumerate(img_files):
        basename = os.path.basename(img_file)
        name, _ = os.path.splitext(basename)
        ann_file = os.path.join(ann_dir, name + ".json")
        
        if not os.path.exists(ann_file):
            continue
            
        with open(ann_file, 'r', encoding='utf-8') as f:
            ann = json.load(f)
            
        gt_dict = ann.get('corners_4', {})
        if not gt_dict or 'TL' not in gt_dict:
            continue
            
        gt_corners = np.array([
            gt_dict['TL'], gt_dict['TR'], gt_dict['BR'], gt_dict['BL']
        ], dtype=np.float32)
        
        metrics["total_images"] += 1
        
        img = cv2.imread(img_file)
        if img is None:
            continue
            
        h, w = img.shape[:2]
            
        results = model.predict(img, conf=0.7, imgsz=640, device=0, verbose=False)
        
        success = False
        if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kpts_17 = results[0].keypoints.xy[0].cpu().numpy()[:17]
            if len(kpts_17) >= 17:
                src_outline = kpts_17[OUTLINE_INDICES].astype(np.float32)
                M, _ = cv2.findHomography(src_outline, ideal_pts_12, 0)
                if M is not None:
                    try:
                        M_inv = np.linalg.inv(M)
                        img_corners = cv2.perspectiveTransform(std_corners, M_inv).reshape(-1, 2)
                        
                        # Expand by margin_x=48, margin_y=24 pixels
                        pred_corners = expand_polygon(img_corners)
                        
                        iou = compute_iou(pred_corners, gt_corners, img_size=(w, h))
                        mae = np.mean(np.linalg.norm(pred_corners - gt_corners, axis=1))
                        
                        total_iou += iou
                        total_mae += mae
                        metrics["valid_predictions"] += 1
                        
                        metrics["results"].append({
                            "file": basename,
                            "iou": float(iou),
                            "mae": float(mae),
                            "pred_corners": pred_corners.tolist(),
                            "gt_corners": gt_corners.tolist()
                        })
                        success = True
                        
                        if i in sample_indices:
                            # Draw GT (Green)
                            pts_gt = np.array(gt_corners, np.int32).reshape((-1, 1, 2))
                            cv2.polylines(img, [pts_gt], isClosed=True, color=(0, 255, 0), thickness=3)
                            cv2.putText(img, "GT", (pts_gt[0][0][0], max(30, pts_gt[0][0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                            
                            # Draw Pred (Red)
                            pts_pred = np.array(pred_corners, np.int32).reshape((-1, 1, 2))
                            cv2.polylines(img, [pts_pred], isClosed=True, color=(0, 0, 255), thickness=3)
                            cv2.putText(img, "Pred", (pts_pred[0][0][0], max(30, pts_pred[0][0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                            
                            # Draw 17 keypoints and skeleton (Cyan)
                            skeleton = [(i, (i+1)%12) for i in range(12)] + [(12, 15), (15, 16), (13, 15), (15, 14)]
                            for p1, p2 in skeleton:
                                if p1 < len(kpts_17) and p2 < len(kpts_17):
                                    pt1 = (int(kpts_17[p1][0]), int(kpts_17[p1][1]))
                                    pt2 = (int(kpts_17[p2][0]), int(kpts_17[p2][1]))
                                    cv2.line(img, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)
                                    
                            for idx, pt in enumerate(kpts_17):
                                pt_x, pt_y = int(pt[0]), int(pt[1])
                                cv2.circle(img, (pt_x, pt_y), 6, (255, 255, 0), -1)
                                cv2.putText(img, str(idx), (pt_x+5, pt_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                                
                            text = f"IoU: {iou:.3f} | MAE: {mae:.1f}px"
                            cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                            
                            cv2.imwrite(os.path.join(vis_dir, basename), img)
                    except np.linalg.LinAlgError:
                        pass
        
        if not success:
            metrics["results"].append({
                "file": basename,
                "iou": 0.0,
                "mae": None,
                "error": "No valid prediction or homography"
            })
            
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(img_files)} images...")
            
    if metrics["valid_predictions"] > 0:
        metrics["avg_iou"] = float(total_iou / metrics["valid_predictions"])
        metrics["avg_corner_mae"] = float(total_mae / metrics["valid_predictions"])
        
    out_json = os.path.join(out_dir, "evaluation_report.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
        
    print(f"Evaluation complete. Results saved to {out_json}")
    print(f"Total Images evaluated: {metrics['total_images']}")
    print(f"Valid Predictions: {metrics['valid_predictions']}")
    print(f"Average IoU: {metrics['avg_iou']:.4f}")
    print(f"Average Corner MAE (pixels): {metrics['avg_corner_mae']:.4f}")

if __name__ == '__main__':
    main()
