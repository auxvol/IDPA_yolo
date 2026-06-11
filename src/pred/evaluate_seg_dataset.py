import cv2
import json
import numpy as np
import os
import glob
import random
from ultralytics import YOLO
import itertools

DILATE_BBOX_RATIO = 0.05
USE_HOUGHLINES = True

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

def fit_4_lines_and_intersect(contour_pts):
    orig_bx, orig_by, orig_bw, orig_bh = cv2.boundingRect(contour_pts)
    bx, by, bw, bh = orig_bx, orig_by, orig_bw, orig_bh
    
    if DILATE_BBOX_RATIO > 0:
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        bw = bw * (1.0 + DILATE_BBOX_RATIO * 2)
        bh = bh * (1.0 + DILATE_BBOX_RATIO * 2)
        bx = cx - bw / 2.0
        by = cy - bh / 2.0
    
    left_pts   = [p for p in contour_pts if p[0] < bx + 0.25*bw and by + 0.2*bh < p[1] < by + 0.6*bh]
    right_pts  = [p for p in contour_pts if p[0] > bx + 0.75*bw and by + 0.2*bh < p[1] < by + 0.6*bh]
    top_pts    = [p for p in contour_pts if p[1] < by + 0.25*bh and bx + 0.3*bw < p[0] < bx + 0.7*bw]
    bottom_pts = [p for p in contour_pts if p[1] > by + 0.75*bh and bx + 0.4*bw < p[0] < bx + 0.6*bw]
    
    zones_lines = []
    # Top, Right, Bottom, Left
    for i, pts in enumerate([top_pts, right_pts, bottom_pts, left_pts]):
        zone_candidates = []
        if len(pts) < 5:
            if i == 0:  l = np.array([[1.0], [0.0], [orig_bx + orig_bw/2.0], [orig_by]], dtype=np.float32)
            elif i == 1: l = np.array([[0.0], [1.0], [orig_bx + orig_bw], [orig_by + orig_bh/2.0]], dtype=np.float32)
            elif i == 2: l = np.array([[1.0], [0.0], [orig_bx + orig_bw/2.0], [orig_by + orig_bh]], dtype=np.float32)
            elif i == 3: l = np.array([[0.0], [1.0], [orig_bx], [orig_by + orig_bh/2.0]], dtype=np.float32)
            zone_candidates.append(l)
        else:
            if not USE_HOUGHLINES:
                l = cv2.fitLine(np.array(pts, dtype=np.float32), cv2.DIST_L1, 0, 0.01, 0.01)
                zone_candidates.append(l)
            else:
                pts_arr = np.array(pts, dtype=np.int32)
                min_x, min_y = np.min(pts_arr, axis=0) - 5
                max_x, max_y = np.max(pts_arr, axis=0) + 5
                img_w, img_h = max_x - min_x, max_y - min_y
                
                if img_w > 0 and img_h > 0:
                    canvas = np.zeros((img_h, img_w), dtype=np.uint8)
                    local_pts = pts_arr - [min_x, min_y]
                    cv2.polylines(canvas, [local_pts.reshape(-1,1,2)], False, 255, 1)
                    
                    min_len = max(img_w, img_h) * 0.15
                    h_lines = cv2.HoughLinesP(canvas, rho=1, theta=np.pi/180, threshold=8, minLineLength=min_len, maxLineGap=10)
                    
                    if h_lines is not None and len(h_lines) > 0:
                        for hl in h_lines:
                            lx1, ly1, lx2, ly2 = hl[0][0] + min_x, hl[0][1] + min_y, hl[0][2] + min_x, hl[0][3] + min_y
                            vx = lx2 - lx1
                            vy = ly2 - ly1
                            norm = np.hypot(vx, vy)
                            if norm > 0:
                                zone_candidates.append(np.array([[vx/norm], [vy/norm], [lx1], [ly1]], dtype=np.float32))
                                
            if len(zone_candidates) == 0:
                if i == 0:  l = np.array([[1.0], [0.0], [orig_bx + orig_bw/2.0], [orig_by]], dtype=np.float32)
                elif i == 1: l = np.array([[0.0], [1.0], [orig_bx + orig_bw], [orig_by + orig_bh/2.0]], dtype=np.float32)
                elif i == 2: l = np.array([[1.0], [0.0], [orig_bx + orig_bw/2.0], [orig_by + orig_bh]], dtype=np.float32)
                elif i == 3: l = np.array([[0.0], [1.0], [orig_bx], [orig_by + orig_bh/2.0]], dtype=np.float32)
                zone_candidates.append(l)

        if len(zone_candidates) > 5:
            zone_candidates = zone_candidates[:5]
        zones_lines.append(zone_candidates)
        
    def get_intersect(l1, l2):
        vx1, vy1, x1, y1 = l1[0][0], l1[1][0], l1[2][0], l1[3][0]
        vx2, vy2, x2, y2 = l2[0][0], l2[1][0], l2[2][0], l2[3][0]
        a1, b1, c1 = vy1, -vx1, vx1*y1 - vy1*x1
        a2, b2, c2 = vy2, -vx2, vx2*y2 - vy2*x2
        det = a1*b2 - a2*b1
        if abs(det) < 1e-6: return None
        return [(b1*c2 - b2*c1)/det, (a2*c1 - a1*c2)/det]

    best_corners = None
    min_area = float('inf')

    for comb in itertools.product(zones_lines[0], zones_lines[1], zones_lines[2], zones_lines[3]):
        l_top, l_right, l_bottom, l_left = comb
        
        tl = get_intersect(l_top, l_left)
        tr = get_intersect(l_top, l_right)
        br = get_intersect(l_bottom, l_right)
        bl = get_intersect(l_bottom, l_left)
        
        if None in [tl, tr, br, bl]:
            continue
            
        poly = np.array([tl, tr, br, bl], dtype=np.float32)
        
        if cv2.isContourConvex(poly):
            area = cv2.contourArea(poly)
            if area < min_area and area > (orig_bw * orig_bh * 0.1): 
                min_area = area
                best_corners = poly
                
    return best_corners

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    images_dir = os.path.join(base_dir, "data", "dataset_v1", "images")
    ann_dir = os.path.join(base_dir, "data", "dataset_v1", "annotations")
    model_path = os.path.join(base_dir, "result", "train_seg", "run6", "weights", "best.pt")
    out_dir = os.path.join(base_dir, "result", "pred")
    vis_dir = os.path.join(out_dir, "visualizations_seg")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    model = YOLO(model_path)
    
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
        if results and results[0].masks is not None:
            xy = results[0].masks.xy
            valid_masks = []
            
            for pts in xy:
                if len(pts) > 20:
                    poly = pts.astype(np.float32)
                    area = cv2.contourArea(poly)
                    if area > 1000:
                        valid_masks.append({"pts": pts, "area": area})
            
            if len(valid_masks) > 0:
                # 選擇面積最大的 Mask
                best_mask = max(valid_masks, key=lambda x: x["area"])
                contour_pts = best_mask["pts"].reshape(-1, 2)
                
                # 計算虛擬四個角
                virtual_corners = fit_4_lines_and_intersect(contour_pts)
                
                if virtual_corners is not None:
                    # 依據與標籤相同的邏輯向外擴張
                    pred_corners = expand_polygon(virtual_corners)
                    
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
                        
                        # Draw segmentation contour (Cyan)
                        pts_contour = contour_pts.astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [pts_contour], isClosed=True, color=(255, 255, 0), thickness=2)
                        
                        text = f"IoU: {iou:.3f} | MAE: {mae:.1f}px"
                        cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                        
                        cv2.imwrite(os.path.join(vis_dir, basename), img)
                    
        if not success:
            metrics["results"].append({
                "file": basename,
                "iou": 0.0,
                "mae": None,
                "error": "No valid segmentation mask or intersection failed"
            })
            
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(img_files)} images...")
            
    if metrics["valid_predictions"] > 0:
        metrics["avg_iou"] = float(total_iou / metrics["valid_predictions"])
        metrics["avg_corner_mae"] = float(total_mae / metrics["valid_predictions"])
        
    out_json = os.path.join(out_dir, "evaluation_seg_report.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
        
    print(f"Evaluation complete. Results saved to {out_json}")
    print(f"Total Images evaluated: {metrics['total_images']}")
    print(f"Valid Predictions: {metrics['valid_predictions']}")
    print(f"Average IoU: {metrics['avg_iou']:.4f}")
    print(f"Average Corner MAE (pixels): {metrics['avg_corner_mae']:.4f}")

if __name__ == '__main__':
    main()
