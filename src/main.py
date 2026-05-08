import cv2
import numpy as np
import os
import sys
import math
from ultralytics import YOLO

# 取得目前檔案路徑，並定位到專案根目錄
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from config.config import settings

# ==========================================
# 演算法微調開關
# ==========================================
DILATE_BBOX_RATIO = 0.05  # 方案一：Bounding Box 放大比例
USE_HOUGHLINES = True    # 方案二：是否使用 cv2.HoughLinesP
USE_APPROX_POLY = True    # 方案四：多邊形逼近
APPROX_POLY_EPSILON = 0.007 # 0.002 可確保「肩膀的鈍角」不會被當作雜訊削掉

# 標準 IDPA 靶紙 12 點歸一化座標 
NORMALIZED_POINTS = np.array([
    [0.329828, 0.203805], # 0: 左頸部
    [0.329828, 0.000000], # 1: 左頭頂
    [0.679026, 0.000000], # 2: 右頭頂
    [0.679026, 0.203805], # 3: 右頸部
    [0.892086, 0.208972], # 4: 右肩內側
    [1.000000, 0.278788], # 5: 右肩外緣 
    [0.999447, 0.810260], # 6: 右腰部
    [0.827892, 0.996325], # 7: 右底角
    [0.173215, 1.000000], # 8: 左底角 
    [0.000000, 0.810260], # 9: 左腰部 
    [0.008301, 0.278788], # 10: 左肩外緣
    [0.116215, 0.206634]  # 11: 左肩內側
], dtype=np.float32)

# 為了區分多個標靶，設定預設顏色序列
TRACKER_COLORS = [
    (0, 255, 255),   # 黃
    (0, 255, 0),     # 綠
    (255, 0, 255),   # 紫
    (0, 165, 255),   # 橙
    (255, 200, 0),   # 淺藍
    (255, 0, 0)      # 藍
]

class TargetTracker:
    def __init__(self, target_id):
        self.target_id = target_id
        self.color = TRACKER_COLORS[target_id % len(TRACKER_COLORS)]
        
        # 第一階段：校準期 (Phase 1)
        self.calib_history_pts = []
        self.calib_history_areas = []
        self.calib_history_centroids = []
        self.is_calibrated = False
        
        # 第二階段：主動追蹤期 (Phase 2)
        self.global_min_mask_area = float('inf')
        self.global_optimal_contour_pts = None
        self.global_optimal_centroid = None
        self.has_new_candidate = False
        self.backup_min_mask_area = float('inf')
        self.backup_contour_pts = None
        self.backup_centroid = None
        
        self.warped_frame = None
        self.perfect_warped_frame = None
        self.last_valid_org_12_points = None

def fit_4_lines_and_intersect(contour_pts, debug_frame=None, debug_color=(0, 165, 255)):
    orig_bx, orig_by, orig_bw, orig_bh = cv2.boundingRect(contour_pts)
    bx, by, bw, bh = orig_bx, orig_by, orig_bw, orig_bh
    
    if DILATE_BBOX_RATIO > 0:
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        bw = bw * (1.0 + DILATE_BBOX_RATIO * 2)
        bh = bh * (1.0 + DILATE_BBOX_RATIO * 2)
        bx = cx - bw / 2.0
        by = cy - bh / 2.0
    
    if debug_frame is not None:
        cv2.rectangle(debug_frame, (int(orig_bx), int(orig_by)), (int(orig_bx + orig_bw), int(orig_by + orig_bh)), debug_color, 2)
        if DILATE_BBOX_RATIO > 0:
            cv2.rectangle(debug_frame, (int(bx), int(by)), (int(bx + bw), int(by + bh)), debug_color, 1)

    left_pts   = [p for p in contour_pts if p[0] < bx + 0.25*bw and by + 0.2*bh < p[1] < by + 0.6*bh]
    right_pts  = [p for p in contour_pts if p[0] > bx + 0.75*bw and by + 0.2*bh < p[1] < by + 0.6*bh]
    top_pts    = [p for p in contour_pts if p[1] < by + 0.25*bh and bx + 0.3*bw < p[0] < bx + 0.7*bw]
    bottom_pts = [p for p in contour_pts if p[1] > by + 0.75*bh and bx + 0.4*bw < p[0] < bx + 0.6*bw]
    
    import itertools
    
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

def extract_12_points_flat(flat_contour, target_w, target_h):
    from scipy.optimize import linear_sum_assignment
    
    num_targets = len(NORMALIZED_POINTS)
    num_contour = len(flat_contour)
    
    if num_contour < num_targets:
        src_pts = []
        total_cost = 0.0
        for nx, ny in NORMALIZED_POINTS:
            target_x = nx * (target_w - 1)
            target_y = ny * (target_h - 1)
            target_pt = np.array([target_x, target_y])
            distances = np.linalg.norm(flat_contour - target_pt, axis=1)
            best_idx = np.argmin(distances)
            src_pts.append(flat_contour[best_idx])
            total_cost += distances[best_idx]
        return np.array(src_pts, dtype=np.float32), total_cost + 99999.0
        
    cost_matrix = np.zeros((num_targets, num_contour), dtype=np.float32)
    
    for i, (nx, ny) in enumerate(NORMALIZED_POINTS):
        target_x = nx * (target_w - 1)
        target_y = ny * (target_h - 1)
        target_pt = np.array([target_x, target_y])
        
        distances = np.linalg.norm(flat_contour - target_pt, axis=1)
        
        border_margin = 2
        is_on_border = (flat_contour[:, 0] <= border_margin) | \
                       (flat_contour[:, 0] >= target_w - 1 - border_margin) | \
                       (flat_contour[:, 1] <= border_margin) | \
                       (flat_contour[:, 1] >= target_h - 1 - border_margin)
                       
        distances[is_on_border] += 50.0  
        cost_matrix[i, :] = distances
        
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 強制拓樸順序：無論演算法怎麼配對，強制這 12 個點必須沿著實體輪廓依序排列 (測試所有順逆時針與起點平移組合)
    sorted_col = np.sort(col_ind)
    best_seq_cost = float('inf')
    best_seq_col = None
    
    for scan_dir in [sorted_col, sorted_col[::-1]]:
        for shift in range(num_targets):
            shifted_col = np.roll(scan_dir, shift)
            current_cost = 0.0
            for i in range(num_targets):
                current_cost += cost_matrix[i, shifted_col[i]]
            if current_cost < best_seq_cost:
                best_seq_cost = current_cost
                best_seq_col = shifted_col
                
    col_ind = best_seq_col
    src_pts = np.zeros((num_targets, 2), dtype=np.float32)
    total_cost = 0.0
    for i in range(num_targets):
        src_pts[i] = flat_contour[col_ind[i]]
        total_cost += cost_matrix[i, col_ind[i]]
        
    # 確保最終選出來的 12 點順序不會交錯打結：如果點序發生交叉，多邊形面積會急遽縮小
    poly_area = cv2.contourArea(np.float32(src_pts))
    if poly_area < (target_w * target_h * 0.5):
        total_cost += 99999.0  # 順序錯亂或過度擠壓變形，加上毀滅性懲罰
        
    return src_pts, total_cost

def warp_tps(src_img, src_points, dst_points):
    tps = cv2.createThinPlateSplineShapeTransformer()
    pts_src = src_points.reshape(1, -1, 2).astype(np.float32)
    pts_dst = dst_points.reshape(1, -1, 2).astype(np.float32)
    
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
    tps.estimateTransformation(pts_dst, pts_src, matches)
    out_img = tps.warpImage(src_img)
    return out_img

def run_pipeline():
    target_w, target_h = 285, 468
    
    dst_corners = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype=np.float32)

    model_path = settings.get_path("result/train_seg/run6/weights/best.pt")
    video_path = settings.get_path("data\靶場\G7_S01.mp4")
    model = YOLO(model_path) 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 錯誤: 無法讀取影片檔案 {video_path}")
        return

    window_name = "YOLO Seg Multi-Target Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"啟動多目標獨立追蹤管線...")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps): fps = 30
    calibration_frames_target = int(fps * 3) # 3秒基準校準
    
    frame_count = 0
    active_trackers = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        display_frame = frame.copy()

        # 僅清除狀態，不清除歷史變形畫面以防視窗閃爍
        for t in active_trackers:
            t.has_new_candidate = False

        results = model.predict(frame, conf=0.3, imgsz=640, device=0, verbose=False)

        if results and results[0].masks is not None:
            xy = results[0].masks.xy
            
            valid_masks = []
            for pts in xy:
                if len(pts) > 20:
                    poly = pts.astype(np.float32)
                    area = cv2.contourArea(poly)
                    if area > 1000:
                        M_moments = cv2.moments(poly)
                        if M_moments["m00"] != 0:
                            cx = int(M_moments["m10"] / M_moments["m00"])
                            cy = int(M_moments["m01"] / M_moments["m00"])
                            valid_masks.append({
                                "pts": pts, "area": area, "cx": cx, "cy": cy, "poly": poly
                            })

            # ==========================================================
            # 第一階段: 校準收集期 (允許多緩衝 10 禎確保部分對象能收集足夠資訊)
            # ==========================================================
            if frame_count <= calibration_frames_target + 10:
                cv2.putText(display_frame, f"Phase 1: Multi Calibration ({frame_count}/{calibration_frames_target})", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                for m in valid_masks:
                    best_t = None
                    min_dist = float('inf')
                    for t in active_trackers:
                        if len(t.calib_history_centroids) > 0:
                            last_c = t.calib_history_centroids[-1]
                            dist = np.hypot(m["cx"] - last_c[0], m["cy"] - last_c[1])
                            if dist < 150 and dist < min_dist:
                                min_dist = dist
                                best_t = t
                    if best_t:
                        if not best_t.is_calibrated:
                           best_t.calib_history_pts.append(m["pts"])
                           best_t.calib_history_areas.append(m["area"])
                           best_t.calib_history_centroids.append((m["cx"], m["cy"]))
                    else:
                        new_t = TargetTracker(len(active_trackers))
                        new_t.calib_history_pts.append(m["pts"])
                        new_t.calib_history_areas.append(m["area"])
                        new_t.calib_history_centroids.append((m["cx"], m["cy"]))
                        active_trackers.append(new_t)
                
                # 檢查是否有標靶滿足校準畢業門檻 (累積超過一半的影格)
                for t in active_trackers:
                    if not t.is_calibrated and len(t.calib_history_areas) > (calibration_frames_target * 0.4):
                        if frame_count > calibration_frames_target:
                            median_area = np.median(t.calib_history_areas)
                            diffs = np.abs(np.array(t.calib_history_areas) - median_area)
                            best_idx = np.argmin(diffs)
                            
                            t.global_min_mask_area = t.calib_history_areas[best_idx]
                            t.global_optimal_contour_pts = t.calib_history_pts[best_idx].copy()
                            t.global_optimal_centroid = t.calib_history_centroids[best_idx]
                            t.is_calibrated = True
                            print(f"✅ Target ID:{t.target_id} 鎖定完成！基準基準面積為: {int(t.global_min_mask_area)}")

            # ==========================================================
            # 第二階段: 獨立主動追蹤期
            # ==========================================================
            else:
                cv2.putText(display_frame, f"Phase 2: Active Tracking ({len([t for t in active_trackers if t.is_calibrated])} Targets)", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                for m in valid_masks:
                    best_t = None
                    min_dist = float('inf')
                    for t in active_trackers:
                        if t.is_calibrated:
                            dist = np.hypot(m["cx"] - t.global_optimal_centroid[0], m["cy"] - t.global_optimal_centroid[1])
                            if dist < 300 and dist < min_dist:
                                min_dist = dist
                                best_t = t
                    
                    if best_t:
                        # 2.5% 防呆門檻更新最小面積
                        if m["area"] < best_t.global_min_mask_area:
                            if (best_t.global_min_mask_area - m["area"]) <= (best_t.global_min_mask_area * 0.05):
                                best_t.backup_min_mask_area = best_t.global_min_mask_area
                                best_t.backup_contour_pts = best_t.global_optimal_contour_pts.copy() if best_t.global_optimal_contour_pts is not None else None
                                best_t.backup_centroid = best_t.global_optimal_centroid
                                best_t.has_new_candidate = True
                                
                                best_t.global_min_mask_area = m["area"]
                                best_t.global_optimal_contour_pts = m["pts"].copy()
                                best_t.global_optimal_centroid = (m["cx"], m["cy"])
        
        # 對每個校準完成的標靶獨立進行仿射提取
        for t in active_trackers:
            if not t.is_calibrated or t.global_optimal_contour_pts is None:
                continue

            orig_contour_pts = t.global_optimal_contour_pts
            
            if USE_APPROX_POLY:
                epsilon = APPROX_POLY_EPSILON * cv2.arcLength(orig_contour_pts, True)
                contour_pts = cv2.approxPolyDP(orig_contour_pts, epsilon, True).reshape(-1, 2)
            else:
                contour_pts = orig_contour_pts
            
            poly = contour_pts.astype(np.int32)
            cv2.polylines(display_frame, [poly], True, t.color, 3)
            # 在畫面標上 Tracker ID
            cv2.putText(display_frame, f"ID:{t.target_id}", (int(t.global_optimal_centroid[0]-25), int(t.global_optimal_centroid[1]-40)), cv2.FONT_HERSHEY_SIMPLEX, 1, t.color, 3)

            virtual_corners = fit_4_lines_and_intersect(contour_pts, display_frame, t.color)
            
            if virtual_corners is not None:
                virtual_poly = virtual_corners.astype(np.int32)
                cv2.polylines(display_frame, [virtual_poly], True, t.color, 2)
                for pt in virtual_corners:
                    cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 8, t.color, -1)
                
                M, _ = cv2.findHomography(virtual_corners, dst_corners, 0)
                
                if M is not None:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [poly], 255)
                    
                    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                    b, g, r = cv2.split(masked_frame)
                    bgra_frame = cv2.merge((b, g, r, mask))
                    
                    warped_frame_bgra = cv2.warpPerspective(bgra_frame, M, (target_w, target_h))
                    temp_warped = warped_frame_bgra[:, :, :3].copy()
                    
                    flat_contour = cv2.perspectiveTransform(contour_pts.reshape(-1, 1, 2), M).reshape(-1, 2)
                    flat_12_points, frame_cost = extract_12_points_flat(flat_contour, target_w, target_h)
                    
                    try:
                        M_inv = np.linalg.inv(M)
                        org_12_points = cv2.perspectiveTransform(flat_12_points.reshape(-1, 1, 2), M_inv).reshape(-1, 2)
                    except np.linalg.LinAlgError:
                        continue
                    
                    clean_warped_frame_bgra = warped_frame_bgra.copy()
                    flat_poly = flat_contour.astype(np.int32)
                    cv2.polylines(temp_warped, [flat_poly], True, (255, 0, 255), 2)
                    
                    ideal_pts_arr = np.array([[nx * (target_w - 1), ny * (target_h - 1)] for nx, ny in NORMALIZED_POINTS], dtype=np.float32)

                    for i, pt in enumerate(flat_12_points):
                        ideal_pt = ideal_pts_arr[i]
                        cv2.drawMarker(temp_warped, (int(ideal_pt[0]), int(ideal_pt[1])), (255, 255, 0), cv2.MARKER_CROSS, 10, 1)
                        cv2.line(temp_warped, (int(ideal_pt[0]), int(ideal_pt[1])), (int(pt[0]), int(pt[1])), (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.circle(temp_warped, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
                        cv2.putText(temp_warped, str(i), (int(pt[0])+5, int(pt[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    perfect_warped_frame_bgra = warp_tps(clean_warped_frame_bgra, flat_12_points, ideal_pts_arr)
                    temp_perfect = perfect_warped_frame_bgra[:, :, :3].copy()
                    
                    is_valid_candidate = True
                    if t.has_new_candidate:
                        alpha_channel = perfect_warped_frame_bgra[:, :, 3]
                        ideal_poly = ideal_pts_arr.astype(np.int32)
                        target_mask = np.zeros((target_h, target_w), dtype=np.uint8)
                        cv2.fillPoly(target_mask, [ideal_poly], 255)
                        # 收縮範圍 2 像素，避免 TPS 邊緣抗鋸齒造成的外框單像素透明度誤判
                        target_mask = cv2.erode(target_mask, np.ones((3,3), np.uint8), iterations=1)
                        
                        transparent_void = cv2.bitwise_and(np.uint8(alpha_channel < 100) * 255, np.uint8(alpha_channel < 100) * 255, mask=target_mask)
                        void_count = np.count_nonzero(transparent_void)
                        
                        if void_count > 100:
                            # 由於 YOLO 邊緣震盪會產生看似更緊但殘缺的遮罩，這裡靜默回朔不更新緩衝，避免無限洗頻
                            t.global_min_mask_area = t.backup_min_mask_area
                            t.global_optimal_contour_pts = t.backup_contour_pts
                            t.global_optimal_centroid = t.backup_centroid
                            is_valid_candidate = False
                        elif frame_cost > 3000:
                            # 三維空間形變過大（距離標準標靶骨架超過安全閾值），表示抓到了極度扭曲的假遮罩
                            t.global_min_mask_area = t.backup_min_mask_area
                            t.global_optimal_contour_pts = t.backup_contour_pts
                            t.global_optimal_centroid = t.backup_centroid
                            is_valid_candidate = False
                        else:
                            print(f"🌟 Target ID:{t.target_id} 破紀錄！面積縮小並通過邊界測試：{int(t.global_min_mask_area)} (Cost:{int(frame_cost)})")
                            
                    if is_valid_candidate:
                        t.warped_frame = temp_warped
                        t.perfect_warped_frame = temp_perfect
                        t.last_valid_org_12_points = org_12_points
                        cv2.putText(t.perfect_warped_frame, f"ID:{t.target_id} TPS Ready", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, t.color, 2)
                        
                    pts_to_draw = t.last_valid_org_12_points if t.last_valid_org_12_points is not None else org_12_points
                    for i, pt in enumerate(pts_to_draw):
                        cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
                        cv2.putText(display_frame, str(i), (int(pt[0])+5, int(pt[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
        # ==========================================
        # 動態表格(Grid)儀表板合成顯示
        # ==========================================
        calibrated_trackers = [t for t in active_trackers if t.is_calibrated]
        num_targets = len(calibrated_trackers)
        
        # 單一獨立視圖 = 570 x 468 (兩個 Warp 畫面組合)
        block_w = target_w * 2  
        block_h = target_h
        
        if num_targets > 0:
            cols = math.ceil(math.sqrt(num_targets))
            if num_targets == 2: cols = 1   # 2個目標，設定為 1 列 2 排 (垂直疊加)，或 2列1排
                
            # 依使用者需求：1個=不變，2個=1*2，3個=2*2
            # 2個目標我們採用: 1行2列 (水平) 或 2行1列(垂直)? 這裡實作為水平 cols=2
            if num_targets == 2: cols = 2
            
            rows = math.ceil(num_targets / cols)
            
            grid_w = cols * block_w
            grid_h = rows * block_h
            
            grid_view = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            
            for i, t in enumerate(calibrated_trackers):
                r = i // cols
                c = i % cols
                x_offset = c * block_w
                y_offset = r * block_h
                if t.warped_frame is not None and t.perfect_warped_frame is not None:
                    grid_view[y_offset : y_offset+block_h, x_offset : x_offset+target_w] = t.warped_frame
                    grid_view[y_offset : y_offset+block_h, x_offset+target_w : x_offset+target_w*2] = t.perfect_warped_frame
                
            # 將主畫面等比例縮放以對齊 Grid 的高度
            h, w = display_frame.shape[:2]
            ratio = grid_h / h
            resized_display = cv2.resize(display_frame, (int(w * ratio), grid_h))
            
            combined_view = np.hstack((resized_display, grid_view))
        else:
            # 只有1個人或還沒校準時，就維持空框
            h, w = display_frame.shape[:2]
            ratio = target_h / h
            resized_display = cv2.resize(display_frame, (int(w * ratio), target_h))
            empty_grid = np.zeros((target_h, target_w * 2, 3), dtype=np.uint8)
            combined_view = np.hstack((resized_display, empty_grid))
            
        cv2.imshow(window_name, combined_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: 
            break
            
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 處理結束。")

if __name__ == "__main__":
    run_pipeline()
