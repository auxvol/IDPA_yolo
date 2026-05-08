import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO

# 取得目前檔案路徑，並定位到專案根目錄
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from config.config import settings

# ==========================================
# 演算法微調開關
# ==========================================
DILATE_BBOX_RATIO = 0.05  # 方案一：Bounding Box 放大比例 (例如 0.05 = 向外擴張 5%)
USE_HOUGHLINES = True    # 方案二：是否使用 cv2.HoughLinesP 取代 cv2.fitLine
USE_APPROX_POLY = True    # 方案四：多邊形逼近 (幾何簡化)
APPROX_POLY_EPSILON = 0.007 # (降低強度!) 0.002 可確保「肩膀的鈍角」不會被當作雜訊削掉

# 標準 IDPA 靶紙 12 點歸一化座標 (用在拉平後的 2D 畫面上)
NORMALIZED_POINTS = np.array([
    [0.329828, 0.203805], # 0: 左頸部
    [0.329828, 0.000000], # 1: 左頭頂 (最高點)
    [0.679026, 0.000000], # 2: 右頭頂 (最高點)
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

def fit_4_lines_and_intersect(contour_pts, debug_frame=None):
    """
    步騞 1 & 2: 從混亂邊緣提取直線並計算完美的數學交點
    """
    orig_bx, orig_by, orig_bw, orig_bh = cv2.boundingRect(contour_pts)
    bx, by, bw, bh = orig_bx, orig_by, orig_bw, orig_bh
    
    # ==== 方案一：稍微「膨脹 (Dilate)」Bounding Box ====
    if DILATE_BBOX_RATIO > 0:
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        bw = bw * (1.0 + DILATE_BBOX_RATIO * 2)
        bh = bh * (1.0 + DILATE_BBOX_RATIO * 2)
        bx = cx - bw / 2.0
        by = cy - bh / 2.0
    
    if debug_frame is not None:
        # 畫出基礎的 BoundingBox (橙色)
        cv2.rectangle(debug_frame, (int(orig_bx), int(orig_by)), (int(orig_bx + orig_bw), int(orig_by + orig_bh)), (0, 165, 255), 2)
        cv2.putText(debug_frame, "BBox (Real)", (int(orig_bx), int(orig_by) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        if DILATE_BBOX_RATIO > 0:
            cv2.rectangle(debug_frame, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 120, 200), 1)

    # 取邊緣純粹的 60% 中段區域，徹底避開容易被木架遮擋或受轉角圓弧干擾的 4 個邊角區域
    left_pts   = [p for p in contour_pts if p[0] < bx + 0.25*bw and by + 0.2*bh < p[1] < by + 0.6*bh]
    right_pts  = [p for p in contour_pts if p[0] > bx + 0.75*bw and by + 0.2*bh < p[1] < by + 0.6*bh]
    top_pts    = [p for p in contour_pts if p[1] < by + 0.25*bh and bx + 0.4*bw < p[0] < bx + 0.6*bw]
    bottom_pts = [p for p in contour_pts if p[1] > by + 0.75*bh and bx + 0.4*bw < p[0] < bx + 0.6*bw]
    
    if debug_frame is not None:
        # 將過濾出來參與擬合的邊緣點群用不同顏色點畫出來
        for pt in left_pts:   cv2.circle(debug_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)   # 綠
        for pt in right_pts:  cv2.circle(debug_frame, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)   # 藍
        for pt in top_pts:    cv2.circle(debug_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1) # 黃
        for pt in bottom_pts: cv2.circle(debug_frame, (int(pt[0]), int(pt[1])), 3, (255, 255, 0), -1) # 青
        
    import itertools
    
    zones_lines = []
    # 順序: Top, Right, Bottom, Left
    for i, pts in enumerate([top_pts, right_pts, bottom_pts, left_pts]):
        zone_candidates = []
        if len(pts) < 5:
            # 觸發切線邊界判定條件：如果點太少，強制替換為 BoundingBox 切線
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
                    
                    min_len = max(img_w, img_h) * 0.3
                    h_lines = cv2.HoughLinesP(canvas, rho=1, theta=np.pi/180, threshold=10, minLineLength=min_len, maxLineGap=10)
                    
                    if h_lines is not None and len(h_lines) > 0:
                        # 收集所有潛在的長線段 (保留多種選擇，以便找出包裹面積最小的組合)
                        for hl in h_lines:
                            lx1, ly1, lx2, ly2 = hl[0][0] + min_x, hl[0][1] + min_y, hl[0][2] + min_x, hl[0][3] + min_y
                            vx = lx2 - lx1
                            vy = ly2 - ly1
                            norm = np.hypot(vx, vy)
                            if norm > 0:
                                zone_candidates.append(np.array([[vx/norm], [vy/norm], [lx1], [ly1]], dtype=np.float32))
                                
            # 若依然找不到可用線條，使用 Fallback 避開死區
            if len(zone_candidates) == 0:
                if i == 0:  l = np.array([[1.0], [0.0], [orig_bx + orig_bw/2.0], [orig_by]], dtype=np.float32)
                elif i == 1: l = np.array([[0.0], [1.0], [orig_bx + orig_bw], [orig_by + orig_bh/2.0]], dtype=np.float32)
                elif i == 2: l = np.array([[1.0], [0.0], [orig_bx + orig_bw/2.0], [orig_by + orig_bh]], dtype=np.float32)
                elif i == 3: l = np.array([[0.0], [1.0], [orig_bx], [orig_by + orig_bh/2.0]], dtype=np.float32)
                zone_candidates.append(l)

        # 為了效能，每邊最多保留最重要的前 5 條線進入排列組合
        if len(zone_candidates) > 5:
            zone_candidates = zone_candidates[:5]
        zones_lines.append(zone_candidates)
        
    def get_intersect(l1, l2):
        # 兩直線無限延伸取幾何數學交點
        vx1, vy1, x1, y1 = l1[0][0], l1[1][0], l1[2][0], l1[3][0]
        vx2, vy2, x2, y2 = l2[0][0], l2[1][0], l2[2][0], l2[3][0]
        # a*x + b*y + c = 0
        a1, b1, c1 = vy1, -vx1, vx1*y1 - vy1*x1
        a2, b2, c2 = vy2, -vx2, vx2*y2 - vy2*x2
        det = a1*b2 - a2*b1
        if abs(det) < 1e-6: return None
        return [(b1*c2 - b2*c1)/det, (a2*c1 - a1*c2)/det]

    best_corners = None
    min_area = float('inf')

    # 將四個邊界的潛在切線做組合，尋找包絡面積最小的組合！
    for comb in itertools.product(zones_lines[0], zones_lines[1], zones_lines[2], zones_lines[3]):
        l_top, l_right, l_bottom, l_left = comb
        
        tl = get_intersect(l_top, l_left)
        tr = get_intersect(l_top, l_right)
        br = get_intersect(l_bottom, l_right)
        bl = get_intersect(l_bottom, l_left)
        
        if None in [tl, tr, br, bl]:
            continue
            
        poly = np.array([tl, tr, br, bl], dtype=np.float32)
        
        # 基本防呆：必須是正常凸多邊形，不能是交叉打結的怪形狀
        if cv2.isContourConvex(poly):
            area = cv2.contourArea(poly)
            if area < min_area and area > (orig_bw * orig_bh * 0.1): # 面積至少要合理
                min_area = area
                best_corners = poly
                
    return best_corners

def extract_12_points_flat(flat_contour, target_w, target_h):
    """
    步騞 4: 在被完美拉平（透視消除）的 2D 對齊空間內，使用幾何對應抓 12 個精準位置。
    加入「匈牙利演算法 (Hungarian Algorithm)」，找出全局距離總和最小的唯一對應！
    完美解決局部重疊搶點導致「編號跑掉、順序錯亂」的拓樸失真問題。
    """
    from scipy.optimize import linear_sum_assignment
    
    num_targets = len(NORMALIZED_POINTS)
    num_contour = len(flat_contour)
    
    # 萬一輪廓點真的少於 12 個 (極端特例)，退回貪婪算法
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
        
    # 建構 12 x N 的成本矩陣 (幾何距離)
    cost_matrix = np.zeros((num_targets, num_contour), dtype=np.float32)
    
    for i, (nx, ny) in enumerate(NORMALIZED_POINTS):
        target_x = nx * (target_w - 1)
        target_y = ny * (target_h - 1)
        target_pt = np.array([target_x, target_y])
        
        # 計算該理想點到所有物理輪廓點的距離
        distances = np.linalg.norm(flat_contour - target_pt, axis=1)
        
        # 依據使用者指示：優先採取並忽視黏在 BBOX 邊框上的點
        # 如果候選點位剛好位在數學拉伸邊線上 (距離邊框 2 像素以內)，給予 50 像素的代價懲罰
        border_margin = 2
        is_on_border = (flat_contour[:, 0] <= border_margin) | \
                       (flat_contour[:, 0] >= target_w - 1 - border_margin) | \
                       (flat_contour[:, 1] <= border_margin) | \
                       (flat_contour[:, 1] >= target_h - 1 - border_margin)
                       
        distances[is_on_border] += 50.0  # +50 懲罰，促使演算法優先尋找稍往內縮的「真實幾何轉角」
        
        cost_matrix[i, :] = distances
        
    # 透過匈牙利演算法，計算全局最優的 12 對 12 獨立分配
    # 系統會自動協調衝突，絕對不會發生「提前被搶走導致下一個點錯亂跑掉」的情況
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 依序提取出來
    src_pts = np.zeros((num_targets, 2), dtype=np.float32)
    for i in range(num_targets):
        src_pts[i] = flat_contour[col_ind[i]]
        
    # 依據使用者指示：計算這 12 個特徵點連成的多邊形面積，面積越小代表被 Mask 沾黏外擴的誤差越小 (最緊密貼合)
    poly_area = cv2.contourArea(np.float32(src_pts))
        
    return src_pts, poly_area

def warp_tps(src_img, src_points, dst_points):
    """
    薄板樣條插值變形 (Thin Plate Spline Warping)：
    """
    # 建立 TPS 轉換器
    tps = cv2.createThinPlateSplineShapeTransformer()
    
    # 準備資料格式 (TPS 規定必須是 1 x N x 2 的形狀)
    pts_src = src_points.reshape(1, -1, 2).astype(np.float32)
    pts_dst = dst_points.reshape(1, -1, 2).astype(np.float32)
    
    # 計算 TPS 內部變形參數 (這裡包含了所有的彈性數學公式)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
    
    # 【重大修正】：OpenCV 的 TPS warpImage 有個底層特報坑，
    # 若要將 src 畫面推到 dst，必須傳入 (pts_dst, pts_src) 讓它計算「後向查表 (Backward mapping)」的變形向量！
    tps.estimateTransformation(pts_dst, pts_src, matches)
    
    # 執行全局橡膠皮拉伸！
    # 這裡的變形會涵蓋整張 src_img，完全不需要擔心邊界點
    out_img = tps.warpImage(src_img)
    
    return out_img

def run_pipeline():
    # 顯示比例改為 47.5 : 78 (等比例放大 6 倍為 285 : 468 畫素，確保畫面比例不走樣)
    target_w, target_h = 285, 468
    
    # 標準的畫布極值 4 點 (TL, TR, BR, BL)
    dst_corners = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype=np.float32)

    model_path = settings.get_path("result/train_seg/run2/weights/best.pt")
    video_path = settings.get_path("data/靶場/G5_S03.mp4")
    model = YOLO(model_path) 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 錯誤: 無法讀取影片檔案 {video_path}")
        return

    window_name = "YOLO Seg (Hardcore Line Fitting + Flat 12-Pts Warp)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"OpenCV 演算法：(無極限延長線擬合 + 透視攤平雙重變形)")
    
    # 建立第一階段 5 秒統計機制與第二階段全局最佳變數
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps): fps = 30
    calibration_frames_target = int(fps * 3)
    
    calib_history_pts = []
    calib_history_areas = []
    calib_history_centroids = []
    is_calibrated = False

    global_optimal_contour_pts = None
    global_min_mask_area = float('inf')
    global_optimal_centroid = None

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        global_has_new_candidate = False # 紀錄這幀是否提出了新最佳解，以執行透明通道回朔測量

        results = model.predict(frame, conf=0.3, imgsz=640, device=0, verbose=False)
        display_frame = frame.copy()
        warped_frame = np.zeros((target_h, target_w, 3), dtype="uint8")

        if results and results[0].masks is not None:
            xy = results[0].masks.xy
            
            if not is_calibrated:
                # 第一階段：前 5 秒統計學基準建模
                # 在每一幀中尋找「面積最大」的遮罩 (確保抓的是主靶紙，避免一開始就抓歪成小點)
                best_main_pts = None
                max_area = 0
                for pts in xy:
                    if len(pts) > 20:
                        poly = pts.astype(np.float32)
                        area = cv2.contourArea(poly)
                        if area > max_area:
                            max_area = area
                            best_main_pts = pts
                            
                if best_main_pts is not None:
                    M_moments = cv2.moments(best_main_pts.astype(np.float32))
                    if M_moments["m00"] != 0:
                        cx = int(M_moments["m10"] / M_moments["m00"])
                        cy = int(M_moments["m01"] / M_moments["m00"])
                        calib_history_pts.append(best_main_pts.copy())
                        calib_history_areas.append(max_area)
                        calib_history_centroids.append((cx, cy))
                        
                cv2.putText(display_frame, f"Phase 1: Profiling Root Object {len(calib_history_areas)}/{calibration_frames_target}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # 若收集滿 5 秒
                if len(calib_history_areas) >= calibration_frames_target:
                    # 尋找前五秒內「出現次數最多 (面積最常態/中位數)」的點位為基準
                    median_area = np.median(calib_history_areas)
                    diffs = np.abs(np.array(calib_history_areas) - median_area)
                    best_idx = np.argmin(diffs) # 最接近中位數的那一幀
                    
                    global_min_mask_area = calib_history_areas[best_idx]
                    global_optimal_contour_pts = calib_history_pts[best_idx].copy()
                    global_optimal_centroid = calib_history_centroids[best_idx]
                    is_calibrated = True
                    print(f"✅ 基準鎖定完成！主體中位面積為: {int(global_min_mask_area)}")
                    
                # 收集期間維持運作，將 orig_contour_pts 設為當前幀的最大目標
                if best_main_pts is not None:
                    orig_contour_pts = best_main_pts
                else:
                    continue
                    
            else:
                # 第二階段：全局最優面積鎖定追蹤 (帶有 2% 防呆域值)
                cv2.putText(display_frame, f"Phase 2: Global Best-Fit Tracking (Best: {int(global_min_mask_area)})", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # 掃描當前所有遮罩，尋找可以「突破歷史最低紀錄」的遮罩
                for pts in xy:
                    if len(pts) > 20: 
                        poly = pts.astype(np.float32)
                        area = cv2.contourArea(poly)
                        
                        if area > 1000:
                            # 1. 重心不能偏移超過歷史基準 (防跳到其他物體)
                            M_moments = cv2.moments(poly)
                            if M_moments["m00"] != 0:
                                cx = int(M_moments["m10"] / M_moments["m00"])
                                cy = int(M_moments["m01"] / M_moments["m00"])
                                dist = np.hypot(cx - global_optimal_centroid[0], cy - global_optimal_centroid[1])
                                if dist > 300:
                                    continue # 位置偏離太遠，忽視
                            else:
                                continue
                                
                            # 2. 面積必須更小，且「不得過於偏離上次最優面積 (2%域值)」
                            if area < global_min_mask_area:
                                # 如果面積縮水幅度 超過歷史基準的 2%，極大機率是影像破裂或是強烈遮擋的異常值，予以忽視！
                                if (global_min_mask_area - area) > (global_min_mask_area * 0.02):
                                    continue
                                    
                                # 加入備份機制：為了下方的 Alpha 透明度黑洞檢測準備回朔點
                                backup_min_mask_area = global_min_mask_area
                                backup_contour_pts = global_optimal_contour_pts.copy() if global_optimal_contour_pts is not None else None
                                backup_centroid = global_optimal_centroid
                                global_has_new_candidate = True
                                
                                # 若通過 2% 審查，則正式破紀錄，取代為新的全局極值！
                                global_min_mask_area = area
                                global_optimal_contour_pts = pts.copy()
                                global_optimal_centroid = (cx, cy)
                                print(f"🌟 新的最優解出現！歷史最小面積更新為：{int(global_min_mask_area)}")
                                
                # 永遠強制拿全歷史中最完美的那個遮罩出來做變形！
                if global_optimal_contour_pts is not None:
                    orig_contour_pts = global_optimal_contour_pts
                
                # ==== 方案四：多邊形幾何逼近簡化 (cv2.approxPolyDP) ====
                if USE_APPROX_POLY:
                    epsilon = APPROX_POLY_EPSILON * cv2.arcLength(orig_contour_pts, True)
                    contour_pts = cv2.approxPolyDP(orig_contour_pts, epsilon, True).reshape(-1, 2)
                else:
                    contour_pts = orig_contour_pts
                
                # 畫出經過 (或未經過) 簡化的 Segmentation 輪廓 (藍色粗線)
                poly = contour_pts.astype(np.int32)
                cv2.polylines(display_frame, [poly], True, (255, 0, 0), 3)

                # ========================================================
                # 步騞 1 & 2：邊緣直線提取 與 虛擬交點計算
                # ========================================================
                virtual_corners = fit_4_lines_and_intersect(contour_pts, display_frame)
                
                if virtual_corners is not None:
                    # 在畫面上畫出這推算出來的【紫紅色虛擬交點】與包絡線
                    virtual_poly = virtual_corners.astype(np.int32)
                    cv2.polylines(display_frame, [virtual_poly], True, (255, 0, 255), 2)
                    for pt in virtual_corners:
                        cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 8, (255, 0, 255), -1)
                    
                    # ========================================================
                    # 步騞 3：破解死迴圈，算出 3D 傾斜矩陣 M
                    # ========================================================
                    M, _ = cv2.findHomography(virtual_corners, dst_corners, 0)
                    
                    if M is not None:
                        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask, [poly], 255)
                        
                        # 實作「去背」：將不在遮罩內的原始背景像素全部抹除為純黑黑底
                        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                        
                        # 改為建立包含 Alpha 通道的 BGRA 影像以偵測「切出界」產生的透明空景
                        b, g, r = cv2.split(masked_frame)
                        bgra_frame = cv2.merge((b, g, r, mask))
                        
                        # 這是影像裁切與視覺化的部分 (拉出被攤平的靶紙照片)
                        warped_frame_bgra = cv2.warpPerspective(bgra_frame, M, (target_w, target_h))
                        warped_frame = warped_frame_bgra[:, :, :3].copy()
                        
                        # ========================================================
                        # 步騞 4：套用數學轉換 (Warping) 到所有輪廓節點
                        # ========================================================
                        # 核心觀念：把這幾百個邊緣點通通變形到拉平的二維空間中
                        flat_contour = cv2.perspectiveTransform(contour_pts.reshape(-1, 1, 2), M).reshape(-1, 2)
                        
                        # 由於現在靶紙是 100% 水平垂直正面的，用這套去找 12 點絕對精確
                        flat_12_points, frame_cost = extract_12_points_flat(flat_contour, target_w, target_h)
                        
                        try:
                            M_inv = np.linalg.inv(M)
                            org_12_points = cv2.perspectiveTransform(flat_12_points.reshape(-1, 1, 2), M_inv).reshape(-1, 2)
                        except np.linalg.LinAlgError:
                            continue
                        
                        # 備份一個「完全乾淨」的純粹畫面
                        clean_warped_frame_bgra = warped_frame_bgra.copy()
                        
                        # 視覺化步騞 4：畫出被透視去除後，平躺在 2D 空間的「完整數學輪廓 (紫色)」
                        flat_poly = flat_contour.astype(np.int32)
                        cv2.polylines(warped_frame, [flat_poly], True, (255, 0, 255), 2)
                        
                        ideal_pts_arr = np.array([[nx * (target_w - 1), ny * (target_h - 1)] for nx, ny in NORMALIZED_POINTS], dtype=np.float32)

                        # 在攤平的圖面上用【黃色】畫靶紙這 12 個特徵點
                        for i, pt in enumerate(flat_12_points):
                            # 理想的設計目標落點 (青色十字)
                            ideal_pt = ideal_pts_arr[i]
                            cv2.drawMarker(warped_frame, (int(ideal_pt[0]), int(ideal_pt[1])), (255, 255, 0), cv2.MARKER_CROSS, 10, 1)
                            
                            # 視覺化匈牙利演算法匹配成果：用白線連接目標點跟實際抓到的輪廓點
                            cv2.line(warped_frame, (int(ideal_pt[0]), int(ideal_pt[1])), (int(pt[0]), int(pt[1])), (255, 255, 255), 1, cv2.LINE_AA)
                            
                            # 實際物理特徵點 (紅點+黃字)
                            cv2.circle(warped_frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
                            cv2.putText(warped_frame, str(i), (int(pt[0])+5, int(pt[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        # ========================================================
                        # 步騞 6：二次強制拉伸 (Thin Plate Spline 橡膠皮平滑變形)
                        # ========================================================
                        perfect_warped_frame_bgra = warp_tps(clean_warped_frame_bgra, flat_12_points, ideal_pts_arr)
                        perfect_warped_frame = perfect_warped_frame_bgra[:, :, :3].copy()
                        
                        # ========================================================
                        # 步騞 6.5：透明通道邊界空洞檢查 (黑洞防禦回朔系統)
                        # ========================================================
                        if global_has_new_candidate:
                            alpha_channel = perfect_warped_frame_bgra[:, :, 3]
                            ideal_poly = ideal_pts_arr.astype(np.int32)
                            target_mask = np.zeros((target_h, target_w), dtype=np.uint8)
                            cv2.fillPoly(target_mask, [ideal_poly], 255)
                            
                            # 找出在理想標準靶紙內部，Alpha 值卻小於 100 (透明) 的像素
                            transparent_void = cv2.bitwise_and(np.uint8(alpha_channel < 100) * 255, np.uint8(alpha_channel < 100) * 255, mask=target_mask)
                            void_count = np.count_nonzero(transparent_void)
                            
                            if void_count > 500:
                                print(f"⚠️ 警告：新遮罩讓主體空缺了 {void_count} 個透明像素！觸發回朔防護！")
                                global_min_mask_area = backup_min_mask_area
                                global_optimal_contour_pts = backup_contour_pts
                                global_optimal_centroid = backup_centroid
                                # 既然這個幀的幾何形狀損壞了，我們直接跳過後續渲染與顯示，下一幀它會自己用回朔的安全數值重新計算！
                                continue
                            else:
                                print(f"🌟 新的最優解通過透明通道測試！面積無損完整！正式採用！ (邊界損失微塵: {void_count})")
                                
                        cv2.putText(perfect_warped_frame, "12-Pts TPS Rubber Stretch", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

                        # ========================================================
                        # 步騞 7：投射回攝影機原始視角畫面！
                        # ========================================================
                        for i, pt in enumerate(org_12_points):
                            # 畫上我們最終解逆計算出來的【完美綠點】
                            cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
                            cv2.putText(display_frame, str(i), (int(pt[0])+5, int(pt[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
        # 合併顯示
        h, w = display_frame.shape[:2]
        ratio = target_h / h
        resized_display = cv2.resize(display_frame, (int(w * ratio), target_h))
        
        # 將三者並排顯示：原始畫面(標註綠點)、BBOX一階拉平(標註找出的紅點及理想綠框)、12點二階強制拉伸(對齊綠框)
        if 'perfect_warped_frame' not in locals():
            perfect_warped_frame = np.zeros((target_h, target_w, 3), dtype="uint8")
            
        combined_view = np.hstack((resized_display, warped_frame, perfect_warped_frame))
        
        cv2.imshow(window_name, combined_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 支援 q 或 ESC 鍵
            break
            
        # 允許使用者直接點擊視窗右上角的 'X' 關閉
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 處理結束。")

if __name__ == "__main__":
    run_pipeline()
