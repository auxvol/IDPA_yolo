"""
YOLO Pose + 彈孔偵測 + IDPA 計分 整合管線
===========================================
基於 main_pose.py 的多目標追蹤管線，在攤平 (TPS) 後的靶紙影像上
使用幀差法偵測新彈孔，並以 IDPA 加時制計分。

彈孔偵測邏輯參考自:
  C:/local_python/GIT/Multimodal-AI-Agent-with-Persistent-Target-Map-/src/shot_detector.py
"""
import cv2
import numpy as np
import os
import sys
import math
import json
from datetime import datetime
from ultralytics import YOLO

# 取得目前檔案路徑，並定位到專案根目錄
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from config.config import settings
from idpa_scoring import IDPAScorer

# ==========================================
# 標準 IDPA 靶紙 17 點歸一化座標
# ==========================================
NORMALIZED_POINTS_17 = np.array([
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
    [0.116215, 0.206634], # 11: 左肩內側
    [0.000000, 0.544524], # 12: 左中
    [0.504427, 0.000000], # 13: 頭頂中心
    [0.504427, 1.000000], # 14: 底部中心
    [0.504427, 0.544524], # 15: 中心
    [1.000000, 0.544524], # 16: 右中
], dtype=np.float32)

OUTLINE_INDICES = list(range(12))

TRACKER_COLORS = [
    (0, 255, 255),   # 黃
    (0, 255, 0),     # 綠
    (255, 0, 255),   # 紫
    (0, 165, 255),   # 橙
    (255, 200, 0),   # 淺藍
    (255, 0, 0)      # 藍
]

# 計分區對應的顏色 (BGR)
ZONE_COLORS = {
    0: (0, 255, 0),       # -0 區: 綠色
    1: (0, 255, 255),     # -1 區: 黃色
    3: (0, 165, 255),     # -3 區: 橙色
    "Miss": (0, 0, 255),  # Miss: 紅色
}

ZONE_PENALTY = {0: 0, 1: 1, 3: 3, "Miss": 5}


# ==========================================
# 彈孔偵測器 (在攤平靶紙上做幀差法)
# 參考 Multimodal-AI-Agent 的 ShotDetector
# ==========================================
class WarpedShotDetector:
    """
    在攤平 (TPS-warped) 後的靶紙影像上做彈孔偵測。
    核心邏輯：用上一幀（已確認背景）與當前幀做灰階差分，
    找出新出現的暗點（彈孔），以時序持續性確認。
    """
    def __init__(
        self,
        min_contour_area: int = 1,         # 提高最小面積，濾除極小雜點
        max_contour_area: int = 3200,
        diff_thresh: int = 15,             # 提高差分門檻，過濾微小光影變化
        max_hole_brightness: int = 400,    # 降低亮度上限，確保是明顯暗點
        distance_thresh: float = 2.0,      # 放寬追蹤距離容許度，抵抗靶紙震動
        known_hole_radius: float = 2.0,   # 提高去重半徑
        persistence_frames: int = 10,       # 提高確認幀數，必須穩定出現
        cooldown_frames: int = 3,
    ):
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.diff_thresh = diff_thresh
        self.max_hole_brightness = max_hole_brightness
        self.distance_thresh = distance_thresh
        self.known_hole_radius = known_hole_radius
        self.persistence_frames = persistence_frames
        self.cooldown_frames = cooldown_frames

        self.ref_gray = None
        self.roi_mask = None
        self.known_holes = []       # [(x, y, shot_index), ...]
        self.pending_holes = []     # [{"center": (x,y), "count": int, "seen": bool}, ...]
        self.shot_index = 0
        self.frames_since_last_shot = 999
        self.initialized = False

    def init_reference(self, warped_bgr: np.ndarray, roi_mask: np.ndarray = None):
        """用校準期結束時的攤平影像作為基準（乾淨無彈孔）。"""
        self.ref_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        # 中值濾波：專門剋制影片壓縮造成的區塊狀噪點 (Macroblocking artifacts)

        if roi_mask is not None:
            self.roi_mask = roi_mask.astype(bool)
        else:
            # 預設 ROI: 非純黑區域（warped 後黑邊不算）
            self.roi_mask = self.ref_gray > 20
            # 做一個小的腐蝕去掉邊緣
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            roi_u8 = self.roi_mask.astype(np.uint8) * 255
            roi_u8 = cv2.erode(roi_u8, kernel, iterations=2)
            self.roi_mask = roi_u8.astype(bool)
        self.known_holes.clear()
        self.pending_holes.clear()
        self.shot_index = 0
        self.frames_since_last_shot = 999
        self.initialized = True
        self.debug_frames = {}

    def detect(self, warped_bgr: np.ndarray):
        """
        偵測新彈孔。
        回傳: new_holes: [(x, y, shot_index), ...]
        """
        if not self.initialized:
            return []

        self.frames_since_last_shot += 1

        cur_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        # 中值濾波：專門剋制影片壓縮造成的區塊狀噪點 (Macroblocking artifacts)

        
        # ── 影像對齊 (Image Alignment) ──
        # 抵銷相機震動或靶紙物理晃動造成的像素位移
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            # 使用 ECC 演算法計算平移矩陣 (MOTION_TRANSLATION)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01)
            cv2.findTransformECC(self.ref_gray, cur_gray, warp_matrix, cv2.MOTION_TRANSLATION, criteria, None, 1)
            # 將 cur_gray 逆向平移推回與 ref_gray 完全重合的位置
            aligned_cur = cv2.warpAffine(cur_gray, warp_matrix, (cur_gray.shape[1], cur_gray.shape[0]), 
                                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, 
                                         borderMode=cv2.BORDER_REPLICATE)
        except cv2.error:
            # 如果對齊失敗（例如差異太大不收斂），則退回使用原始影像
            aligned_cur = cur_gray

        # 預設 Debug 畫面
        self.debug_frames['ref'] = self.ref_gray.copy() if self.ref_gray is not None else aligned_cur.copy()
        self.debug_frames['cur'] = aligned_cur.copy()
        self.debug_frames['diff'] = np.zeros_like(aligned_cur)
        self.debug_frames['mask'] = np.zeros_like(aligned_cur)
        self.debug_frames['candidates'] = cv2.cvtColor(aligned_cur, cv2.COLOR_GRAY2BGR)

        # 冷卻期內不偵測（避免同一槍重複計算）
        if self.frames_since_last_shot < self.cooldown_frames:
            return []

        # 幀差法: 只看變暗的部分 (ref_gray - aligned_cur，負值被 clip)
        diff = cv2.subtract(self.ref_gray, aligned_cur)
        self.debug_frames['diff'] = diff.copy()

        if self.roi_mask is not None:
            diff_masked = diff.copy()
            diff_masked[~self.roi_mask] = 0
        else:
            diff_masked = diff

        # 閾值 + 形態學
        _, mask = cv2.threshold(diff_masked, self.diff_thresh, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        self.debug_frames['mask'] = mask.copy()

        # 找輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area or area > self.max_contour_area:
                continue

            # 計算輪廓內平均亮度（必須是暗色區域）
            hole_mask = np.zeros_like(cur_gray, dtype=np.uint8)
            cv2.drawContours(hole_mask, [cnt], -1, 255, -1)
            mean_int = cv2.mean(cur_gray, mask=hole_mask)[0]
            if mean_int > self.max_hole_brightness:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])

            # 邊界過濾：忽略太靠近邊緣的雜訊 (通常是 TPS 攤平時的邊緣抖動)
            h, w = cur_gray.shape
            border_margin = 15
            if cx < border_margin or cx > w - border_margin or cy < border_margin or cy > h - border_margin:
                continue

            # 檢查中心點亮度，避免將白邊誤認為彈孔排除 ROI 外的候選
            if self.roi_mask is not None:
                if cy < 0 or cy >= self.roi_mask.shape[0] or cx < 0 or cx >= self.roi_mask.shape[1]:
                    continue
                if not self.roi_mask[cy, cx]:
                    continue

            # 排除已知彈孔附近的候選
            if not self._is_truly_new((cx, cy)):
                continue

            detected.append((cx, cy))
            cv2.drawContours(self.debug_frames['candidates'], [cnt], -1, (0, 255, 0), 1)

        # ── 時序持續性篩選 ──
        for p in self.pending_holes:
            p["seen"] = False

        for pt in detected:
            cx, cy = pt
            best_p = None
            best_d = float('inf')
            for p in self.pending_holes:
                d = np.hypot(cx - p["center"][0], cy - p["center"][1])
                if d <= self.distance_thresh and d < best_d:
                    best_d = d
                    best_p = p

            if best_p is None:
                self.pending_holes.append({
                    "center": (float(cx), float(cy)),
                    "count": 1,
                    "seen": True
                })
            else:
                x0, y0 = best_p["center"]
                n = best_p["count"]
                best_p["center"] = ((x0 * n + cx) / (n + 1), (y0 * n + cy) / (n + 1))
                best_p["count"] = n + 1
                best_p["seen"] = True

        # 確認持續出現的候選為新彈孔
        new_holes = []
        still_pending = []
        for p in self.pending_holes:
            # 繪製追蹤中的候選點 (黃色圈圈)
            px, py = int(p["center"][0]), int(p["center"][1])
            cv2.circle(self.debug_frames['candidates'], (px, py), int(self.known_hole_radius), (0, 255, 255), 1)
            cv2.putText(self.debug_frames['candidates'], str(p["count"]), (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            if p["seen"] and p["count"] >= self.persistence_frames:
                cx = int(round(p["center"][0]))
                cy = int(round(p["center"][1]))
                self.shot_index += 1
                hole_record = (cx, cy, self.shot_index)
                self.known_holes.append(hole_record)
                new_holes.append(hole_record)
                # 繪製確認的彈孔 (紅色)
                cv2.circle(self.debug_frames['candidates'], (cx, cy), 5, (0, 0, 255), -1)
            else:
                if p["seen"]:
                    still_pending.append(p)
        self.pending_holes = still_pending

        # 確認新彈孔後，只「局部修補」背景，絕對不可整張替換！
        if new_holes:
            for (cx, cy, _) in new_holes:
                # 只把新彈孔周圍的 25x25 像素更新到背景，保留其餘完美的乾淨背景
                r = int(self.known_hole_radius) + 5
                y1 = max(0, cy - r)
                y2 = min(self.ref_gray.shape[0], cy + r)
                x1 = max(0, cx - r)
                x2 = min(self.ref_gray.shape[1], cx + r)
                self.ref_gray[y1:y2, x1:x2] = aligned_cur[y1:y2, x1:x2]
            
            self.frames_since_last_shot = 0
        else:
            # 關閉全局 EMA 融合：因為 Homography 攤平必有 1~2 pixel 的微小抖動，
            # 若持續融合會讓背景的計分線變成模糊的「重影(Ghosting)」，進而引發邊緣誤判。
            pass

        return new_holes

    def _is_truly_new(self, pt):
        for hx, hy, _ in self.known_holes:
            if np.hypot(pt[0] - hx, pt[1] - hy) <= self.known_hole_radius:
                return False
        return True


# ==========================================
# Tracker 類別 (帶彈孔偵測器)
# ==========================================
class TargetTracker:
    def __init__(self, target_id):
        self.target_id = target_id
        self.color = TRACKER_COLORS[target_id % len(TRACKER_COLORS)]

        # Phase 1: 校準期
        self.calib_history_kpts = []
        self.calib_history_areas = []
        self.calib_history_centroids = []
        self.is_calibrated = False

        # Phase 2: 主動追蹤期
        self.global_min_mask_area = float('inf')
        self.global_optimal_kpts = None
        self.global_optimal_centroid = None
        self.has_new_candidate = False
        self.backup_min_mask_area = float('inf')
        self.backup_kpts = None
        self.backup_centroid = None

        self.warped_frame = None
        self.perfect_warped_frame = None
        self.last_valid_org_12_points = None

        # 彈孔偵測器 (針對此靶)
        self.shot_detector_raw = WarpedShotDetector()
        self.shot_detector_warped = WarpedShotDetector()
        self.shot_detector_initialized = False

        # IDPA 計分結果暫存
        self.all_holes_raw = []
        self.all_holes_warped = []
        self.raw_frame_scored = None


# ==========================================
# TPS 變形
# ==========================================
def warp_tps(src_img, src_points, dst_points):
    tps = cv2.createThinPlateSplineShapeTransformer()
    pts_src = src_points.reshape(1, -1, 2).astype(np.float32)
    pts_dst = dst_points.reshape(1, -1, 2).astype(np.float32)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
    tps.estimateTransformation(pts_dst, pts_src, matches)
    out_img = tps.warpImage(src_img)
    return out_img


# ==========================================
# 在攤平靶紙上繪製計分區域
# ==========================================
def draw_scoring_zones(img, target_w, target_h, scorer: IDPAScorer):
    """在攤平靶紙影像上繪製 IDPA 計分區域輔助線。"""
    overlay = img.copy()
    t = scorer.target

    # 繪製 -3 區邊界
    pts_3 = np.array(t.boundary_3, dtype=np.int32)
    cv2.polylines(overlay, [pts_3], True, (0, 165, 255), 1, cv2.LINE_AA)

    # 繪製 -1 區 (頭部)
    pts_h1 = np.array(t.head_1, dtype=np.int32)
    cv2.polylines(overlay, [pts_h1], True, (0, 255, 255), 1, cv2.LINE_AA)

    # 繪製 -1 區 (身體)
    pts_b1 = np.array(t.body_1, dtype=np.int32)
    cv2.polylines(overlay, [pts_b1], True, (0, 255, 255), 1, cv2.LINE_AA)

    # 繪製 -0 區 (頭部橢圓)
    cv2.ellipse(overlay,
                (int(t.head_0['cx']), int(t.head_0['cy'])),
                (int(t.head_0['rx']), int(t.head_0['ry'])),
                0, 0, 360, (0, 255, 0), 1, cv2.LINE_AA)

    # 繪製 -0 區 (身體橢圓)
    cv2.ellipse(overlay,
                (int(t.body_0['cx']), int(t.body_0['cy'])),
                (int(t.body_0['rx']), int(t.body_0['ry'])),
                0, 0, 360, (0, 255, 0), 1, cv2.LINE_AA)

    # 半透明疊加
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    return img


# ==========================================
# 在攤平靶紙上繪製彈孔標註
# ==========================================
def draw_holes_on_warped(img, holes, scorer: IDPAScorer):
    """
    在攤平靶紙影像上標註所有彈孔。
    每個彈孔標示：發次序號 + 計分區。
    """
    for (x, y, shot_idx) in holes:
        zone = scorer.target.get_hit_zone(x, y)
        color = ZONE_COLORS.get(zone, (255, 255, 255))
        penalty = ZONE_PENALTY.get(zone, 5)

        # 彈孔圓圈
        cv2.circle(img, (x, y), 8, color, 2, cv2.LINE_AA)
        cv2.circle(img, (x, y), 2, color, -1, cv2.LINE_AA)

        # 標註文字: "#發次 -區 (+罰秒)"
        zone_str = f"-{zone}" if isinstance(zone, int) else "Miss"
        label = f"#{shot_idx} {zone_str}(+{penalty}s)"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        # 文字背景
        tx, ty = x + 10, y - 5
        cv2.rectangle(img, (tx - 1, ty - text_size[1] - 2),
                      (tx + text_size[0] + 1, ty + 2), (0, 0, 0), -1)
        cv2.putText(img, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return img


# ==========================================
# 計分面板繪製
# ==========================================
def draw_score_panel_generic(target_w, target_h, target_id, title, color, holes, scorer: IDPAScorer):
    """繪製計分面板，顯示每發擊中資訊與總計分。"""
    panel = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # 深灰背景

    y_cursor = 30
    line_h = 22

    # 標題
    cv2.putText(panel, f"Target ID:{target_id} {title}",
                (10, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    y_cursor += line_h + 5

    cv2.line(panel, (10, y_cursor), (target_w - 10, y_cursor), (100, 100, 100), 1)
    y_cursor += 10

    total_penalty = 0

    if not holes:
        cv2.putText(panel, "No shots detected",
                    (10, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        y_cursor += line_h
    else:
        for (x, y, shot_idx) in holes:
            zone = scorer.target.get_hit_zone(x, y)
            color_text = ZONE_COLORS.get(zone, (255, 255, 255))
            penalty = ZONE_PENALTY.get(zone, 5)
            total_penalty += penalty

            zone_str = f"-{zone}" if isinstance(zone, int) else "Miss"
            text = f"#{shot_idx}: ({x},{y}) -> {zone_str} (+{penalty}s)"

            if y_cursor < target_h - 60:
                cv2.putText(panel, text, (10, y_cursor),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color_text, 1, cv2.LINE_AA)
                y_cursor += line_h

    # 底部總計
    cv2.line(panel, (10, target_h - 50), (target_w - 10, target_h - 50), (100, 100, 100), 1)
    cv2.putText(panel, f"Total Shots: {len(holes)}",
                (10, target_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, f"Points Down: +{total_penalty}s",
                (10, target_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    return panel


# ==========================================
# 主管線
# ==========================================
def run_pipeline():
    target_w, target_h = 285, 468

    # 標準 17 點 & 12 點像素座標
    ideal_pts_17 = NORMALIZED_POINTS_17 * [target_w - 1, target_h - 1]
    ideal_pts_12 = ideal_pts_17[OUTLINE_INDICES]

    model_path = settings.get_path("result", "train_pose_17kpt_merged", "run-3", "weights", "best.pt")
    video_path = settings.get_path("data", "靶場", "G5_S03.mp4")
    model = YOLO(model_path)

    # 初始化 IDPA 計分器 (與攤平影像同尺寸)
    scorer = IDPAScorer(target_width_pixels=target_w, target_height_pixels=target_h)

    # 輸出目錄
    output_dir = settings.get_path("result", "scoring_output")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 錯誤: 無法讀取影片檔案 {video_path}")
        return

    window_name = "IDPA Scoring Pipeline"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"啟動 IDPA 計分管線...")
    print(f"影片: {video_path}")
    print(f"輸出: {output_dir}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30
    calibration_frames_target = int(fps * 3)  # 3 秒校準

    frame_count = 0
    active_trackers = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display_frame = frame.copy()

        for t in active_trackers:
            t.has_new_candidate = False

        # ── 影像銳利化 ──
        # 使用更強的銳利化卷積核，最大幅增強邊緣對比度，幫助 YOLO Pose 更穩定地抓取靶紙特徵點
        sharpen_kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
        sharpened_frame = cv2.filter2D(frame, -1, sharpen_kernel)

        results = model.predict(sharpened_frame, conf=0.6, imgsz=640, device=0, verbose=False)

        # ── 解析所有偵測到的目標 ──
        valid_detections = []
        if results and results[0].keypoints is not None:
            all_kpts = results[0].keypoints.xy
            for det_idx in range(len(all_kpts)):
                kpts_17 = all_kpts[det_idx].cpu().numpy()[:17]
                if len(kpts_17) < 17:
                    continue

                outline = kpts_17[OUTLINE_INDICES]
                area = cv2.contourArea(outline.astype(np.float32))
                if area < 500:
                    continue

                cx = np.mean(outline[:, 0])
                cy = np.mean(outline[:, 1])
                valid_detections.append({
                    "kpts_17": kpts_17,
                    "outline": outline,
                    "area": area,
                    "cx": cx, "cy": cy
                })

        # ==========================================================
        # Phase 1: 校準收集期
        # ==========================================================
        if frame_count <= calibration_frames_target + 10:
            cv2.putText(display_frame,
                        f"Phase 1: Calibration ({frame_count}/{calibration_frames_target})",
                        (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            for m in valid_detections:
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
                        best_t.calib_history_kpts.append(m["kpts_17"])
                        best_t.calib_history_areas.append(m["area"])
                        best_t.calib_history_centroids.append((m["cx"], m["cy"]))
                else:
                    new_t = TargetTracker(len(active_trackers))
                    new_t.calib_history_kpts.append(m["kpts_17"])
                    new_t.calib_history_areas.append(m["area"])
                    new_t.calib_history_centroids.append((m["cx"], m["cy"]))
                    active_trackers.append(new_t)

            for t in active_trackers:
                if not t.is_calibrated and len(t.calib_history_areas) > (calibration_frames_target * 0.4):
                    if frame_count > calibration_frames_target:
                        median_area = np.median(t.calib_history_areas)
                        diffs = np.abs(np.array(t.calib_history_areas) - median_area)
                        best_idx = np.argmin(diffs)

                        t.global_min_mask_area = t.calib_history_areas[best_idx]
                        t.global_optimal_kpts = t.calib_history_kpts[best_idx].copy()
                        t.global_optimal_centroid = t.calib_history_centroids[best_idx]
                        t.is_calibrated = True
                        print(f"✅ Target ID:{t.target_id} 鎖定完成！基準面積: {int(t.global_min_mask_area)}")

        # ==========================================================
        # Phase 2: 主動追蹤 + 彈孔偵測
        # ==========================================================
        else:
            cv2.putText(display_frame,
                        f"Phase 2: Tracking + Scoring ({len([t for t in active_trackers if t.is_calibrated])} Targets)",
                        (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            for m in valid_detections:
                best_t = None
                min_dist = float('inf')
                for t in active_trackers:
                    if t.is_calibrated:
                        dist = np.hypot(m["cx"] - t.global_optimal_centroid[0],
                                        m["cy"] - t.global_optimal_centroid[1])
                        if dist < 300 and dist < min_dist:
                            min_dist = dist
                            best_t = t

                if best_t:
                    if m["area"] < best_t.global_min_mask_area:
                        if (best_t.global_min_mask_area - m["area"]) <= (best_t.global_min_mask_area * 0.05):
                            best_t.backup_min_mask_area = best_t.global_min_mask_area
                            best_t.backup_kpts = best_t.global_optimal_kpts.copy()
                            best_t.backup_centroid = best_t.global_optimal_centroid
                            best_t.has_new_candidate = True

                            best_t.global_min_mask_area = m["area"]
                            best_t.global_optimal_kpts = m["kpts_17"].copy()
                            best_t.global_optimal_centroid = (m["cx"], m["cy"])

        # ── 對每個校準完成的標靶進行 Homography + TPS 提取 + 彈孔偵測 ──
        for t in active_trackers:
            if not t.is_calibrated or t.global_optimal_kpts is None:
                continue

            kpts_17 = t.global_optimal_kpts
            src_outline = kpts_17[OUTLINE_INDICES].astype(np.float32)
            poly = src_outline.astype(np.int32)

            # 畫外框多邊形
            cv2.polylines(display_frame, [poly], True, t.color, 3)
            cv2.putText(display_frame, f"ID:{t.target_id}",
                        (int(t.global_optimal_centroid[0] - 25), int(t.global_optimal_centroid[1] - 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, t.color, 3)

            # ── Homography ──
            M, _ = cv2.findHomography(src_outline, ideal_pts_12, 0)

            if M is not None:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                b, g, r = cv2.split(masked_frame)
                bgra_frame = cv2.merge((b, g, r, mask))

                warped_frame_bgra = cv2.warpPerspective(bgra_frame, M, (target_w, target_h))
                temp_warped = warped_frame_bgra[:, :, :3].copy()

                flat_17_points = cv2.perspectiveTransform(
                    kpts_17.reshape(-1, 1, 2).astype(np.float32), M
                ).reshape(-1, 2)
                flat_12_points = flat_17_points[OUTLINE_INDICES]

                try:
                    M_inv = np.linalg.inv(M)
                    org_12_points = cv2.perspectiveTransform(
                        flat_12_points.reshape(-1, 1, 2), M_inv
                    ).reshape(-1, 2)
                except np.linalg.LinAlgError:
                    continue

                # Homography warped 上畫標記
                flat_poly = flat_12_points.astype(np.int32)
                cv2.polylines(temp_warped, [flat_poly], True, (255, 0, 255), 2)

                for i, pt in enumerate(flat_12_points):
                    ideal_pt = ideal_pts_12[i]
                    cv2.drawMarker(temp_warped, (int(ideal_pt[0]), int(ideal_pt[1])),
                                   (255, 255, 0), cv2.MARKER_CROSS, 10, 1)
                    cv2.line(temp_warped,
                             (int(ideal_pt[0]), int(ideal_pt[1])),
                             (int(pt[0]), int(pt[1])),
                             (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(temp_warped, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

                # ── TPS 精修 ──
                clean_warped_frame_bgra = warped_frame_bgra.copy()
                perfect_warped_frame_bgra = warp_tps(clean_warped_frame_bgra, flat_17_points, ideal_pts_17)
                temp_perfect = perfect_warped_frame_bgra[:, :, :3].copy()
                
                # ── Raw Crop (BBOX 切割並 Resize) ──
                min_x, min_y = np.min(kpts_17, axis=0)
                max_x, max_y = np.max(kpts_17, axis=0)
                min_x = max(0, int(min_x))
                min_y = max(0, int(min_y))
                max_x = min(frame.shape[1], int(max_x))
                max_y = min(frame.shape[0], int(max_y))
                
                raw_crop = frame[min_y:max_y, min_x:max_x]
                if raw_crop.size > 0:
                    temp_raw = cv2.resize(raw_crop, (target_w, target_h))
                else:
                    temp_raw = np.zeros((target_h, target_w, 3), dtype=np.uint8)

                # ── 候選驗證 ──
                is_valid_candidate = True
                if t.has_new_candidate:
                    alpha_channel = perfect_warped_frame_bgra[:, :, 3]
                    ideal_poly = ideal_pts_12.astype(np.int32)
                    target_mask = np.zeros((target_h, target_w), dtype=np.uint8)
                    cv2.fillPoly(target_mask, [ideal_poly], 255)
                    target_mask = cv2.erode(target_mask, np.ones((3, 3), np.uint8), iterations=1)

                    transparent_void = cv2.bitwise_and(
                        np.uint8(alpha_channel < 100) * 255,
                        np.uint8(alpha_channel < 100) * 255,
                        mask=target_mask
                    )
                    void_count = np.count_nonzero(transparent_void)

                    frame_cost = np.sum(np.linalg.norm(flat_12_points - ideal_pts_12, axis=1))

                    if void_count > 100:
                        t.global_min_mask_area = t.backup_min_mask_area
                        t.global_optimal_kpts = t.backup_kpts
                        t.global_optimal_centroid = t.backup_centroid
                        is_valid_candidate = False
                    else:
                        print(f"🌟 Target ID:{t.target_id} 更新！面積: {int(t.global_min_mask_area)} (Cost:{int(frame_cost)})")

                if is_valid_candidate:
                    # ── 初始化雙軌彈孔偵測器 ──
                    if not t.shot_detector_initialized:
                        t.shot_detector_raw.init_reference(temp_raw)
                        t.shot_detector_warped.init_reference(temp_perfect)
                        t.shot_detector_initialized = True
                        print(f"🔫 Target ID:{t.target_id} 雙軌偵測器初始化完成")
                    else:
                        # ── 在 raw 影像上偵測 ──
                        new_holes_raw = t.shot_detector_raw.detect(temp_raw)
                        if new_holes_raw:
                            for (hx, hy, shot_idx) in new_holes_raw:
                                t.all_holes_raw.append((hx, hy, shot_idx))

                        # ── 在攤平影像上偵測 ──
                        new_holes_warped = t.shot_detector_warped.detect(temp_perfect)
                        if new_holes_warped:
                            for (hx, hy, shot_idx) in new_holes_warped:
                                t.all_holes_warped.append((hx, hy, shot_idx))

                    # 繪製計分區域與彈孔 (Raw)
                    scored_raw = temp_raw.copy()
                    draw_holes_on_warped(scored_raw, t.all_holes_raw, scorer)
                    cv2.putText(scored_raw, "[NO WARP]", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    t.raw_frame_scored = scored_raw

                    # 繪製計分區域與彈孔 (Warped)
                    scored_perfect = temp_perfect.copy()
                    draw_scoring_zones(scored_perfect, target_w, target_h, scorer)
                    draw_holes_on_warped(scored_perfect, t.all_holes_warped, scorer)
                    cv2.putText(scored_perfect, "[WARPED]", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    t.warped_frame = temp_warped
                    t.perfect_warped_frame = scored_perfect
                    t.last_valid_org_12_points = org_12_points

                # 在原圖上畫特徵點
                pts_to_draw = t.last_valid_org_12_points if t.last_valid_org_12_points is not None else org_12_points
                for i, pt in enumerate(pts_to_draw):
                    cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

        # ==========================================
        # 動態表格(Grid)儀表板合成顯示
        # ==========================================
        calibrated_trackers = [t for t in active_trackers if t.is_calibrated]
        num_targets = len(calibrated_trackers)

        # 每個靶的顯示區: 上半部(Raw), 下半部(Warped)
        # 每列寬度為 3 個 block (Scored, Diff, Score Panel)
        block_w = target_w * 3
        block_h = target_h * 2

        if num_targets > 0:
            cols = math.ceil(math.sqrt(num_targets))
            if num_targets == 2:
                cols = 2
            rows = math.ceil(num_targets / cols)

            grid_w = cols * block_w
            grid_h = rows * block_h
            grid_view = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            for i, t in enumerate(calibrated_trackers):
                r = i // cols
                c = i % cols
                x_offset = c * block_w
                y_offset = r * block_h

                if t.perfect_warped_frame is not None and t.raw_frame_scored is not None:
                    # ===== 上半部 (RAW) =====
                    # 1. Raw Scored
                    grid_view[y_offset:y_offset + target_h,
                              x_offset:x_offset + target_w] = t.raw_frame_scored
                    
                    # 2. Raw Mask (取代 Diff，因為 Diff 數值太小肉眼看不見)
                    raw_mask = t.shot_detector_raw.debug_frames.get('mask', np.zeros((target_h, target_w), dtype=np.uint8))
                    raw_mask_bgr = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
                    cv2.putText(raw_mask_bgr, "Raw Mask", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    grid_view[y_offset:y_offset + target_h,
                              x_offset + target_w:x_offset + target_w * 2] = raw_mask_bgr
                    
                    # 3. Raw Score Panel
                    raw_panel = draw_score_panel_generic(target_w, target_h, t.target_id, "Raw Score", (0, 0, 255), t.all_holes_raw, scorer)
                    grid_view[y_offset:y_offset + target_h,
                              x_offset + target_w * 2:x_offset + target_w * 3] = raw_panel

                    # ===== 下半部 (WARPED) =====
                    # 1. Warped Scored
                    grid_view[y_offset + target_h:y_offset + target_h * 2,
                              x_offset:x_offset + target_w] = t.perfect_warped_frame
                    
                    # 2. Warped Mask (取代 Diff)
                    warped_mask = t.shot_detector_warped.debug_frames.get('mask', np.zeros((target_h, target_w), dtype=np.uint8))
                    warped_mask_bgr = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)
                    cv2.putText(warped_mask_bgr, "Warped Mask", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    grid_view[y_offset + target_h:y_offset + target_h * 2,
                              x_offset + target_w:x_offset + target_w * 2] = warped_mask_bgr
                    
                    # 3. Warped Score Panel
                    warped_panel = draw_score_panel_generic(target_w, target_h, t.target_id, "Warped Score", (0, 255, 0), t.all_holes_warped, scorer)
                    grid_view[y_offset + target_h:y_offset + target_h * 2,
                              x_offset + target_w * 2:x_offset + target_w * 3] = warped_panel

            h, w = display_frame.shape[:2]
            ratio = grid_h / h
            resized_display = cv2.resize(display_frame, (int(w * ratio), grid_h))
            combined_view = np.hstack((resized_display, grid_view))
        else:
            h, w = display_frame.shape[:2]
            ratio = target_h / h
            resized_display = cv2.resize(display_frame, (int(w * ratio), target_h))
            empty_grid = np.zeros((target_h, target_w * 3, 3), dtype=np.uint8)
            combined_view = np.hstack((resized_display, empty_grid))

        cv2.imshow(window_name, combined_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    # ==========================================
    # 結束：輸出最終計分報告
    # ==========================================
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("  IDPA 計分報告 (RAW vs WARPED 對比)")
    print("=" * 60)

    for t in [t for t in active_trackers if t.is_calibrated]:
        print(f"\n🎯 Target ID:{t.target_id}")
        
        # 輸出 RAW
        print(f"  [RAW 模式] 偵測到 {len(t.all_holes_raw)} 發")
        if t.all_holes_raw:
            hits_xy_raw = [(x, y) for x, y, _ in t.all_holes_raw]
            result_raw = scorer.score_single_target(hits_xy_raw, shots_required=2)
            print(f"  [RAW 模式] Points Down: +{result_raw.get('total_penalty', 0)}s")
            
        # 輸出 WARPED
        print(f"  [WARPED 模式] 偵測到 {len(t.all_holes_warped)} 發")
        if t.all_holes_warped:
            hits_xy_warped = [(x, y) for x, y, _ in t.all_holes_warped]
            result_warped = scorer.score_single_target(hits_xy_warped, shots_required=2)
            print(f"  [WARPED 模式] Points Down: +{result_warped.get('total_penalty', 0)}s")

    print("\n✅ 處理結束。")

if __name__ == "__main__":
    run_pipeline()
