"""
YOLO Pose 版多目標追蹤管線
以 YOLO-Pose 直接偵測 17 個關鍵點取代 YOLO-Seg 的輪廓提取流程。
保留原 main.py 的：校準期 → 主動追蹤期 → Homography + TPS 精修 → Grid 儀表板。
"""
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

# 外框 12 點索引 (用於多邊形遮罩與 Homography)
OUTLINE_INDICES = list(range(12))

# 為了區分多個標靶，設定預設顏色序列
TRACKER_COLORS = [
    (0, 255, 255),   # 黃
    (0, 255, 0),     # 綠
    (255, 0, 255),   # 紫
    (0, 165, 255),   # 橙
    (255, 200, 0),   # 淺藍
    (255, 0, 0)      # 藍
]

# ==========================================
# Tracker 類別
# ==========================================
class TargetTracker:
    def __init__(self, target_id):
        self.target_id = target_id
        self.color = TRACKER_COLORS[target_id % len(TRACKER_COLORS)]

        # 第一階段：校準期 (Phase 1)
        self.calib_history_kpts = []       # 17 keypoints per frame
        self.calib_history_areas = []
        self.calib_history_centroids = []
        self.is_calibrated = False

        # 第二階段：主動追蹤期 (Phase 2)
        self.global_min_mask_area = float('inf')
        self.global_optimal_kpts = None     # (17, 2) best keypoints
        self.global_optimal_centroid = None
        self.has_new_candidate = False
        self.backup_min_mask_area = float('inf')
        self.backup_kpts = None
        self.backup_centroid = None

        self.warped_frame = None
        self.perfect_warped_frame = None
        self.last_valid_org_12_points = None

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
# 主管線
# ==========================================
def run_pipeline():
    target_w, target_h = 285, 468

    # 標準 17 點 & 12 點像素座標
    ideal_pts_17 = NORMALIZED_POINTS_17 * [target_w - 1, target_h - 1]
    ideal_pts_12 = ideal_pts_17[OUTLINE_INDICES]

    model_path = settings.get_path("result", "train_pose_17kpt", "run7", "weights", "best.pt")
    video_path = settings.get_path("data", "靶場", "G7_S03.mp4")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 錯誤: 無法讀取影片檔案 {video_path}")
        return

    window_name = "YOLO Pose Multi-Target Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"啟動 YOLO Pose 多目標追蹤管線...")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps): fps = 30
    calibration_frames_target = int(fps * 3)  # 3秒校準

    frame_count = 0
    active_trackers = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        display_frame = frame.copy()

        # 重置候選標記
        for t in active_trackers:
            t.has_new_candidate = False

        results = model.predict(frame, conf=0.6, imgsz=640, device=0, verbose=False)

        # ── 解析所有偵測到的目標 ──
        valid_detections = []
        if results and results[0].keypoints is not None:
            all_kpts = results[0].keypoints.xy  # (N, 17, 2)
            for det_idx in range(len(all_kpts)):
                kpts_17 = all_kpts[det_idx].cpu().numpy()[:17]
                if len(kpts_17) < 17:
                    continue

                # 用外框 12 點計算面積與質心
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
        # 第一階段: 校準收集期
        # ==========================================================
        if frame_count <= calibration_frames_target + 10:
            cv2.putText(display_frame,
                        f"Phase 1: Multi Calibration ({frame_count}/{calibration_frames_target})",
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

            # 校準畢業門檻
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
        # 第二階段: 獨立主動追蹤期
        # ==========================================================
        else:
            cv2.putText(display_frame,
                        f"Phase 2: Active Tracking ({len([t for t in active_trackers if t.is_calibrated])} Targets)",
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
                    # 2.5% 防呆門檻更新最小面積
                    if m["area"] < best_t.global_min_mask_area:
                        if (best_t.global_min_mask_area - m["area"]) <= (best_t.global_min_mask_area * 0.05):
                            best_t.backup_min_mask_area = best_t.global_min_mask_area
                            best_t.backup_kpts = best_t.global_optimal_kpts.copy()
                            best_t.backup_centroid = best_t.global_optimal_centroid
                            best_t.has_new_candidate = True

                            best_t.global_min_mask_area = m["area"]
                            best_t.global_optimal_kpts = m["kpts_17"].copy()
                            best_t.global_optimal_centroid = (m["cx"], m["cy"])

        # ── 對每個校準完成的標靶進行 Homography + TPS 提取 ──
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

            # ── Homography: 用外框 12 點直接對齊 ──
            M, _ = cv2.findHomography(src_outline, ideal_pts_12, 0)

            if M is not None:
                # 建立遮罩去背
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                b, g, r = cv2.split(masked_frame)
                bgra_frame = cv2.merge((b, g, r, mask))

                # Homography 變形
                warped_frame_bgra = cv2.warpPerspective(bgra_frame, M, (target_w, target_h))
                temp_warped = warped_frame_bgra[:, :, :3].copy()

                # 將 17 個原始關鍵點投影到標準空間
                flat_17_points = cv2.perspectiveTransform(
                    kpts_17.reshape(-1, 1, 2).astype(np.float32), M
                ).reshape(-1, 2)
                flat_12_points = flat_17_points[OUTLINE_INDICES]

                # 反投影回原圖空間 (用於畫面上標記)
                try:
                    M_inv = np.linalg.inv(M)
                    org_12_points = cv2.perspectiveTransform(
                        flat_12_points.reshape(-1, 1, 2), M_inv
                    ).reshape(-1, 2)
                except np.linalg.LinAlgError:
                    continue

                # 在 warped 圖上畫標記
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
                    cv2.putText(temp_warped, str(i), (int(pt[0])+5, int(pt[1])-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # ── TPS 精修: 用全部 17 點做薄板樣條變形 ──
                clean_warped_frame_bgra = warped_frame_bgra.copy()
                perfect_warped_frame_bgra = warp_tps(clean_warped_frame_bgra, flat_17_points, ideal_pts_17)
                temp_perfect = perfect_warped_frame_bgra[:, :, :3].copy()

                # ── 候選驗證 (防止 YOLO 邊緣震盪) ──
                is_valid_candidate = True
                if t.has_new_candidate:
                    alpha_channel = perfect_warped_frame_bgra[:, :, 3]
                    ideal_poly = ideal_pts_12.astype(np.int32)
                    target_mask = np.zeros((target_h, target_w), dtype=np.uint8)
                    cv2.fillPoly(target_mask, [ideal_poly], 255)
                    target_mask = cv2.erode(target_mask, np.ones((3,3), np.uint8), iterations=1)

                    transparent_void = cv2.bitwise_and(
                        np.uint8(alpha_channel < 100) * 255,
                        np.uint8(alpha_channel < 100) * 255,
                        mask=target_mask
                    )
                    void_count = np.count_nonzero(transparent_void)

                    # 計算 12 點投影偏差作為品質指標
                    frame_cost = np.sum(np.linalg.norm(flat_12_points - ideal_pts_12, axis=1))

                    if void_count > 100:
                        t.global_min_mask_area = t.backup_min_mask_area
                        t.global_optimal_kpts = t.backup_kpts
                        t.global_optimal_centroid = t.backup_centroid
                        is_valid_candidate = False
                    elif frame_cost > 3000:
                        t.global_min_mask_area = t.backup_min_mask_area
                        t.global_optimal_kpts = t.backup_kpts
                        t.global_optimal_centroid = t.backup_centroid
                        is_valid_candidate = False
                    else:
                        print(f"🌟 Target ID:{t.target_id} 更新！面積: {int(t.global_min_mask_area)} (Cost:{int(frame_cost)})")

                if is_valid_candidate:
                    t.warped_frame = temp_warped
                    t.perfect_warped_frame = temp_perfect
                    t.last_valid_org_12_points = org_12_points
                    cv2.putText(t.perfect_warped_frame, f"ID:{t.target_id} TPS Ready",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, t.color, 2)

                # 在原圖上畫 12 個特徵點
                pts_to_draw = t.last_valid_org_12_points if t.last_valid_org_12_points is not None else org_12_points
                for i, pt in enumerate(pts_to_draw):
                    cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
                    cv2.putText(display_frame, str(i), (int(pt[0])+5, int(pt[1])-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ==========================================
        # 動態表格(Grid)儀表板合成顯示
        # ==========================================
        calibrated_trackers = [t for t in active_trackers if t.is_calibrated]
        num_targets = len(calibrated_trackers)

        block_w = target_w * 2
        block_h = target_h

        if num_targets > 0:
            cols = math.ceil(math.sqrt(num_targets))
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
                    grid_view[y_offset:y_offset+block_h, x_offset:x_offset+target_w] = t.warped_frame
                    grid_view[y_offset:y_offset+block_h, x_offset+target_w:x_offset+target_w*2] = t.perfect_warped_frame

            h, w = display_frame.shape[:2]
            ratio = grid_h / h
            resized_display = cv2.resize(display_frame, (int(w * ratio), grid_h))
            combined_view = np.hstack((resized_display, grid_view))
        else:
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
