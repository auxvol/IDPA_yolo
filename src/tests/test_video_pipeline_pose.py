import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO

# 取得目前檔案路徑，並定位到專案根目錄
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from config.config import settings

# 標準 IDPA 靶紙 17 點歸一化座標 (0~16)
# 0~11: 外框輪廓點, 12~16: 內部結構點
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

# 外框 12 點的索引 (用於多邊形遮罩與 Homography)
OUTLINE_INDICES = list(range(12))

# 多目標顏色序列
TARGET_COLORS = [
    (0, 255, 255),   # 黃
    (0, 255, 0),     # 綠
    (255, 0, 255),   # 紫
    (0, 165, 255),   # 橙
    (255, 200, 0),   # 淺藍
    (255, 0, 0),     # 藍
]

def run_pipeline():
    # 1. 初始化解析度與目標路徑
    target_w, target_h = 279, 467
    
    # 計算此解析度下的 17 點目標座標
    dst_points_17 = NORMALIZED_POINTS_17 * [target_w - 1, target_h - 1]
    # 外框 12 點 (用於 Homography)
    dst_points_12 = dst_points_17[OUTLINE_INDICES]
    
    # 讀取 YOLO Pose 模型的權重 (17 關鍵點版本)
    model_path = settings.get_path("result", "train_pose_17kpt", "run7", "weights", "best.pt")

    video_path = settings.get_path("data/室內箱/G11_S06.mp4")
    model = YOLO(model_path) 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 錯誤: 無法讀取影片檔案 {video_path}")
        return

    window_name = "YOLO11-Pose Pipeline (17-Pt Multi-Target)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"🚀 啟動姿勢估計 (Yolo Pose 17kpt) + 多目標直接變形管線")
    print(f"📐 標準比例尺框: {target_w} x {target_h}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 2. YOLO Pose 推論
        results = model.predict(frame, conf=0.5, imgsz=640, device=0, verbose=False)
        display_frame = frame.copy()
        warped_frames = []

        if results and results[0].keypoints is not None:
            all_keypoints = results[0].keypoints.xy
            num_targets = len(all_keypoints)

            for t_idx in range(num_targets):
                if len(all_keypoints[t_idx]) < 17:
                    continue

                color = TARGET_COLORS[t_idx % len(TARGET_COLORS)]
                all_17_pts = all_keypoints[t_idx].cpu().numpy()[:17]

                # 外框 12 點 (用於遮罩與 Homography)
                src_outline = all_17_pts[OUTLINE_INDICES]
                poly = src_outline.astype(np.int32)

                # A. 影像裁切去背：用外框 12 點圍成多邊形遮罩
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

                # 在畫面上畫出外框多邊形 (使用該目標專屬顏色)
                cv2.polylines(display_frame, [poly], True, color, 2)

                # 標註目標 ID
                centroid_x = int(np.mean(src_outline[:, 0]))
                centroid_y = int(np.mean(src_outline[:, 1]))
                cv2.putText(display_frame, f"T{t_idx}", (centroid_x - 10, centroid_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # B. 使用外框 12 點計算 Homography
                M, _ = cv2.findHomography(src_outline, dst_points_12, 0)

                if M is not None:
                    warped = cv2.warpPerspective(masked_frame, M, (target_w, target_h))
                    warped_frames.append(warped)

                    # 視覺化：畫上全部 17 個關鍵點
                    for i, (px, py) in enumerate(all_17_pts.astype(np.int32)):
                        kpt_color = color if i < 12 else (255, 0, 255)
                        cv2.circle(display_frame, (int(px), int(py)), 5, kpt_color, -1)
                        cv2.putText(display_frame, str(i), (int(px)+5, int(py)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # 3. 合併顯示：左側原圖 + 右側所有 warped 結果縱向排列
        h, w = display_frame.shape[:2]
        ratio = target_h / h
        resized_display = cv2.resize(display_frame, (int(w * ratio), target_h))

        if warped_frames:
            # 將多個 warped 結果垂直堆疊，再 resize 使高度與 display 一致
            warped_stack = np.vstack(warped_frames)
            ws_h, ws_w = warped_stack.shape[:2]
            scale = target_h / ws_h
            warped_resized = cv2.resize(warped_stack, (int(ws_w * scale), target_h))
            combined_view = np.hstack((resized_display, warped_resized))
        else:
            blank = np.zeros((target_h, target_w, 3), dtype="uint8")
            combined_view = np.hstack((resized_display, blank))
        
        cv2.imshow(window_name, combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 處理結束。")

if __name__ == "__main__":
    run_pipeline()
