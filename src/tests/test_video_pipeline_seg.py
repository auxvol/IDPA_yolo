import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO

# 取得目前檔案路徑，並定位到專案根目錄
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from config.config import settings

# 這是由用戶提供的標準 IDPA 靶紙 12 點歸一化座標
NORMALIZED_POINTS = np.array([
    [0.329828, 0.203805], # 0: 左頸部
    [0.329828, 0.000000], # 1: 左頭頂 (最高點, Y=0)
    [0.679026, 0.000000], # 2: 右頭頂 (最高點, Y=0)
    [0.679026, 0.203805], # 3: 右頸部
    [0.892086, 0.208972], # 4: 右肩內側
    [1.000000, 0.278788], # 5: 右肩外緣 (最右點, X=1)
    [0.999447, 0.810260], # 6: 右腰部
    [0.827892, 0.996325], # 7: 右底角
    [0.173215, 1.000000], # 8: 左底角 (最低點, Y=1)
    [0.000000, 0.810260], # 9: 左腰部 (最左點, X=0)
    [0.008301, 0.278788], # 10: 左肩外緣
    [0.116215, 0.206634]  # 11: 左肩內側
], dtype=np.float32)

def extract_12_points(contour_pts, bbox):
    """
    從 Segmentation 的不規則邊緣 (contour_pts) 中，
    逼近計算出最符合 12 個特定幾何點的像素座標。
    """
    bx, by, bw, bh = bbox
    src_pts = []
    
    # 對每一根標準點位，去尋找畫面上與其預期位置「距離最近」的實際遮罩輪廓點
    for nx, ny in NORMALIZED_POINTS:
        target_x = bx + nx * bw
        target_y = by + ny * bh
        
        # 計算 contour 所有點與目標點的距離
        distances = np.linalg.norm(contour_pts - np.array([target_x, target_y]), axis=1)
        best_idx = np.argmin(distances)
        src_pts.append(contour_pts[best_idx])
        
    return np.array(src_pts, dtype=np.float32)

def run_pipeline():
    # 1. 初始化解析度與目標路徑
    target_w, target_h = 279, 467
    
    dst_points = NORMALIZED_POINTS * [target_w - 1, target_h - 1]
    
    # 讀取專屬於 Segmentation 的模型權重目錄
    model_path = settings.get_path("result/train_seg/run2/weights/best.pt")
    video_path = settings.get_path("data/靶場/G5_S01.mp4")
    model = YOLO(model_path) 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 錯誤: 無法讀取影片檔案 {video_path}")
        return

    window_name = "YOLO11-Seg Pipeline (12-Corner Approximated Warp)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"🚀 啟動語意分割 (Yolo Seg) + 12幾何點逼近變形管線")
    print(f"📐 標準比例尺框: {target_w} x {target_h}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 2. YOLO Seg 推論
        results = model.predict(frame, conf=0.5, imgsz=640, device=0, verbose=False)
        display_frame = frame.copy()
        warped_frame = np.zeros((target_h, target_w, 3), dtype="uint8")

        # Segmentation 特有：結果保存在 .masks.xy 陣列裡
        if results and results[0].masks is not None:
            xy = results[0].masks.xy
            if len(xy) > 0 and len(xy[0]) >= 12:
                contour_pts = xy[0]
                poly = contour_pts.astype(np.int32)
                
                # A. 影像裁切去背：把模型直出的遮罩輪廓當邊界
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                # 把靶紙以外的背景一律變黑
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                
                # 在畫面上用綠色細線畫出模型判斷的不規則形狀
                cv2.polylines(display_frame, [poly], True, (0, 255, 0), 2)
                
                # B. 從不規則輪廓中，萃取靶紙的 "12個角落"
                bbox = cv2.boundingRect(contour_pts.astype(np.float32)) # 取得該標籤的邊界框 x,y,w,h
                src_points = extract_12_points(contour_pts, bbox)
                
                # C. 計算投影視角 (Homography)，因為有 12 點 (超定方程)，使用 0 或 cv2.RANSAC
                M, _ = cv2.findHomography(src_points, dst_points, 0)
                
                if M is not None:
                    # 使用去背圖進行拉伸對位
                    warped_frame = cv2.warpPerspective(masked_frame, M, (target_w, target_h))
                    
                    # 視覺化提示：畫上我們自動找出的 12 個角 (紅圈)
                    for i, (px, py) in enumerate(src_points):
                        cv2.circle(display_frame, (int(px), int(py)), 4, (0, 0, 255), -1)
                        cv2.putText(display_frame, str(i), (int(px)+5, int(py)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        # 3. 合併顯示兩邊視窗
        h, w = display_frame.shape[:2]
        ratio = target_h / h
        resized_display = cv2.resize(display_frame, (int(w * ratio), target_h))
        combined_view = np.hstack((resized_display, warped_frame))
        
        cv2.imshow(window_name, combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 處理結束。")

if __name__ == "__main__":
    run_pipeline()
