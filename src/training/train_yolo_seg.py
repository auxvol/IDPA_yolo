import os
import sys
from ultralytics import YOLO

# 取得專案根目錄以便導入 config
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from config.config import settings

def train_model():
    # 1. 取得絕對路徑
    data_yaml = settings.get_path("data/yoloseg_dataset/data.yaml")
    results_project = settings.get_path("result/train_seg")  # 對結果進行分類，獨立放到 train_seg 目錄下
    
    # 確保路徑存在
    if not os.path.exists(data_yaml):
        print(f"❌ 錯誤: 找不到設定檔 {data_yaml}")
        return

    os.makedirs(results_project, exist_ok=True)

    # 2. 載入預訓練模型 (YOLO11n-seg 語意分割模型)
    model = YOLO("yolo11n-seg.pt")
    
    # 3. 開始訓練
    print(f"🚀 [YOLO11-seg] 語意分割訓練啟動中...")
    print(f"📍 資料路徑: {data_yaml}")
    print(f"📍 儲存專案: {results_project}")
    
    # --- Albumentations 鏡頭畸變增強 (需 pip install albumentations) ---
    try:
        import albumentations as A
        custom_augmentations = [
            # 桶形/枕形鏡頭畸變 (模擬廣角/長焦鏡頭變形)
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.05, p=0.4),
            # 網格畸變 (模擬不均勻的鏡頭畸變場)
            A.GridDistortion(num_steps=15, distort_limit=0.3, p=0.3),
            # 高斯雜訊 (模擬低光源下的感測器雜訊)
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        ]
        print("✅ Albumentations 鏡頭畸變增強已載入")
    except ImportError:
        custom_augmentations = None
        print("⚠️ 未安裝 albumentations，跳過鏡頭畸變增強")

    try:
        train_kwargs = dict(
            data=data_yaml,
            epochs=30,       
            imgsz=640,
            batch=32,        
            project=results_project,
            name="run",      # 結果會存於 result/train_seg/run
            device=0,        
            plots=True,

            # === YOLO 原生空間增強 (模擬 3D 視角變化) ===
            degrees=25.0,        # 隨機旋轉 ±25度 (模擬攝影機歪斜與手持晃動)
            translate=0.15,      # 隨機平移 ±15% (模擬不同構圖)
            scale=0.5,           # 隨機縮放 ±50% (模擬遠近距離)
            shear=10.0,          # 錯切變形 ±10度 (模擬斜視角拍攝)
            perspective=0.0005,  # 透視變形 (模擬不同 3D 拍攝角度，俯仰/側視)

            # === 色彩增強 ===
            hsv_h=0.015,         # 色調變化
            hsv_s=0.7,           # 飽和度變化
            hsv_v=0.4,           # 明暗度變化 (對抗曝光不足/過曝)

            # === 拼接與混合增強 ===
            mosaic=1.0,          # 馬賽克拼圖增強
            mixup=0.1,           # 混圖增強

            # === 遮擋增強 ===
            erasing=0.5,         # 隨機遮罩 (Cutout): 提升抗遮擋魯棒性
            copy_paste=0.3,      # 複製貼上 (模擬真實物件遮擋)

            # === 翻轉控制 ===
            fliplr=0.0,          # ⚠️ 強制關閉左右翻轉 (靶紙不對稱)
            flipud=0.0,          # ⚠️ 強制關閉上下翻轉
        )

        # 如果 Albumentations 可用，加入自訂鏡頭畸變增強
        if custom_augmentations is not None:
            train_kwargs["augmentations"] = custom_augmentations

        model.train(**train_kwargs)
        print(f"\n🎉 訓練成功結束！權重檔案位於 {results_project}/run/weights/best.pt")
    except Exception as e:
        print(f"❌ 訓練過程中發生錯誤: {e}")

if __name__ == "__main__":
    train_model()
