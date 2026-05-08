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
    results_project = settings.get_path("result")
    
    # 確保路徑存在
    if not os.path.exists(data_yaml):
        print(f"錯誤: 找不到設定檔 {data_yaml}")
        return

    # 2. 載入預訓練模型 (YOLO11n-seg)
    # 第一次執行會自動下載 .pt 權重
    model = YOLO("yolo11n-seg.pt")
    
    # 3. 開始訓練
    print(f"🚀 [YOLO11-seg] 訓練啟動中...")
    print(f"📍 資料路徑: {data_yaml}")
    print(f"📍 儲存專案: {results_project}")
    
    try:
        model.train(
            data=data_yaml,
            epochs=50,       # 針對 1000 張圖 50 epoach 是一個好的開始
            imgsz=640,
            batch=16,        # 如果顯存不足可以調小
            project=results_project,
            name="train_run",
            device=0,        # 指定使用 GPU (CUDA)
            plots=True
        )
        print(f"\n🎉 訓練成功結束！權重檔案位於 {results_project}/train_run/weights/best.pt")
    except Exception as e:
        print(f"❌ 訓練過程中發生錯誤: {e}")

if __name__ == "__main__":
    train_model()
