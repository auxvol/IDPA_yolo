# IDPA 靶紙偵測與追蹤系統

基於 **YOLOv11** 的 IDPA（國際防禦手槍協會）靶紙即時偵測、多目標追蹤與幾何校正管線。
系統能從實彈射擊影片中自動辨識標準 IDPA 靶紙輪廓，透過 Homography + TPS（薄板樣條）雙重變形，將任意視角的靶紙校正為標準正面影像，支援同時追蹤多張靶紙。

---

## 功能概覽

| 功能 | 說明 |
|------|------|
| 🎯 **靶紙語意分割** | YOLO11-Seg 即時偵測靶紙遮罩輪廓 |
| 📐 **17 點姿勢估計** | YOLO11-Pose 直接偵測靶紙 17 個關鍵點 |
| 🔄 **透視校正** | Homography 消除 3D 傾斜，還原正面視角 |
| 🧲 **TPS 精修** | 薄板樣條插值修正殘餘非線性形變 |
| 👥 **多目標追蹤** | 校準期 → 主動追蹤期，獨立追蹤多張靶紙 |
| 📊 **Grid 儀表板** | 即時顯示原始畫面 + 各靶紙變形結果 |
| 🛡️ **品質防護** | Alpha 透明空洞檢測 + 面積回朔防護機制 |

---

## 環境需求

- **作業系統**：Windows
- **Python**：3.12.x
- **GPU**：NVIDIA GPU + CUDA 12.8（推薦）
- **套件管理**：[uv](https://docs.astral.sh/uv/)

### 核心依賴

| 套件 | 用途 |
|------|------|
| `opencv-contrib-python` ≥4.10 | 電腦視覺（含 TPS 變換器） |
| `ultralytics` ≥8.4 | YOLOv11 訓練與推論 |
| `torch` ≥2.11 + CUDA 12.8 | 深度學習框架 |
| `scipy` ≥1.13 | 匈牙利演算法（特徵點匹配） |
| `albumentations` ≥1.4 | 訓練資料增強（鏡頭畸變） |
| `numpy`, `pillow`, `matplotlib` | 數值計算與視覺化 |

---

## 快速開始

### 1. 環境安裝（一鍵腳本）

```bat
setup_env.bat
```

此腳本會自動完成：
1. 安裝 `uv` 套件管理工具（若未安裝）
2. 建立 Python 3.12 虛擬環境 (`.venv/`)
3. 安裝所有依賴套件（含 PyTorch CUDA 12.8）
4. 驗證關鍵套件安裝狀態

### 2. 啟動虛擬環境

```bat
.venv\Scripts\activate
```

### 3. 設定專案路徑（選用）

編輯 `config/.env`：

```env
# 留空則自動偵測專案根目錄（推薦）
PROJECT_ROOT=
```

> 💡 一般情況下不需要修改，系統會根據 `config.py` 的位置自動偵測根目錄。

---

## 執行方式

### 1. 單一視窗即時動態計分 (YOLO-Pose)

```bash
python src/main_pose_scoring.py
```

這支主程式整合了 **YOLO-Pose 標靶追蹤** 與 **動態彈孔計分**。
- 同時支援多張靶紙的獨立追蹤與變形校正
- 結合前/後景差分運算，找出靶紙上的新增彈孔
- 根據 IDPA 幾何定義 (Down 0, Down 1, Down 3) 動態顯示計分結果與命中率儀表板
- 按 `Q` 或 `ESC` 退出

### 2. 演算法對比測試 (Warped vs Raw)

```bash
python src/main_compare_scoring.py
```

這支程式用於對比「形變校正後 (Warped)」與「直接在原畫面 (Raw)」進行彈孔偵測的差異。
- 畫面上半部顯示 Raw 原始視角與彈孔位置
- 畫面下半部顯示 Warped 透過 TPS 校正成正面的視角與彈孔位置
- 用於驗證 TPS 形變校正帶來的計分準確度提升
- 按 `Q` 或 `ESC` 退出

---

## 訓練模型

### 資料準備

1. 將 Unity Perception 匯出的 SOLO 資料集放入 `data/unity_data/solo_*/`
2. 執行轉換腳本：

```bash
# 轉換為 YOLO-Seg 格式
python data/transfer_dataset.py

# 轉換為 YOLO-Pose 格式（4 點版）
python data/transfer_pose_dataset.py

# 分割訓練/驗證集（8:2）
python data/split_data.py
```

3. 驗證標註正確性：

```bash
python data/draw_yolo_boxes.py        # 檢查 BBox 標註
python data/draw_yoloseg_poly.py      # 檢查 Seg 多邊形標註
python data/draw_yolo_pose.py         # 檢查 Pose 12 點標註
python data/draw_yolo_pose4.py        # 檢查 Pose 4 點標註
```

### 執行訓練

```bash
# Seg 語意分割模型
python src/training/train_yolo_seg.py

# Pose 姿勢估計模型（17 關鍵點）
python src/training/train_yolo_pose.py

# BBox 物件偵測模型
python src/training/train_yolo_bbox.py
```

訓練結果（權重、圖表、日誌）將儲存於 `result/` 目錄。

---

## 路徑管理規範

本專案 **禁止硬編碼絕對路徑**。所有路徑皆透過統一的設定模組管理：

```python
from config.config import settings

# 使用相對路徑取得絕對路徑
model_path = settings.get_path("result", "train_seg", "run6", "weights", "best.pt")
video_path = settings.get_path("data", "靶場", "G7_S01.mp4")
```

詳細開發規範請見 [agent_rule.md](agent_rule.md)。

---

## 演算法流程

```text
影片輸入
  │
  ├─ YOLO-Pose 目標追蹤與幾何校正 ─────────────┐
  │   1. 17 關鍵點直出                        │
  │   2. 外框 12 點 Homography 初步對齊       │
  │   3. 全 17 點 TPS 薄板樣條精修            │
  │   4. 輸出標準化、完美正面的「靶紙影像」   │
  │                                           │
  ├─ 動態彈孔偵測與計分 ───────────────────────┤
  │   1. 前後景差分運算 (找出新增彈孔)        │
  │   2. 輪廓提取與面積過濾                   │
  │   3. 對照 IDPA 標準靶紙幾何區域           │
  │   4. 判定分數 (Down 0, Down 1, Down 3)    │
  │                                           │
  └───────────────── 輸出計分儀表板 ◄─────────┘
```

### 兩階段追蹤機制

1. **Phase 1 — 校準期**（前 3 秒）：統計各靶紙遮罩面積中位數，建立基準
2. **Phase 2 — 主動追蹤期**：鎖定全歷史最小面積遮罩，帶有 2~5% 防呆閾值與 Alpha 空洞回朔防護

---

## 授權

本專案為學術研究用途。
