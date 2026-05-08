# 專案開發規範與環境說明 (AI Assistant 專用)

## 開發環境
- **作業系統**: Windows
- **包管理工具**: `uv`
- **虛擬環境**: `./.venv/` (Python 3.12.13)
- **相依套件**: 詳見 `pyproject.toml`（核心：`opencv-contrib-python`, `numpy`, `scipy`, `Pillow`, `ultralytics`, `torch`, `albumentations`, `python-dotenv`）。
- **環境建置**: 執行 `setup_env.bat` 可一鍵完成 uv 安裝、虛擬環境建立、套件安裝。

## 開發規範
1. **路徑與設定解耦**:
    - **禁止**在程式碼中硬編碼任何實體絕對路徑。
    - **設定方式**: `config/.env` 僅需設定 `PROJECT_ROOT`（選填，若留空則自動偵測專案根目錄）。
    - **調用方式**: 程式應匯入 `config.config.settings` 物件，並使用其 `get_path("相對路徑")` 方法。
    - **範例**: `settings.get_path("data/solo_32")` 會自動將相對路徑轉為絕對路徑。
    - **優點**: 減少 `.env` 維護成本，程式碼僅需關心相對路徑。
2. **目錄結構**:
    - `config/`: 存放配置管理模組與 `.env`。
    - `data/`: 存放原始資料集 (SOLO), 標註後的資料集 (YOLO/SEG) 與各種檢查、轉換工具。
    - `src/`: 存放核心邏輯。

## 維護規則
- **環境變動同步**: 每次對專案環境（包含目錄結構、路徑引用方式、新增全域依賴等）做出變動時，**必須同時更新此 `agent_rule.md` 的內容**，以確保下一個 Agent 能精確了解目前的開發環境。

---
*Created by Antigravity AI Assistant.*
