import os
from dotenv import load_dotenv

class Settings:
    """
    配置管理類別，讀取 .env 或自動偵測專案根目錄。
    """
    def __init__(self):
        # 1. 自動偵測根目錄 (此檔案位於根目錄下的 config 資料夾)
        self.DEFAULT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._env_path = os.path.join(self.DEFAULT_BASE_DIR, "config", ".env")
        
        # 2. 載入 .env 檔案
        if os.path.exists(self._env_path):
            load_dotenv(self._env_path)
            
        # 3. 取得專案根目錄 (優先讀取環境變數，若無則使用自動偵測)
        env_root = os.getenv("PROJECT_ROOT")
        if env_root and env_root.strip():
            self.BASE_DIR = os.path.normpath(env_root)
        else:
            self.BASE_DIR = self.DEFAULT_BASE_DIR

    def get_path(self, *rel_paths):
        """傳入相對路徑（如 'data/solo_32'）並與根目錄結合，回傳絕對路徑"""
        return os.path.normpath(os.path.join(self.BASE_DIR, *rel_paths))

# 單例實例供其他模組直接導入
settings = Settings()
