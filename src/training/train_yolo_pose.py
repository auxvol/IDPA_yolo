import os
import sys
from ultralytics import YOLO

# 取得專案根目錄以便導入 config
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from config.config import settings

# --- 修正 YOLO Pose 原生資料增強，加入長寬比(Aspect Ratio)變化 ---
import math
import random
import cv2
import numpy as np
from ultralytics.data.augment import RandomPerspective

def custom_compute_affine_matrix(self, img, size):
    """Monkey-patched function to inject aspect ratio variation natively in YOLO"""
    # Center
    C = np.eye(3, dtype=np.float32)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3, dtype=np.float32)
    P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3, dtype=np.float32)
    a = random.uniform(-self.degrees, self.degrees)
    if isinstance(self.scale, (tuple, list)):
        s = random.uniform(self.scale[0], self.scale[1])
    else:
        s = random.uniform(1 - self.scale, 1 + self.scale)
        
    # [新增] 獨立的長寬比隨機變化 (Aspect Ratio Jitter)
    # 這裡我們讓 X 軸與 Y 軸縮放比例有差異，範圍為 0.5 到 2.0 以加強拉伸效果
    aspect_ratio_jitter = random.uniform(0.8, 1.4)
    sx = s * math.sqrt(aspect_ratio_jitter)
    sy = s / math.sqrt(aspect_ratio_jitter)
    
    # 產生旋轉矩陣 (不帶縮放)
    rot_mat = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=1.0)
    # 將非等比例縮放應用到旋轉矩陣
    rot_mat[0, :] *= sx
    rot_mat[1, :] *= sy
    R[:2] = rot_mat

    # Shear
    S = np.eye(3, dtype=np.float32)
    S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3, dtype=np.float32)
    T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * size[0]  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * size[1]  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    return M, s

# 覆蓋原函數，讓 YOLO 原生的空間增強自帶長寬比變化
RandomPerspective._compute_affine_matrix = custom_compute_affine_matrix
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# --- 修正 YOLO Pose 整合 Albumentations 缺乏 Keypoints 支援的問題 ---
from ultralytics.data.augment import Albumentations as YoloAlbumentations
import albumentations as A

_original_albu_init = YoloAlbumentations.__init__

def custom_albu_init(self, p=1.0, transforms=None):
    _original_albu_init(self, p, transforms)
    if self.contains_spatial and self.transform is not None:
        try:
            T = self.transform.transforms
            self.transform = A.Compose(
                T,
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels", "bbox_ids"]),
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False, label_fields=["keypoint_labels"])
            )
        except Exception as e:
            print("⚠️ 自訂 Albumentations 初始化失敗:", e)

def custom_albu_call(self, labels):
    if self.transform is None or random.random() > self.p:
        return labels

    im = labels["img"]
    if im.shape[2] != 3:
        return labels

    if self.contains_spatial:
        cls = labels["cls"]
        if len(cls):
            labels["instances"].convert_bbox("xywh")
            labels["instances"].normalize(*im.shape[:2][::-1])
            bboxes = labels["instances"].bboxes
            bbox_ids = list(range(len(bboxes)))
            
            kpts = labels["instances"].keypoints if hasattr(labels["instances"], "keypoints") else None
            has_kpts = kpts is not None and len(kpts) > 0
            
            flat_kpts = []
            kpt_labels = []
            if has_kpts:
                N, K, _ = kpts.shape
                for i in range(N):
                    for j in range(K):
                        # ⚠️ 修正：Albumentations 的 format="xy" 要求「絕對像素座標」
                        # 但此時 YOLO 的 keypoints 已被 normalize，所以必須先還原成像素座標
                        abs_x = float(kpts[i, j, 0]) * im.shape[1]
                        abs_y = float(kpts[i, j, 1]) * im.shape[0]
                        flat_kpts.append((abs_x, abs_y))
                        kpt_labels.append(i * K + j)  # 紀錄 keypoint 的原始索引
            
            if has_kpts:
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls, bbox_ids=bbox_ids, keypoints=flat_kpts, keypoint_labels=kpt_labels)
            else:
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls, bbox_ids=bbox_ids)

            im = new["image"]
            
            if len(new["class_labels"]) > 0:
                labels["img"] = new["image"]
                labels["cls"] = np.array(new["class_labels"]).reshape(-1, 1)
                labels["instances"].update(bboxes=np.array(new["bboxes"], dtype=np.float32))
                
                if has_kpts:
                    new_kpts = np.zeros((len(new["bbox_ids"]), K, 3), dtype=np.float32)
                    returned_kpts_dict = {int(lbl): (kp[0], kp[1]) for kp, lbl in zip(new["keypoints"], new["keypoint_labels"])}
                    
                    surviving_bbox_ids = new["bbox_ids"]
                    for i, orig_idx in enumerate(surviving_bbox_ids):
                        orig_idx = int(orig_idx)
                        for j in range(K):
                            kpt_lbl = orig_idx * K + j
                            if kpt_lbl in returned_kpts_dict:
                                abs_x, abs_y = returned_kpts_dict[kpt_lbl]
                                # ⚠️ 修正：Albumentations 處理完後是絕對像素座標，必須再正規化回 0~1 給 YOLO
                                norm_x = abs_x / im.shape[1]
                                norm_y = abs_y / im.shape[0]
                                new_kpts[i, j, 0] = norm_x
                                new_kpts[i, j, 1] = norm_y
                                new_kpts[i, j, 2] = kpts[orig_idx, j, 2]
                            
                    labels["instances"].keypoints = new_kpts
            else:
                # 所有目標都被裁切掉了
                labels["img"] = new["image"]
                labels["cls"] = np.zeros((0, 1), dtype=np.float32)
                labels["instances"].update(bboxes=np.zeros((0, 4), dtype=np.float32))
                if has_kpts:
                    labels["instances"].keypoints = np.zeros((0, K, 3), dtype=np.float32)
        return labels
    else:
        labels["img"] = self.transform(image=labels["img"])["image"]
        return labels

YoloAlbumentations.__init__ = custom_albu_init
YoloAlbumentations.__call__ = custom_albu_call
# ---------------------------------------------------------------

def train_model():
    # 1. 取得絕對路徑
    data_yaml = settings.get_path("data", "yolo_pose_dataset_merged_occluded", "data.yaml")
    results_project = settings.get_path("result", "train_pose_17kpt_merged")
    
    # 確保路徑存在
    if not os.path.exists(data_yaml):
        print(f"錯誤: 找不到設定檔 {data_yaml}")
        return

    # 確保輸出目錄存在
    if not os.path.exists(results_project):
        os.makedirs(results_project, exist_ok=True)

    # 2. 載入預訓練模型 (使用 yolo11n-pose)
    model = YOLO("yolo11n-pose.pt")
    
    # 3. 開始訓練
    print(f"🚀 [YOLO11-pose] 訓練啟動中...")
    print(f"📍 資料路徑: {data_yaml}")
    print(f"📍 儲存專案: {results_project}")
    
    # --- Albumentations 空間與像素增強 (需 pip install albumentations) ---
    try:
        import albumentations as A
        custom_augmentations = [
            # === 破壞剛性幾何先驗的非線性扭曲 (保證 100% 點位保留) ===
            # 因為座標正規化 Bug 已修復，現在我們可以使用速度快 10 倍的 ElasticTransform (彈性形變)
            # 既能製造不規則凹折，又不會弄丟 17 個點。
            A.ElasticTransform(alpha=1500, sigma=30, p=0.4),
            
            # === 像素雜訊 ===
            A.GaussNoise(std_range=(0.1, 0.3), p=0.2),
            A.Blur(blur_limit=3, p=0.3),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.CLAHE(p=0.3),
        ]
        print("✅ Albumentations 破壞剛性先驗增強 (PiecewiseAffine) 已載入")
    except ImportError:
        custom_augmentations = None
        print("⚠️ 未安裝 albumentations，跳過增強")

    try:
        # 加入各式資料增強 (Data Augmentations) 以增加對不同環境與拍攝角度的魯棒性
        train_kwargs = dict(
            data=data_yaml,
            epochs=30,       # 訓練回合數
            imgsz=640,
            batch=32,        
            project=results_project,
            name="run",      # 結果會存於 result/train_pose_17kpt/run
            device=0,        # 使用 GPU
            plots=True,

            # === YOLO 原生空間增強 (模擬 3D 視角變化，純數學矩陣轉換，絕對不會掉點) ===
            degrees=30.0,        # 隨機旋轉 ±30度 (模擬攝影機歪斜與手持晃動)
            translate=0.15,      # 隨機平移 ±15% (模擬不同構圖)
            scale=0.5,           # 隨機縮放 ±50% (模擬遠近距離)
            shear=20.0,          # 錯切變形 ±20度 (大幅度扭曲為平行四邊形)
            perspective=0.0005,  # 透視變形加強 (強烈的 3D 俯仰/側視扭曲效果)

            # === 色彩增強 ===
            hsv_h=0.015,         # 色調變化
            hsv_s=0.7,           # 飽和度變化
            hsv_v=0.4,           # 明暗度變化 (對抗曝光不足/過曝)

            # === 拼接與混合增強 ===
            mosaic=1.0,          # 馬賽克拼圖增強
            mixup=0.1,           # 混圖增強

            # === 遮擋增強 ===
            erasing=0.5,         # 隨機遮罩 (Cutout): 提升抗遮擋魯棒性

            # === 翻轉控制 ===
            fliplr=0.0,          # ⚠️ 強制關閉左右翻轉 (靶紙不對稱，避免混淆關鍵點順序)
            flipud=0.0,          # ⚠️ 強制關閉上下翻轉
        )

        # 如果 Albumentations 可用，加入自訂鏡頭畸變增強
        if custom_augmentations is not None:
            train_kwargs["augmentations"] = custom_augmentations

        model.train(**train_kwargs)
        print(f"\n🎉 訓練成功結束！權重檔案位於 {results_project}\\run\\weights\\best.pt")
    except Exception as e:
        print(f"❌ 訓練過程中發生錯誤: {e}")

if __name__ == "__main__":
    train_model()
