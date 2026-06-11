import time, albumentations as A, numpy as np
img = np.zeros((640, 640, 3), dtype=np.uint8)
t1 = A.Compose(
    [A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0)],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']),
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, label_fields=['keypoint_labels'])
)
start = time.time()
pts = [(100+i*5, 100+i*5) for i in range(17)]
for _ in range(100):
    out = t1(image=img, bboxes=[[0.5,0.5,0.1,0.1]], class_labels=[0], keypoints=pts, keypoint_labels=list(range(17)))
print('Elastic Time:', time.time()-start)
print('Retained points:', len(out['keypoints']))
