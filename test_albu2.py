import albumentations as A
import numpy as np
try:
    t = A.Compose(
        [A.PiecewiseAffine(scale=(0.03, 0.05), nb_rows=4, nb_cols=4, p=1.0)], 
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']), 
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, label_fields=['keypoint_labels'])
    )
    # Generate 17 points
    pts = [(100+i*10, 100+i*10) for i in range(17)]
    out = t(
        image=np.zeros((640,640,3), dtype=np.uint8), 
        bboxes=[[0.5,0.5,0.1,0.1]], 
        class_labels=[0], 
        keypoints=pts,
        keypoint_labels=list(range(17))
    )
    print('Output keypoints len:', len(out['keypoints']))
except Exception as e:
    print(e)
