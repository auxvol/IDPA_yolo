import albumentations as A
try:
    t = A.Compose(
        [A.OpticalDistortion(p=1.0)], 
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']), 
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, label_fields=['keypoint_labels'])
    )
    out = t(
        image=__import__('numpy').zeros((640,640,3)), 
        bboxes=[[0.5,0.5,0.1,0.1]], 
        class_labels=[0], 
        keypoints=[(100,100),(999,999)],
        keypoint_labels=[0, 1]
    )
    print('Output keypoints len:', len(out['keypoints']))
    print('Output keypoint_labels:', out['keypoint_labels'])
except Exception as e:
    print(e)
