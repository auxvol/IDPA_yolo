import time, albumentations as A, numpy as np
img = np.zeros((640, 640, 3), dtype=np.uint8)
t1 = A.PiecewiseAffine(scale=(0.01, 0.02), nb_rows=5, nb_cols=5, p=1.0)
start = time.time()
[t1(image=img, keypoints=[(100,100)]*17) for _ in range(100)]
print('5x5 Time:', time.time()-start)

t2 = A.PiecewiseAffine(scale=(0.01, 0.02), nb_rows=3, nb_cols=3, p=1.0)
start = time.time()
[t2(image=img, keypoints=[(100,100)]*17) for _ in range(100)]
print('3x3 Time:', time.time()-start)
