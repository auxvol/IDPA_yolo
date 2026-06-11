import albumentations as A, numpy as np, cv2
img=np.zeros((640,640,3),dtype=np.uint8)
cv2.rectangle(img,(100,100),(540,540),(255,255,255),2)
t=A.ElasticTransform(alpha=1500, sigma=25, p=1.0)
out=t(image=img)['image']
cv2.imwrite('test_warp3.jpg', out)
