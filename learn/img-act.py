import cv2
import numpy as np

img = cv2.imread('/Volumes/Untitled/guo-qiang/kaggle/statefarm/rawdata/train/c0/img_34.jpg')
resized = cv2.resize(img, (224, 224))
print "original size",resized.shape
resized = resized.transpose((2,0,1))
print resized.shape
resized = np.expand_dims(resized, axis=0)
print img.shape
print img.size
print resized.shape

cv2.imshow("hh",img)

cv2.waitKey(0)