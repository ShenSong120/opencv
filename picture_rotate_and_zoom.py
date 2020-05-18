import cv2
import numpy as np


img = cv2.imread("picture/target_backup.jpg", 1)
imgInformation = img.shape
height = imgInformation[0]
width = imgInformation[1]
mat = cv2.getRoationMatrix2D((height*0.5, width*0.5), 45, 0.5)
# 这里有三个参数 分别是中心位置，旋转角度，缩放程度
dst = cv2.warpAffine(img, mat, (height, width))
cv2.imshow("dst", dst)
cv2.waitKey(0)
