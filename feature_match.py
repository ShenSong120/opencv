# _*_ coding: utf-8 _*_
"""
特征匹配
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


img1 = cv2.imread('picture/img1.jpg', 0)
img2 = cv2.imread('picture/img2.jpg', 0)

# 初始化SIFT检测器
sift = cv2.xfeatures2d.SIFT_create()

# 用SIFT找到关键点和描述
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN参数
FLANN_INDEX_KOTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KOTREE, trees=5)
search_params = dict(checks=50)  # 或者传递空字典

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# 只需要绘制好的匹配, 所以创建一个掩膜
matchesMask = [[0, 0] for i in range(len(matches))]

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=0)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

plt.imshow(img3)
plt.show()
