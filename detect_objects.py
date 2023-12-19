import time

import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance

def detectpoint(image,x,y):
    # start = time.perf_counter()  # 计时函数
    # cap=cv2.VideoCapture(1)
    # ret,image=cap.read()
    # image = cv2.blur(image, (3, 3))  # 滤波
    image = cv2.medianBlur(image, 3)
    # cv2.imwrite('blur_image.jpg', image)
    # def imgBrightness(img1, c, b):
    #     rows, cols, channels = img1.shape
    #     blank = np.zeros([rows, cols, channels], img1.dtype)
    #     rst = cv2.addWeighted(img1, c, blank, 1 - c, b)
    #     return rst
    #
    # dst = imgBrightness(image, 0.75, 0)
    # dst2 = imgBrightness(image, 2.0, 0)
    # cv2.imwrite('brightness.jpg', dst2)
    # end = time.perf_counter()
    # print("定位用时：", end - start, "秒")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #  #黑色
    #     lower=np.array([0,0,0])
    #     upper=np.array([180,255,46])
    #  #黄色
    #     lower=np.array([26,43,46])
    #     upper=np.array([34,255,255])
    # #绿色
    #    lower=np.array([35,43,46])
    #    upper=np.array([77,255,255])
    #  #橙色
    #     lower=np.array([11,43,46])
    #     upper=np.array([25,255,255])
    #  #蓝色
    #     lower=np.array([100,43,46])
    #     upper=np.array([124,255,255])
    #   #红色
    lower = np.array([0, 50, 245])
    upper = np.array([15, 110, 255])

    mask = cv2.inRange(hsv, lower, upper)
    #    res=cv2.bitwise_and(image,image,mask=mask)
    res = cv2.bitwise_and(image, image, mask=mask)
    cv2.namedWindow("image", 0)
    cv2.resizeWindow("image", 816, 612)
    # cv2.namedWindow("mask", 0)
    # cv2.resizeWindow("mask", 816, 612)
    # cv2.namedWindow("res", 0)
    # cv2.resizeWindow("res", 816, 612)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    # cv2.imshow('mask',mask)
    # cv2.waitKey(0)
    # cv2.imshow('res',res)
    point=np.where(mask==255)
    dis=[]
    for points in point:
        dis.append(np.sqrt((points[0]-y)**2+(points[1]-x)**2))
    inde=dis.index(np.min(dis))
    point_xy1 = [point[1][inde], point[0][inde]]
    point_xy=[int(np.average(point[1])),int(np.average(point[0]))]
    # end = time.perf_counter()
    # print("定位用时：", end - start, "秒")
    # cv2.namedWindow("image", 0)
    # cv2.resizeWindow("image", 816, 612)
    return point_xy1
# cv2.namedWindow("mask", 0)
# cv2.resizeWindow("mask", 816, 612)
# cv2.namedWindow("res", 0)
# cv2.resizeWindow("res", 816, 612)
# cv2.imshow('image', image)
# cv2.imshow('mask',mask)
# cv2.imshow('res',res)
cv2.waitKey(0)
k=cv2.waitKey(5)&0xff
cv2.destroyAllWindows()