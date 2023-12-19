import numpy as np
import cv2 as cv
import json
def save_2_jason(arr):
        data = {}
        cnt = 0
        for i in arr:
            data['KeyPoint_%d'%cnt] = []
            data['KeyPoint_%d'%cnt].append({'x': i.pt[0]})
            data['KeyPoint_%d'%cnt].append({'y': i.pt[1]})
            data['KeyPoint_%d'%cnt].append({'size': i.size})
            cnt+=1
        with open('points1/data.txt', 'w') as outfile:
            json.dump(data, outfile)
from matplotlib import pyplot as plt
from numpy.lib.twodim_base import mask_indices
def how_match(img_1, img_2):
    orb = cv.ORB_create()
    kp1 = orb.detect(img_1)  # 用cv库里自带的ORB特征提取算法（查找边缘和角点 灰度值骤变的点）快
    # img2=img_1.copy()
    # img2 = cv.drawKeypoints(img_1, kp1, img2, color=(0, 0, 255),flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.namedWindow("name", 0)
    # cv.resizeWindow("name", 780, 1040)
    # cv.imshow("name", img2)
    # cv.imwrite('save.png', img2)
    # cv.waitKey(0)
    kp2 = orb.detect(img_2)

    kp1, des1 = orb.compute(img_1, kp1)
    kp2, des2 = orb.compute(img_2, kp2)

    bf = cv.BFMatcher(cv.NORM_HAMMING)  # 用cv库里面自带的暴力匹配包 但效果一般 肯定要用更高效算法

    matches = bf.match(des1, des2)
    min_distance = matches[0].distance
    max_distance = matches[0].distance

    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance

    print("Max dist:", max_distance)
    print("Min dist:", min_distance)
    good_match = []
    good_match = sorted(matches, key=lambda x: x.distance)
    img3 = cv.drawMatches(img_1, kp1, img_2, kp2, good_match[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img3 = cv.resize(img3, None, fx=0.2, fy=0.3)  # 改变图像大小
    cv.imshow("Matches", img3)

    cv.waitKey(0)
    cv.destroyAllWindows()
    good_match = good_match[0:10]
    distance=0
    for match in good_match:
        distance+=match.distance
    distance=distance/len(good_match)
    #
    # for x in matches:
    #     if x.distance <= max(2 * min_distance, 30.0):
    #         good_match.append(x)
    return distance


def find_feature_matches(img_1, img_2):
    orb = cv.ORB_create()
    kp1 = orb.detect(img_1)  # 用cv库里自带的ORB特征提取算法（查找边缘和角点 灰度值骤变的点）快
    # img2=img_1.copy()
    # img2 = cv.drawKeypoints(img_1, kp1, img2, color=(0, 0, 255),flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.namedWindow("name", 0)
    # cv.resizeWindow("name", 780, 1040)
    # cv.imshow("name", img2)
    # cv.imwrite('save.png', img2)
    # cv.waitKey(0)
    kp2 = orb.detect(img_2)

    kp1, des1 = orb.compute(img_1, kp1)
    kp2, des2 = orb.compute(img_2, kp2)

    bf = cv.BFMatcher(cv.NORM_HAMMING)  # 用cv库里面自带的暴力匹配包 但效果一般 肯定要用更高效算法

    matches = bf.match(des1, des2)

    save_2_jason(kp1)
    np.savez('points1/pointsdata', des1)
    min_distance = matches[0].distance
    max_distance = matches[0].distance

    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance

    print("Max dist:", max_distance)
    print("Min dist:", min_distance)
    good_match = []
    good_match = sorted(matches, key=lambda x: x.distance)
    good_match = good_match[0:20]
    #
    # for x in matches:
    #     if x.distance <= max(2 * min_distance, 30.0):
    #         good_match.append(x)
    return kp1, kp2, good_match


def poes_estimation_2d2d(keypoint_1, keypoint_2, matches):
    k = [[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]]
    k = np.array(k)
    print("相机内参：", k)

    # print(keypoint_1)
    # print("描述子：", matches)

    good = []
    pts2 = []
    pts1 = []
    # for i in range(int(len(matches))):
    for i in range(int(len(matches)/2)):
        pts1.append(keypoint_1[matches[i].queryIdx].pt)
        pts2.append(keypoint_2[matches[i].trainIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    # cv自带的库，能够计算基础矩阵 采用8点法
    f, mask = cv.findFundamentalMat(points1=pts1, points2=pts2, method=cv.FM_8POINT)
    #print("基础矩阵：", f)f

    # cv自带的库，能够计算本质矩阵
    e, mask = cv.findEssentialMat(points1=pts1, points2=pts2, cameraMatrix=k)
    #print("本质矩阵： ", e)

    # cv自带的库，能够计算单应矩阵
    h, mask = cv.findHomography(pts1, pts2, method=cv.RANSAC)
    #print("单应矩阵： ", h)

    # cv自带的库，能够从本质矩阵中恢复旋转信息和平移信息
    retval2, R, t, mask = cv.recoverPose(E=e, points1=pts1, points2=pts2, cameraMatrix=k)
    #print("旋转矩阵R:", R)
    #print("平移矩阵t:", t)
    # print(mask)
    return R, t, h


if __name__ == "__main__":
    # img_1 = cv.imread("/home/ubuntu/users/lirj/wj1e/guominan/VO_task_easyversion_2/dataset/1.png")
    # img_2 = cv.imread("/home/ubuntu/users/lirj/wj1e/guominan/VO_task_easyversion_2/dataset/2.png")
    img_1 =cv.imread("sys.jpg")  # 读取图片
    #img_2 = cv.imread("/home/ubuntu/users/lirj/wj1e/guominan/VO_task_easyversion_2/dataset/0006.jpg")

    # 图像匹配
    keypoint_1, keypoint_2, matches = find_feature_matches(img_1, img_1)
   # print("共计匹配点:", len(matches))

    # 预测位姿
    # R, t = poes_estimation_2d2d(keypoint_1, keypoint_2, matches)
    #poes_estimation_2d2d(keypoint_1, keypoint_2, matches)