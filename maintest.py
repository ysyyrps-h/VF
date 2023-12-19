import time

import cv2

import Ellipse_position
import VO_position
import VO_position_auto
import first_decoding
import fourth_decoding
import numpy as np
import position_fun
import scipy.io
import second_decoding
import third_decoding
import try_decoding
import match
import detect_objects
import matchpoints

def ini_process(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 变成灰度图
    blur_image = cv2.blur(gray, (10, 10))  # 滤波
    # cv2.imwrite('blur_image.jpg', blur_image)
    # ret, binary = cv2.threshold(blur_image, 0, 255, cv2.THRESH_OTSU)  # 二值化，阈值要设的小一点
    ret, binary = cv2.threshold(blur_image, 220, 255, cv2.THRESH_BINARY)  # 二值化，阈值要设的小一点
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 找轮廓
    contours1 = []
    for ind, cont in enumerate(contours):
        if len(cont) > 500:
            contours1.append(cont)
    for ind, cont in enumerate(contours1):
        elps = cv2.fitEllipse(cont)
        cv2.ellipse(image, elps, (0, 0, 255))
    contours = contours1
    # scipy.io.savemat('resultroll.mat', mdict={'contours': contours})
    # print(len(contours))
    # cv2.drawContours(image, contours, -1, (0, 0, 255), 4)  # 画出轮廓，可以省略
    # # cv2.imwrite('image.jpg', image)
    # cv2.namedWindow("img", 0)
    # cv2.resizeWindow("img", 816, 612)
    # cv2.imshow("img", image)
    # cv2.waitKey(0)
    c = []
    for i in range(len(contours)):
        # count = len(contours[i])
        # if count < 6:
        #     continue
        # box = cv2.fitEllipse(contours[i])
        # #画出拟合的椭圆
        # cv2.ellipse(binary, box, (0, 0, 255), 1)
        M = cv2.moments(contours[i])
        con = np.array(contours[i])
        a = con.reshape(len(con), len(con[0][0])).T
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        b = np.array([[cX], [cY]])
        c.append(np.hstack((b, a)))
        print(cX, cY)
    # cv2.namedWindow("name", 0)
    # cv2.resizeWindow("name", 780, 1040)
    # cv2.imshow("name", binary)
    #
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]  # ROI区域
    return [bounding_boxes, c]


def adap_threshold(bounding_boxes, images):
    threshold_list = []
    for bbox in range(len(bounding_boxes)):
        [x, y, w, h] = bounding_boxes[bbox]
        img_roi = images[y:(y + h), x:(x + w)]
        gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)  # 变成灰度图
        temp = []
        if w < 50:
            continue
        if flag1 == 1:
            for j in range(h):
                temp.append(gray_roi[j][int(w / 2)])
        if flag1 == 0:
            for j in range(w):
                temp.append(gray_roi[int(h / 2)][j])
        threshold_list.append(np.mean(temp))
    th = np.mean(threshold_list)
    return th


def delet_counter(counters):
    lines = []
    for counter in counters:
        step = int(np.floor(np.size(counter) / 40))
        line = 0
        for i in range(20):
            # if (i == 18):
            #     point1 = counter[:,20 * 18-1]
            #     point2 = counter[:,20 * 19-1]
            #     point3 = counter[:,0]
            if (i == 19):
                point1 = counter[:,step * 19-1]
                point2 = counter[:,0]
                point3 = counter[:,step]
            else:
                point1 = counter[:,step * i-1]
                point2 = counter[:,(step * (i + 1)-1)]
                point3 = counter[:,step * (i + 2)-1]
            # k1 = (point2[1] - point1[1]) / (point2[0] - point1[0])
            # k2 = (point3[1] - point1[1]) / (point3[0] - point1[0])
            # k3 = (point3[1] - point2[1]) / (point3[0] - point2[0])
            if ((abs(point1[0] - point2[0]) == 0)|(abs(point1[1] - point2[1]) == 0))&((abs(point2[0] - point3[0]) == 0)|(abs(point2[1] - point3[1]) == 0)):
                line = 1
        lines.append(line)
    return lines


def main(path1, path2):
    start = time.perf_counter()  # 计时函数
    # ################### 变量定义 #######################
    id_list = []  # 最终识别的LED列表
    center_list = []  # 最终识别的LED中心列表    `
    contours_list = []  # 最终识别的轮廓列表
    flag1 = 0  # 手机横竖标志位 1表示横屏拍摄，0表·    示竖屏拍摄
    flag2 = 0  # 倾斜角度标志位 1表示180° 0表示0°
    flag3 = 0  # 标识两帧是否找到相同LED
    # ################### 输入图一 #######################
    # path1 = "20230213164004"
    # path2 = "20230213164006"
    img = cv2.imread("image/VLP1/" + path1 + ".jpg")  # 读取图片
    img_match = cv2.imread("image/VLP1/" + path1 + "C.jpg")
    row = len(img)
    col = len(img[0])
    if col > row:
        flag1 = 1
    else:
        flag1 = 0
    # ################### 图像处理 #######################
    [led_boxes, contours] = ini_process(img_match)  # 找到ROI区域
    # lines = delet_counter(contours)
    # count=0
    # for i in range(np.size(lines)):
    #     if (lines[i]==1):
    #         led_boxes.pop(i - count)
    #         #np.delete(led_boxes,i-count)
    #         contours.pop(i-count)
    #         #np.delete(contours, [i-count])
    #         count=count+1
    # [led_boxes, a] = second_process(img)  # 找到ROI区域
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 4)  # 画出轮廓，可以省略
    # # cv2.imwrite('image.jpg', image)
    # cv2.namedWindow("img", 0)
    # cv2.resizeWindow("img", 816, 612)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    err_corr_list = [0] * len(led_boxes)  # 没有成功识别的LED列表
    # ################### 自适应阈值 #######################
    threshold = adap_threshold(led_boxes, img)  # 自适应二值化阈值计算
    end = time.perf_counter()
    print("筛选：", end - start, "秒")
    # ################### 解码 #######################
    try_decoding.try_decode(img, led_boxes, threshold, err_corr_list, id_list, center_list, contours_list, flag1,
                            contours)  # 尝试解码,找到旋转方向
    # [position_x, position_y] = VO_position.Ellipse(id_list, contours_list, flag1)
    # ################### 变量定义 #######################
    id_list1 = []  # 最终识别的LED列表
    center_list1 = []  # 最终识别的LED中心列表
    contours_list1 = []  # 最终识别的轮廓列表
    # ################### 输入图二 #######################
    img1 = cv2.imread("image/VLP1/" + path2 + ".jpg")  # 读取图片
    img1_match = cv2.imread("image/VLP1/" + path2 + "C.jpg")
    row = len(img1)
    col = len(img1[0])
    if col > row:
        flag1 = 1
    else:
        flag1 = 0
    # ################### 图像处理 #######################
    [led_boxes1, contours1] = ini_process(img1_match)  # 找到ROI区域
    # [led_boxes1, a] = second_process(img1)  # 找到ROI区域
    # lines=[]
    # lines = delet_counter(contours1)
    # count=0
    # for i in range(np.size(lines)):
    #     if (lines[i]==1):
    #         led_boxes1.pop(i - count)
    #         #np.delete(led_boxes,i-count)
    #         contours1.pop(i-count)
    #         #np.delete(contours, [i-count])
    #     count=count+1
    err_corr_list1 = [0] * len(led_boxes1)  # 没有成功识别的LED列表
    # ################### 自适应阈值 #######################
    threshold1 = adap_threshold(led_boxes1, img1)  # 自适应二值化阈值计算
    # ################### 解码 #######################
    try_decoding.try_decode(img1, led_boxes1, threshold1, err_corr_list1, id_list1, center_list1, contours_list1, flag1,
                            contours1)  # 尝试解码,找到旋转方向
    # [position_x, position_y] = VO_position.Ellipse(id_list1, contours_list1, flag1)
    indexid1 = 0
    indexid2 = 0


    for i in range(len(id_list)):
        if flag3 == 1:
            break
        for j in range(len(id_list1)):
            if id_list[i] == id_list1[j]:
                indexid1 = i
                indexid2 = j
                flag3 = flag3 + 1
                break
    if flag3 == 1:
        id_list = id_list[indexid1]
        contours_list = contours_list[indexid1]
        center_list = center_list[indexid1]
        id_list1 = id_list1[indexid2]
        contours_list1 = contours_list1[indexid2]
        center_list1 = center_list1[indexid2]
    # if len(id_list) == 0:
    #     flag2 = 1  # 倾斜角度标志位 1表示180° 0表示0°
    #     first_decoding.first(img, led_boxes, threshold,  # 第一次解码，要放到if语句里面，try和first都相当于第一次解码，正确方向时只需要解码一次
    #                          err_corr_list, id_list, center_list, contours_list, contours, flag1, flag2)
    # if len(id_list) < 4:
    #     second_decoding.second(img, led_boxes, threshold,
    #                            err_corr_list, id_list, center_list, contours_list, contours, flag1, flag2)  # 第二次解码，只解码第一次没有识别成功的
    #     if len(id_list) < 4:
    #         third_decoding.third(img, led_boxes, threshold,
    #                              err_corr_list, id_list, center_list, contours_list, contours, flag1, flag2)  # 第三次解码，只解码第二次没有识别成功的
    #         if len(id_list) < 4:
    #             fourth_decoding.fourth(img, led_boxes, threshold,
    #                                    err_corr_list, id_list, center_list, contours_list, contours, flag1, flag2)  # 第四次解码，只解码第三次没有识别成功的
    # ################### 手机横竖坐标转换 #######################
    # if flag1 == 0:
    #     center_list_new = center_list[:]
    #     for i in range(len(center_list)):
    #         center_list_new[i][0], center_list_new[i][1] = center_list[i][1], center_list[i][0]
    # contours_list[i][[0,1], :] = contours_list[i][[1,0], :]
    # ################### 定位 #######################
    end = time.perf_counter()
    print("定位用时：", end - start, "秒")
    keypoint_1, keypoint_2, matches = match.find_feature_matches(img_match, img1_match)
    [R, t] = match.poes_estimation_2d2d(keypoint_1, keypoint_2, matches)
    end = time.perf_counter()
    print("定位用时：", end - start, "秒")
    # ################### 寻找标志点 #######################

    point_box = contours_list
    minx=np.min(point_box[0])
    maxx=np.max(point_box[0])
    miny = np.min(point_box[1])
    maxy = np.max(point_box[1])
    gray_roi = cv2.cvtColor(img_match, cv2.COLOR_BGR2GRAY)  # 变成灰度图
    sub_image = img_match[miny:maxy, minx:maxx]
    # cv2.namedWindow("img", 0)
    # cv2.resizeWindow("img", 816, 612)
    # cv2.imshow("img", sub_image)
    # cv2.waitKey(0)
    point_xy = detect_objects.detectpoint(sub_image,(minx+maxx)/2,(miny+maxy)/2)
    point_xy[0] = point_xy[0] + minx
    point_xy[1] = point_xy[1] + miny
    np.savez('points1/data', id_list, contours_list, center_list, point_xy)
    point_box1 = contours_list1
    minx=np.min(point_box1[0])
    maxx=np.max(point_box1[0])
    miny = np.min(point_box1[1])
    maxy = np.max(point_box1[1])
    sub_image1 = img1_match[miny:maxy, minx:maxx]
    # cv2.namedWindow("img", 0)
    # cv2.resizeWindow("img", 816, 612)
    # cv2.imshow("img", sub_image)
    # cv2.waitKey(0)
    point_xy1 = detect_objects.detectpoint(sub_image1,(minx+maxx)/2,(miny+maxy)/2)
    point_xy1[0] = point_xy1[0] + minx
    point_xy1[1] = point_xy1[1] + miny
    # ################### 定位 #######################
    np.savez('points1/data', id_list, contours_list, center_list, point_xy)
    # pos = VO_position.Vo_Frames(id_list, contours_list, center_list, flag1, id_list1, contours_list1, center_list1, flag2, R)
    pos = VO_position_auto.Vo_Frames(id_list, contours_list, center_list, flag1, id_list1, contours_list1, center_list1,
                                     flag2, R, point_xy, point_xy1)
    # [position_x, position_y] = Ellipse_position.Ellipse(id_list, contours, flag1)
    # [position_x, position_y] = VO_position.positioning(id_list, center_list, contours_list, flag1)
    # print("X坐标：", position_x, "Y坐标：", position_y)
    # ################### 输出 #######################
    end = time.perf_counter()
    print("定位用时：", end - start, "秒")
    return pos, end - start


flag1 = 0  # 手机横竖标志位 1表示横屏拍摄，0表·    示竖屏拍摄
flag2 = 0  # 倾斜角度标志位 1表示180° 0表示0°
pos = []
times = []
for i in range(100):
    num1 =20230411232316+2 * i
    num2 = num1 + 10
    path1 = str(num1)
    path2 = str(num2)
    pos1, time1 = main(path1, path2)
    print("num" + "\t" + str(num2))
    pos = "1.4m.txt"
    with open(pos, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        file.write(str(pos1[0]) + "\t" + str(pos1[1]) + "\t" + str(pos1[2]) + "\t" + str(time1) + "\n")
