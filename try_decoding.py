import cv2
import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
from itertools import chain


def column_probability(grayvalue):
    """
    计算一维向量内像素值出现概率
    :param grayvalue一维像素值
    :return:prob[256]
    """
    prob = np.zeros(shape=256)
    for cv in grayvalue:
        prob[cv] += 1
    prob = prob / len(grayvalue)
    return prob


def pixel_probability(img):
    """
    计算像素值出现概率
    :param img:
    :return:
    """
    assert isinstance(img, np.ndarray)
    prob = np.zeros(shape=256)
    for rv in img:
        for cv in rv:
            prob[cv] += 1

    r, c = img.shape
    prob = prob / (r * c)
    return prob


def probability_to_histogram(img, prob):
    """
    根据像素概率将原始图像直方图均衡化
    :param img:
    :param prob:
    :return: 直方图均衡化后的图像
    """
    prob = np.cumsum(prob)  # 累计概率

    img_map = [int(i * prob[i]) for i in range(256)]  # 像素值映射

    # 像素值替换
    assert isinstance(img, np.ndarray)
    r, c = img.shape
    for ri in range(r):
        for ci in range(c):
            img[ri, ci] = img_map[img[ri, ci]]

    return img


def column_probability_to_histogram(grayvalue, prob):
    """
    根据像素概率将原始列向量直方图均衡化
    :param img:
    :param prob:
    :return: 直方图均衡化后的图像
    """
    prob = np.cumsum(prob)  # 累计概率

    img_map = [int(i * prob[i]) for i in range(256)]  # 像素值映射

    # 像素值替换
    grayvalue1 = grayvalue[:]
    c = len(grayvalue)
    for cv in range(c):
        grayvalue1[cv] = img_map[grayvalue[cv]]
    return grayvalue1


def plot(y, name):
    """
    画直方图，len(y)==gray_level
    :param y: 概率值
    :param name:
    :return:
    """
    plt.figure(num=name)
    plt.bar([i for i in range(256)], y, width=1)


def try_decode(img, bounding_boxes, threshold, err_list, id, center, contours_list, flag1, contours):
    for bbox in range(len(bounding_boxes)):
        [x, y, w, h] = bounding_boxes[bbox]
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        # print([center_x, center_y])
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img_roi = img[y:(y + h), x:(x + w)]
        #Blur_roi=cv2.medianBlur(img_roi, 3)
        gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)  # 变成灰度图
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray_roi)
        m = []
        for j in range(h):
            m.append(gray_roi[j][int(w / 2)])
        m1 = np.linspace(0, h, h)
        a = np.polyfit(m1, m, 2)  # 用2次多项式拟合x，y数组
        b = np.poly1d(a)  # 拟合完之后用这个函数来生成多项式对象
        c = b(m1)  # 生成多项式对象之后，就是获取x在这个多项式处的值
        m2 = []
        diff = [0 for index in range(len(c))]
        th=max(m)/2
        #re,th1 = cv2.threshold(gray_roi,th,255,cv2.THRESH_BINARY)
        re, th1 = cv2.threshold(gray_roi, threshold, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        re1, th1 = cv2.threshold(cl, 0, 255, cv2.THRESH_OTSU)
        re2, th2 = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_OTSU)
        #ret, th1 = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY)  # 二值化，阈值要设的小一点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(th1, kernel)
        for j in range(h):
            m2.append(eroded[j][int(len(eroded[1])/2)])
            #m2.append(th1[j][int(len(th1[1]) / 2)])
        # for j in range(h):
        #     if m[j] < c[j]:
        #         m2.append(0)
        #     else:
        #         if m[j] - c[j] > 20 and j > 0 and j + 1 < h and m[j - 1] - c[j - 1] > 20 and m[j + 1] - c[j + 1] > 20:
        #             m2.append(1)
        #         else:
        #             m2.append(0)
        #         diff[j] = m[j] - c[j]

        # index1 = 0
        # index2 = 0
        # flag = 0
        # for j in range(h):
        #     if flag == 0 and 0 < j < len(diff) and (diff[j] != 0 and diff[j - 1] == 0 and diff[j + 1] != 0):
        #         index1 = j
        #         flag = 1
        #     if flag == 1 and 0 < j < len(diff) and (diff[j] == 0 and diff[j - 1] != 0 and diff[j + 1] == 0):
        #         index2 = j
        #         flag = 0
        #         if index2 - index1 < 10:
        #             continue
        #         k1 = np.linspace(0, index2 - index1, index2 - index1)
        #         a1 = np.polyfit(k1, m[index1:index2], 2)  # 用2次多项式拟合x，y数组
        #         b1 = np.poly1d(a1)  # 拟合完之后用这个函数来生成多项式对象
        #         c1 = b1(k1) - 15  # 生成多项式对象之后，就是获取x在这个多项式处的值
        #         # plt.plot(k1, m[index1:index2], label='original datas')  # 对原始数据画散点图
        #         # plt.plot(k1, c1, ls='--', c='red', label='fitting with second-degree polynomial')  # 对拟合之后的数据，也就是x，c数组画图
        #         # plt.legend()
        #         # plt.show()
        #         for jj in range(index1 + 5, index2 - 5):
        #             if m[jj] < c1[jj - index1]:
        #                 m2[jj] = 0
        #         mmm = 1
        # min_index = diff[index1+5:index2-5].index(min(diff[index1+5:index2-5]))
        # if abs(min_index - index1) > 5 and abs(min_index - index2) > 5:
        #     continue
        binlist = [len(list(v)) for k, v in itertools.groupby(m2)]

        # 二项式拟合阈值
        # plt.plot(m1, m, label='original datas')  # 对原始数据画散点图
        # plt.plot(m1, c, ls='--', c='red', label='fitting with second-degree polynomial')  # 对拟合之后的数据，也就是x，c数组画图
        # plt.legend()
        # plt.show()

        # 直方图均衡

        # plt.hist(gray_roi.ravel(), 100)
        # plt.show()
        # prob = column_probability(m)
        # cv2.imshow("roi",gray_roi)
        # cv2.waitKey(0)
        # # 直方图均衡化
        # grayvalue1 = column_probability_to_histogram(m, prob)
        # prob = column_probability(grayvalue1)
        # plot(prob, "直方图均衡化结果")
        # plt.show()

        # kernel = np.ones((2, 2), np.uint8)
        # ret, binary_roi = cv2.threshold(gray_roi, threshold, 255, cv2.THRESH_OTSU)
        # dst = cv2.erode(binary_roi, kernel)
        # cv2.namedWindow("img", 0)
        # cv2.resizeWindow("img", 1040, 780)
        # cv2.imshow("img", binary_roi)
        # cv2.waitKey(0)
        # dst = cv2.dilate(dst, kernel1, 1)
        # contours_roi, hierarchy = cv2.findContours(binary_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
        # for i in range(len(contours_roi)):
        #     count = len(contours_roi[i])
        #     if count < 6:
        #         continue
        #     box = cv2.fitEllipse(contours_roi[i])

        # // 如果长宽比大于30，则排除，不做拟合
        # if (MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height) * 30)
        #     continue;
        # cv2.drawContours(img_roi, contours_roi, -1, (0, 0, 255), 1)
        # 画出拟合的椭圆
        # cv2.ellipse(img_roi, box, (0, 0, 255), 1)
        # cv2.imshow("name", img_roi)
        # cv2.waitKey(0)

        # cv2.namedWindow('img', 0)
        # cv2.resizeWindow('img', 1500, 2000)  # 自己设定窗口图片的大小
        # cv2.imshow("img", binary)
        # cv2.waitKey(0)
        # binlist2 = []
        if len(binlist)<8:
            continue
        binlist1 = binlist[1:len(binlist) - 1]
        # binlist1_j = 0
        # while binlist1_j < len(binlist1):
        #     if binlist1[binlist1_j] < 3 and binlist1_j < len(binlist1) - 1:
        #         binlist2[-1] += binlist1[binlist1_j] + binlist1[binlist1_j + 1]
        #         binlist1_j += 2
        #     else:
        #         binlist2.append(binlist1[binlist1_j])
        #         binlist1_j += 1
        if m2[0] == 0:  # 第一个条纹是暗条纹，则亮条纹在奇数位置，索引为偶数
            lightlist = binlist1[::2]
            darklist = binlist1[1::2]
        else:
            lightlist = binlist1[1::2]
            darklist = binlist1[::2]
        binmax = max (binlist1)
        lightmax = max(lightlist)
        darkmax = max(darklist)
        # light_th1_low = int(lightmax / 3) - 2
        # light_th1_high = int(lightmax / 3) + 2
        # light_th2_low = light_th1_high + 1
        # light_th2_high = int(lightmax / 3) * 2 + 2
        # light_th3_low = light_th2_high + 1
        # light_th3_high = lightmax + 2
        # dark_th1_low = int(darkmax / 3) - 2
        # dark_th1_high = int(darkmax / 3) + 2
        # dark_th2_low = dark_th1_high + 1
        # dark_th2_high = int(darkmax / 3) * 2 + 2
        # dark_th3_low = dark_th2_high + 1
        # dark_th3_high = darkmax + 2
        light_th1_low = int(binmax / 3)-5
        light_th1_high = int(binmax / 3)+2
        light_th2_low = light_th1_high + 1
        light_th2_high = int(binmax / 3) * 2+2
        light_th3_low = light_th2_high + 1
        light_th3_high = binmax + 10
        ligh_width = []
        dark_width = []
        all_width = []
        assitant = []
        for i in range(len(binlist1)):
            if binlist1[i] <= light_th1_high:
                all_width.append(1)
            elif light_th2_low <= binlist1[i] <= light_th2_high:
                all_width.append(2)
            elif light_th3_low <= binlist1[i] <= light_th3_high:
                all_width.append(3)
        # for i in range(len(lightlist)):
        #     if lightlist[i] < light_th1_low:
        #         continue
        #     elif light_th1_low <= lightlist[i] <= light_th1_high:
        #         ligh_width.append(1)
        #     elif light_th2_low <= lightlist[i] <= light_th2_high:
        #         ligh_width.append(2)
        #     elif light_th3_low <= lightlist[i] <= light_th3_high:
        #         ligh_width.append(3)
        # for i in range(len(darklist)):
        #     if darklist[i] < dark_th1_low:
        #         continue
        #     elif dark_th1_low <= darklist[i] <= dark_th1_high:
        #         dark_width.append(1)
        #     elif dark_th2_low <= darklist[i] <= dark_th2_high:
        #         dark_width.append(2)
        #     elif dark_th3_low <= darklist[i] <= dark_th3_high:
        #         dark_width.append(3)
        # for i in range(max(len(dark_width), len(ligh_width))):
        #     if m2[0] == 0:
        #         if i < len(ligh_width):
        #             all_width.append(ligh_width[i])
        #             assitant.append(1)
        #         if i < len(dark_width):
        #             all_width.append(dark_width[i])
        #             assitant.append(0)
        #     else:
        #         if i < len(dark_width):
        #             all_width.append(dark_width[i])
        #             assitant.append(0)
        #         if i < len(ligh_width):
        #             all_width.append(ligh_width[i])
        #             assitant.append(1)
        for i in range(len(binlist1)):
            if m2[0]!= 0 :#第一个条纹是暗条纹(索引为0），则亮条纹在偶数位置（索引为奇数）
                assitant.append((i+2)%2)
            else:
                assitant.append((i + 1) % 2)
        #  这里判断首尾，不需要帧头
        # tail = len(all_width)
        # if tail > 5:
        #     if all_width[0] == 1 and all_width[1] == 2 and all_width[2] == 2 \
        #             and all_width[3] == 1 and all_width[4] == 3 and assitant[0] == 1:
        #         err_list[bbox] = 1
        #         id.append(1)
        #         print("ID=1", [center_x, center_y], a)
        #         continue
        #     if all_width[tail - 5] == 3 and all_width[tail - 4] == 1 and all_width[tail - 3] == 2 \
        #             and all_width[tail - 2] == 2 and all_width[tail - 1] == 1 and assitant[tail - 5] == 0:
        #         err_list[bbox] = 1
        #         id.append(1)
        #         print("ID=1", [center_x, center_y], a)
        #         continue
        #     if all_width[0] == 1 and all_width[1] == 2 and all_width[2] == 3 \
        #             and all_width[3] == 1 and all_width[4] == 2 and assitant[0] == 1:
        #         err_list[bbox] = 1
        #         id.append(2)
        #         print("ID=2", [center_x, center_y], a)
        #         continue
        #     if all_width[tail - 5] == 1 and all_width[tail - 4] == 2 and all_width[tail - 3] == 3 \
        #             and all_width[tail - 2] == 1 and all_width[tail - 1] == 2 and assitant[tail - 5] == 0:
        #         err_list[bbox] = 1
        #         id.append(2)
        #         print("ID=2", [center_x, center_y], a)
        #         continue
        #     if all_width[0] == 3 and all_width[1] == 2 and all_width[2] == 2 \
        #             and all_width[3] == 1 and all_width[4] == 1 and assitant[0] == 1:
        #         err_list[bbox] = 1
        #         id.append(3)
        #         print("ID=3", [center_x, center_y], a)
        #         continue
        #     if all_width[tail - 5] == 2 and all_width[tail - 4] == 2 and all_width[tail - 3] == 1 \
        #             and all_width[tail - 2] == 1 and all_width[tail - 1] == 3 and assitant[tail - 5] == 0:
        #         err_list[bbox] = 1
        #         id.append(3)
        #         print("ID=3", [center_x, center_y], a)
        #         continue
        #     if all_width[0] == 3 and all_width[1] == 2 and all_width[2] == 1 \
        #             and all_width[3] == 1 and all_width[4] == 2 and assitant[0] == 1:
        #         err_list[bbox] = 1
        #         id.append(4)
        #         print("ID=4", [center_x, center_y], a)
        #         continue
        #     if all_width[tail - 5] == 2 and all_width[tail - 4] == 1 and all_width[tail - 3] == 1 \
        #             and all_width[tail - 2] == 2 and all_width[tail - 1] == 3 and assitant[tail - 5] == 0:
        #         err_list[bbox] = 1
        #         id.append(4)
        #         print("ID=4", [center_x, center_y], a)
        #         continue
        #     if all_width[0] == 2 and all_width[1] == 2 and all_width[2] == 3 \
        #             and all_width[3] == 1 and all_width[4] == 1 and assitant[0] == 1:
        #         err_list[bbox] = 1
        #         id.append(5)
        #         print("ID=5", [center_x, center_y], a)
        #         continue
        #     if all_width[tail - 5] == 1 and all_width[tail - 4] == 1 and all_width[tail - 3] == 3 \
        #             and all_width[tail - 2] == 2 and all_width[tail - 1] == 2 and assitant[tail - 5] == 0:
        #         err_list[bbox] = 1
        #         id.append(5)
        #         print("ID=5", [center_x, center_y], a)
        #         continue
        #     if all_width[0] == 2 and all_width[1] == 2 and all_width[2] == 1 \
        #             and all_width[3] == 1 and all_width[4] == 3 and assitant[0] == 1:
        #         err_list[bbox] = 1
        #         id.append(6)
        #         print("ID=6", [center_x, center_y], a)
        #         continue
        #     if all_width[tail - 5] == 3 and all_width[tail - 4] == 2 and all_width[tail - 3] == 2 \
        #             and all_width[tail - 2] == 1 and all_width[tail - 1] == 1 and assitant[tail - 5] == 0:
        #         err_list[bbox] = 1
        #         id.append(6)
        #         print("ID=6", [center_x, center_y], a)
        #         continue
        #     if all_width[0] == 2 and all_width[1] == 1 and all_width[2] == 3 \
        #             and all_width[3] == 2 and all_width[tail - 1] == 1:
        #         err_list[bbox] = 1
        #         id.append(6)
        #         print("ID=6", [center_x, center_y], a)
        #         continue
        #     if all_width[tail - 5] == 2 and all_width[tail - 4] == 1 and all_width[tail - 3] == 3 \
        #             and all_width[tail - 2] == 2 and all_width[tail - 1] == 1:
        #         err_list[bbox] = 1
        #         id.append(6)
        #         print("ID=7", [center_x, center_y], a)
        #         continue
        position_list = []
        for v in range(len(all_width)):
            if 3 == all_width[v]:
                position_list.append(v)
        diff_list_alternate = []  # 相间
        diff_list_adjoin = []  # 相邻
        for i in range(len(position_list) - 2):
            diff_list_alternate.append(position_list[i + 2] - position_list[i])
        for i in range(len(position_list) - 1):
            diff_list_adjoin.append(position_list[i + 1] - position_list[i])
        for i in range(len(diff_list_alternate)):
            if diff_list_alternate[i] == 6:
                temp_decode_list_two = all_width[position_list[i] + 1:position_list[i + 2]]
                if temp_decode_list_two[0] == 1 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 2 \
                        and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 3 and assitant[
                    position_list[i]] == 0:  # 以000为帧头的UID,最开始是暗条纹，由于去除了首尾，所以暗条纹在奇数位置
                    err_list[bbox] = 1
                    id.append(1)
                    center.append([center_x, center_y])
                    contours_list.append(contours[bbox])
                    print("ID=1", [center_x, center_y], a)
                    break
                if temp_decode_list_two[0] == 3 and temp_decode_list_two[1] == 1 and temp_decode_list_two[2] == 2 \
                        and temp_decode_list_two[3] == 2 and temp_decode_list_two[4] == 1 and assitant[
                    position_list[i]] == 1:  # 以111为帧头的UID,最开始是暗条纹，由于去除了首尾，所以亮条纹在偶数位置
                    err_list[bbox] = 1
                    id.append(1)
                    center.append([center_x, center_y])
                    contours_list.append(contours[bbox])
                    print("ID=1", [center_x, center_y], a)
                    break
                if temp_decode_list_two[0] == 1 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 3 \
                        and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 2:
                    err_list[bbox] = 1
                    id.append(2)
                    center.append([center_x, center_y])
                    contours_list.append(contours[bbox])
                    print("ID=2", [center_x, center_y], a)
                    break
                if temp_decode_list_two[0] == 3 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 2 \
                        and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 1 and assitant[
                    position_list[i]] == 0:
                    err_list[bbox] = 1
                    id.append(3)
                    center.append([center_x, center_y])
                    contours_list.append(contours[bbox])
                    print("ID=3", [center_x, center_y], a)
                    break
                if temp_decode_list_two[0] == 2 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 1 \
                        and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 3 and assitant[
                    position_list[i]] == 1:
                    err_list[bbox] = 1
                    id.append(3)
                    center.append([center_x, center_y])
                    contours_list.append(contours[bbox])
                    print("ID=3", [center_x, center_y], a)
                    break
                if temp_decode_list_two[0] == 3 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 1 \
                        and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 2 and assitant[
                    position_list[i]] == 0:
                    err_list[bbox] = 1
                    id.append(4)
                    center.append([center_x, center_y])
                    contours_list.append(contours[bbox])
                    print("ID=4", [center_x, center_y], a)
                    break
                if temp_decode_list_two[0] == 2 and temp_decode_list_two[1] == 1 and temp_decode_list_two[2] == 1 \
                        and temp_decode_list_two[3] == 2 and temp_decode_list_two[4] == 3 and assitant[position_list[i]] == 1:
                    err_list[bbox] = 1
                    id.append(4)
                    center.append([center_x, center_y])
                    contours_list.append(contours[bbox])
                    print("ID=4", [center_x, center_y], a)
                    break
                if temp_decode_list_two[0] == 2 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 3 \
                        and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 1 and assitant[position_list[i]] == 0:
                    err_list[bbox] = 1
                    id.append(5)
                    center.append([center_x, center_y])
                    contours_list.append(contours[bbox])
                    print("ID=5", [center_x, center_y], a)
                    break
                if temp_decode_list_two[0] == 1 and temp_decode_list_two[1] == 1 and temp_decode_list_two[2] == 3 \
                        and temp_decode_list_two[3] == 2 and temp_decode_list_two[4] == 2 and assitant[position_list[i]] == 1:
                    err_list[bbox] = 1
                    id.append(5)
                    center.append([center_x, center_y])
                    contours_list.append(contours[bbox])
                    print("ID=5", [center_x, center_y], a)
                    break
                if temp_decode_list_two[0] == 2 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 1 \
                        and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 3 and assitant[position_list[i]] == 0:
                    err_list[bbox] = 1
                    id.append(6)
                    center.append([center_x, center_y])
                    contours_list.append(contours[bbox])
                    print("ID=6", [center_x, center_y], a)
                    break
                if temp_decode_list_two[0] == 3 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 2 \
                        and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 1 and assitant[position_list[i]] == 1:
                    err_list[bbox] = 1
                    id.append(6)
                    center.append([center_x, center_y])
                    contours_list.append(contours[bbox])
                    print("ID=6", [center_x, center_y], a)
                    break
        # for i in range(len(diff_list_adjoin)):
        #     if diff_list_adjoin[i] == 6:
        #         temp_decode_list_two = decodelist[position_list[i] + 1:position_list[i + 1]]
        #         if temp_decode_list_two[0] == 2 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 2 \
        #                 and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 2:
        #             err_list[bbox] = 1
        #             id.append(6)
        #             center.append([center_x, center_y])
        #             contours_list.append(contours[bbox])
        #             print("ID=6", [center_x, center_y], a)
        #             break
        #         else:
        #             continue
        #     else:
        #         continue
