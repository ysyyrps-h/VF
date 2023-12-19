import cv2
import numpy as np
import itertools
import math


def fourth(img, bounding_boxes, threshold, err_list, id, center, contours_list, contours, flag1, flag2):
    for bbox in range(len(bounding_boxes)):
        if err_list[bbox] == 0:
            [x, y, w, h] = bounding_boxes[bbox]
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img_roi = img[y:(y + h), x:(x + w)]
            gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)  # 变成灰度图
            kernel = np.ones((4, 4), np.uint8)
            ret, binary_roi = cv2.threshold(gray_roi, threshold - 20, 255, cv2.THRESH_BINARY)
            dst = cv2.erode(binary_roi, kernel, 1)
            # dst = cv2.dilate(dst, kernel1, 1)
            contours_roi, hierarchy = cv2.findContours(binary_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
            temp1 = []
            assilist = []
            if flag1 == 1:
                for j in range(h):
                    temp1.append(dst[j][int(w / 2)])
            if flag1 == 0:
                for j in range(w):
                    temp1.append(dst[int(h / 2)][j])
            if flag2 == 1:
                temp2 = list(reversed(temp1))
            if flag2 == 0:
                temp2 = temp1
            binlist = [len(list(v)) for k, v in itertools.groupby(temp2)]
            binlist1 = binlist[1:len(binlist) - 1]
            th1_low = int(max(binlist1) / 3)
            th1_high = th1_low + 2
            th2_high = th1_low * 2 + 2
            th3_high = max(binlist1) + 2
            th4_high = th1_high * 4
            a = []
            for i in range(len(binlist1)):
                if temp2[int(binlist[0] / 2)] == 0:  # 如果ROI区域内起始是0000000，那么第一个条纹就是亮条纹，因为binlist1去掉了binlist的第一个
                    if binlist1[i] <= th1_high:
                        a.append((i + 1) % 2)
                        assilist.append((i + 1) % 2)
                    elif th1_high < binlist1[i] <= th2_high:
                        assilist.append((i + 1) % 2)
                        for k in range(2):
                            a.append((i + 1) % 2)
                    elif th2_high < binlist1[i] <= th3_high:
                        assilist.append((i + 1) % 2)
                        for k in range(3):
                            a.append((i + 1) % 2)
                    elif th3_high < binlist1[i] <= th4_high:
                        assilist.append((i + 1) % 2)
                        for k in range(4):
                            a.append((i + 1) % 2)
                    else:
                        continue
                else:  # 如果ROI区域内起始是255,255,255,255,255，那么第一个条纹就是暗条纹，因为binlist1去掉了binlist的第一个
                    if binlist1[i] <= th1_high:
                        a.append(i % 2)
                        assilist.append(i % 2)
                    elif th1_high < binlist1[i] <= th2_high:
                        assilist.append(i % 2)
                        for k in range(2):
                            a.append(i % 2)
                    elif th2_high < binlist1[i] <= th3_high:
                        assilist.append(i % 2)
                        for k in range(3):
                            a.append(i % 2)
                    elif th3_high < binlist1[i] <= th4_high:
                        assilist.append(i % 2)
                        for k in range(4):
                            a.append(i % 2)
                    else:
                        continue
            decodelist = [len(list(v)) for k, v in itertools.groupby(a)]
            position_list = []
            for v in range(len(decodelist)):
                if 3 == decodelist[v]:
                    position_list.append(v)
                else:
                    continue
            diff_list_alternate = []
            diff_list_adjoin = []
            for i in range(len(position_list) - 2):
                diff_list_alternate.append(position_list[i + 2] - position_list[i])
            for i in range(len(position_list) - 1):
                diff_list_adjoin.append(position_list[i + 1] - position_list[i])
            for i in range(len(diff_list_alternate)):
                if diff_list_alternate[i] == 6:
                    temp_decode_list_two = decodelist[position_list[i] + 1:position_list[i + 2]]
                    if temp_decode_list_two[0] == 1 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 2 \
                            and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 3 and assilist[position_list[i]] == 0:
                        id.append(1)
                        center.append([center_x, center_y])
                        contours_list.append(contours[bbox])
                        print("ID=1", [center_x, center_y], a)
                        break
                    if temp_decode_list_two[0] == 3 and temp_decode_list_two[1] == 1 and temp_decode_list_two[2] == 2 \
                            and temp_decode_list_two[3] == 2 and temp_decode_list_two[4] == 1 and assilist[position_list[i]] == 1:
                        id.append(1)
                        center.append([center_x, center_y])
                        contours_list.append(contours[bbox])
                        print("ID=1", [center_x, center_y], a)
                        break
                    if temp_decode_list_two[0] == 1 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 3 \
                            and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 2:
                        id.append(2)
                        center.append([center_x, center_y])
                        contours_list.append(contours[bbox])
                        print("ID=2", [center_x, center_y], a)
                        break
                    if temp_decode_list_two[0] == 3 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 2 \
                            and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 1 and assilist[position_list[i]] == 0:
                        id.append(3)
                        center.append([center_x, center_y])
                        contours_list.append(contours[bbox])
                        print("ID=3", [center_x, center_y], a)
                        break
                    if temp_decode_list_two[0] == 2 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 1 \
                            and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 3 and assilist[position_list[i]] == 1:
                        id.append(3)
                        center.append([center_x, center_y])
                        contours_list.append(contours[bbox])
                        print("ID=3", [center_x, center_y], a)
                        break
                    if temp_decode_list_two[0] == 3 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 1 \
                            and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 2 and assilist[position_list[i]] == 0:
                        id.append(4)
                        center.append([center_x, center_y])
                        contours_list.append(contours[bbox])
                        print("ID=4", [center_x, center_y], a)
                        break
                    if temp_decode_list_two[0] == 2 and temp_decode_list_two[1] == 1 and temp_decode_list_two[2] == 1 \
                            and temp_decode_list_two[3] == 2 and temp_decode_list_two[4] == 3 and assilist[position_list[i]] == 1:
                        id.append(4)
                        center.append([center_x, center_y])
                        contours_list.append(contours[bbox])
                        print("ID=4", [center_x, center_y], a)
                        break
                    if temp_decode_list_two[0] == 2 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 3 \
                            and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 1 and assilist[position_list[i]] == 0:
                        id.append(5)
                        center.append([center_x, center_y])
                        contours_list.append(contours[bbox])
                        print("ID=5", [center_x, center_y], a)
                        break
                    if temp_decode_list_two[0] == 1 and temp_decode_list_two[1] == 1 and temp_decode_list_two[2] == 3 \
                            and temp_decode_list_two[3] == 2 and temp_decode_list_two[4] == 2 and assilist[position_list[i]] == 1:
                        id.append(5)
                        center.append([center_x, center_y])
                        contours_list.append(contours[bbox])
                        print("ID=5", [center_x, center_y], a)
                        break
                    if temp_decode_list_two[0] == 2 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 1 \
                            and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 3 and assilist[position_list[i]] == 0:
                        id.append(6)
                        center.append([center_x, center_y])
                        contours_list.append(contours[bbox])
                        print("ID=6", [center_x, center_y], a)
                        break
                    if temp_decode_list_two[0] == 3 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 2 \
                            and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 1 and assilist[position_list[i]] == 1:
                        id.append(6)
                        center.append([center_x, center_y])
                        contours_list.append(contours[bbox])
                        print("ID=6", [center_x, center_y], a)
                        break
                    else:
                        continue
                else:
                    continue
            for i in range(len(diff_list_adjoin)):
                if diff_list_adjoin[i] == 6:
                    temp_decode_list_two = decodelist[position_list[i] + 1:position_list[i + 1]]
                    if temp_decode_list_two[0] == 2 and temp_decode_list_two[1] == 2 and temp_decode_list_two[2] == 2 \
                            and temp_decode_list_two[3] == 1 and temp_decode_list_two[4] == 2:
                        id.append(6)
                        center.append([center_x, center_y])
                        contours_list.append(contours[bbox])
                        print("ID=6", [center_x, center_y], a)
                        break
                    else:
                        continue
                else:
                    continue
