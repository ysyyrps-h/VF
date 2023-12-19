from fingerprint import Fingerprint

import Ellipse_position
import Trilateral_position
import Vp4l_position
import numpy as np
import time


def positioning(id_list, center_list, contours, flag):
    position_x = -1
    position_y = -1
    num_led = len(id_list)
    print("led_num=", num_led)
    # print(center_list)

    if num_led == 2:
        [position_x, position_y] = Ellipse_position.Ellipse(id_list, contours, flag)
    elif num_led == 3:
        [position_x, position_y] = Trilateral_position.trilateral(id_list, center_list)
        id_list_new1 = id_list[0:2]
        contours_list_new = contours[0:2]
        [position_x1, position_y1] = Ellipse_position.Ellipse(id_list_new1, contours_list_new, flag)
        print([position_x1, position_y1])
    elif num_led == 4:
        id_list_new = sorted(id_list)
        ind_x_list = []
        ind_y_list = []
        center_list_new = []
        pos_matrix = np.array([[2, 1], [4, 3], [6, 5]]).T  # LED的索引表，以物理坐标（0,0）为起点，行值为x，列值为y
        for i in id_list_new:
            [ind_x, ind_y] = np.where(pos_matrix == i)
            ind_x_list.append(ind_x[0])
            ind_y_list.append(ind_y[0])
        if ind_y_list[0] == ind_y_list[1] and ind_y_list[2] == ind_y_list[3]:  # 保证是矩形
            if ind_x_list[1] == ind_x_list[3] and ind_x_list[0] == ind_x_list[2]:
                id_list_new[0] = pos_matrix[min(ind_x_list), min(ind_y_list)]  # 保证正序或者逆序
                id_list_new[1] = pos_matrix[min(ind_x_list), max(ind_y_list)]
                id_list_new[2] = pos_matrix[max(ind_x_list), max(ind_y_list)]
                id_list_new[3] = pos_matrix[max(ind_x_list), min(ind_y_list)]
        for i in range(len(id_list_new)):  # 像素坐标也要调整
            center_list_new.append(center_list[id_list.index(id_list_new[i])])
        [position_x, position_y] = Vp4l_position.vp4l(id_list_new, center_list_new, flag)
        n1=2
        n2=3
        id_list_new1 = [id_list[n1], id_list[n2]]
        contours_list_new = [contours[n1], contours[n2]]
        [position_x1, position_y1] = Ellipse_position.Ellipse(id_list_new1, center_list, flag)
        # [position_x1, position_y1] = Ellipse_position.Ellipse(id_list_new1, contours_list_new, flag)
        # print([position_x1, position_y1])
    return [position_x, position_y]
