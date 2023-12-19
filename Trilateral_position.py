import numpy as np
import math


def trilateral(id_list, center_list):
    f = 3000  # 焦距
    dx = 352.78
    dy = 352.78  # 图像中像素点的物理长度352.78 * 352.78um
    u0 = 2080
    v0 = 1560  # 主点
    led_gs = np.array([[0, 30], [0, 0], [30, 30], [30, 0], [60, 30], [60, 0]]).T  # LED的物理坐标
    led_pos_pixel = np.zeros((2, 3), dtype=int)
    led_corner_gs = np.zeros((2, 3), dtype=int)
    for i in range(len(id_list)):
        led_corner_gs[:, i] = led_gs[:, id_list[i]-1]  # LED在世界坐标系中的位置坐标
        led_pos_pixel[:, i] = np.array(center_list[i])  # led投影点的像素坐标，通过相机获取图片后，2x4 double
    d_ab = math.sqrt((led_corner_gs[0, 0] - led_corner_gs[0, 1]) ** 2 + (led_corner_gs[1, 0] - led_corner_gs[1, 0]) ** 2)
    d_bc = math.sqrt((led_corner_gs[0, 1] - led_corner_gs[0, 2]) ** 2 + (led_corner_gs[1, 1] - led_corner_gs[1, 2]) ** 2)
    d_ac = math.sqrt((led_corner_gs[0, 0] - led_corner_gs[0, 2]) ** 2 + (led_corner_gs[1, 0] - led_corner_gs[1, 2]) ** 2)
    d_ab_ = math.sqrt(((led_pos_pixel[0, 0] - led_pos_pixel[0, 1]) * dx / 1000000) ** 2 + (
                (led_pos_pixel[1, 0] - led_pos_pixel[1, 1]) * dy / 1000000) ** 2)
    d_bc_ = math.sqrt(((led_pos_pixel[0, 1] - led_pos_pixel[0, 2]) * dx / 1000000) ** 2 + (
                (led_pos_pixel[1, 1] - led_pos_pixel[1, 2]) * dy / 1000000) ** 2)
    d_ac_ = math.sqrt(((led_pos_pixel[0, 0] - led_pos_pixel[0, 2]) * dx / 1000000) ** 2 + (
                (led_pos_pixel[1, 0] - led_pos_pixel[1, 2]) * dy / 1000000) ** 2)
    d_o_a_ = math.sqrt(((led_pos_pixel[0, 0] - u0) * dx / 1000000) ** 2 + ((led_pos_pixel[1, 0] - v0) * dy / 1000000) ** 2)
    d_o_b_ = math.sqrt(((led_pos_pixel[0, 1] - u0) * dx / 1000000) ** 2 + ((led_pos_pixel[1, 1] - v0) * dy / 1000000) ** 2)
    d_o_c_ = math.sqrt(((led_pos_pixel[0, 2] - u0) * dx / 1000000) ** 2 + ((led_pos_pixel[1, 2] - v0) * dy / 1000000) ** 2)
    h_ab = f * d_ab / d_ab_
    h_bc = f * d_bc / d_bc_
    h_ac = f * d_ac / d_ac_
    h = (h_ab+h_bc+h_ac)/3
    d_o1a = d_o_a_ * h / f
    d_o1b = d_o_b_ * h / f
    d_o1c = d_o_c_ * h / f
    M = np.array([[led_corner_gs[0, 1]-led_corner_gs[0, 0], led_corner_gs[0, 2]-led_corner_gs[0, 0]],
                   [led_corner_gs[1, 1]-led_corner_gs[1, 0], led_corner_gs[1, 2]-led_corner_gs[1, 0]]]).T
    D = np.array([(d_o1a**2-d_o1b**2+led_corner_gs[0, 1]**2+led_corner_gs[1, 1]**2-led_corner_gs[0, 0]**2+led_corner_gs[1, 0]**2)/2,
                  (d_o1a**2-d_o1c**2+led_corner_gs[0, 2]**2+led_corner_gs[1, 2]**2-led_corner_gs[0, 0]**2+led_corner_gs[1, 0]**2)/2]).T
    temp1 = np.dot(M.T, M)
    temp2 = np.dot(np.linalg.pinv(temp1), M.T)
    result = np.dot(temp2, D)
    return [result[0], result[1]]
