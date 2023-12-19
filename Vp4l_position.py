import numpy as np
import math


def vp4l(id_list, center_list, flag):
    f = 3460  # 焦距
    dx = 1.12
    dy = 1.12  # 图像中像素点的物理长度1.12μm
    if flag == 0:
        u0 = 1560
        v0 = 2080  # 主点
    else:
        u0 = 2080
        v0 = 1560  # 主点
    a = np.array([[1/dx, 0, u0], [0, 1/dy, v0], [0, 0, 1]])  # 内参数矩阵
    h = 90  # 实验平台高度
    led_l = 60  # led长60cm
    led_w = 30  # led宽30cm
    led_area = led_l * led_w  # led面积
    led_num = 4  # led个数
    led_gs = np.array([[0, 30, h], [0, 0, h], [30, 30, h], [30, 0, h], [60, 30, h], [60, 0, h]]).T  # LED的物理坐标
    led_pos_pixel = np.zeros((2, 4), dtype=int)
    led_corner_gs = np.zeros((3, 4), dtype=int)
    for i in range(len(id_list)):  # led投影点的像素坐标，通过相机获取图片后，2x4 double
        led_corner_gs[:, i] = led_gs[:, id_list[i]-1]  # LED在世界坐标系中的位置坐标
        led_pos_pixel[:, i] = np.array(center_list[i])
    led_pos_img = np.zeros((3, 4), dtype=float)
    for i in range(led_num):
        temp = np.array([led_pos_pixel[0][i], led_pos_pixel[1][i], 1])
        led_pos_img[:, i] = np.dot(np.linalg.pinv(a), temp)
    theta1_1 = math.asin(abs(led_pos_img[0, 1] - led_pos_img[0, 0])/
                         math.sqrt((led_pos_img[1, 0] - led_pos_img[1, 1]) ** 2 + (led_pos_img[0, 0] - led_pos_img[0, 1]) ** 2))
    theta1_2 = math.asin(-abs(led_pos_img[0, 1] - led_pos_img[0, 0]) /
                         math.sqrt((led_pos_img[1, 0] - led_pos_img[1, 1]) ** 2 + (led_pos_img[0, 0] - led_pos_img[0, 1]) ** 2))

    theta2_1 = math.asin(abs(led_pos_img[0, 2] - led_pos_img[0, 1]) /
                         math.sqrt((led_pos_img[1, 1] - led_pos_img[1, 2]) ** 2 + (led_pos_img[0, 1] - led_pos_img[0, 2]) ** 2))
    theta2_2 = math.asin(-abs(led_pos_img[0, 2] - led_pos_img[0, 1]) /
                         math.sqrt((led_pos_img[1, 1] - led_pos_img[1, 2]) ** 2 + (led_pos_img[0, 1] - led_pos_img[0, 2]) ** 2))

    theta3_1 = math.asin(abs(led_pos_img[0, 3] - led_pos_img[0, 2]) /
                         math.sqrt((led_pos_img[1, 2] - led_pos_img[1, 3]) ** 2 + (led_pos_img[0, 2] - led_pos_img[0, 3]) ** 2))
    theta3_2 = math.asin(-abs(led_pos_img[0, 3] - led_pos_img[0, 2]) /
                         math.sqrt((led_pos_img[1, 2] - led_pos_img[1, 3]) ** 2 + (led_pos_img[0, 2] - led_pos_img[0, 3]) ** 2))

    theta4_1 = math.asin(abs(led_pos_img[0, 0] - led_pos_img[0, 3]) /
                         math.sqrt((led_pos_img[1, 3] - led_pos_img[1, 0]) ** 2 + (led_pos_img[0, 3] - led_pos_img[0, 0]) ** 2))
    theta4_2 = math.asin(-abs(led_pos_img[0, 0] - led_pos_img[0, 3]) /
                         math.sqrt((led_pos_img[1, 3] - led_pos_img[1, 0]) ** 2 + (led_pos_img[0, 3] - led_pos_img[0, 0]) ** 2))
    theta1 = theta2 = theta3 = theta4 = p1 = p2 = p3 = p4 = 0.0
    p1_1 = led_pos_img[0, 0] * math.cos(theta1_1) + led_pos_img[1, 0] * math.sin(theta1_1)
    p1_2 = led_pos_img[0, 0] * math.cos(theta1_2) + led_pos_img[1, 0] * math.sin(theta1_2)
    if abs(p1_1 - (led_pos_img[0, 1] * math.cos(theta1_1) + led_pos_img[1, 1] * math.sin(theta1_1))) < 1e-5:
        p1 = p1_1
        theta1 = theta1_1
    elif abs(p1_2 - (led_pos_img[0, 1] * math.cos(theta1_2) + led_pos_img[1, 1] * math.sin(theta1_2))) < 1e-5:
        p1 = p1_2
        theta1 = theta1_2

    p2_1 = led_pos_img[0, 1] * math.cos(theta2_1) + led_pos_img[1, 1] * math.sin(theta2_1)
    p2_2 = led_pos_img[0, 1] * math.cos(theta2_2) + led_pos_img[1, 1] * math.sin(theta2_2)
    if abs(p2_1 - (led_pos_img[0, 2] * math.cos(theta2_1) + led_pos_img[1, 2] * math.sin(theta2_1))) < 1e-5:
        p2 = p2_1
        theta2 = theta2_1
    elif abs(p2_2 - (led_pos_img[0, 2] * math.cos(theta2_2) + led_pos_img[1, 2] * math.sin(theta2_2))) < 1e-5:
        p2 = p2_2
        theta2 = theta2_2

    p3_1 = led_pos_img[0, 2] * math.cos(theta3_1) + led_pos_img[1, 2] * math.sin(theta3_1)
    p3_2 = led_pos_img[0, 2] * math.cos(theta3_2) + led_pos_img[1, 2] * math.sin(theta3_2)
    if abs(p3_1 - (led_pos_img[0, 3] * math.cos(theta3_1) + led_pos_img[1, 3] * math.sin(theta3_1))) < 1e-5:
        p3 = p3_1
        theta3 = theta3_1
    elif abs(p3_2 - (led_pos_img[0, 3] * math.cos(theta3_2) + led_pos_img[1, 3] * math.sin(theta3_2))) < 1e-5:
        p3 = p3_2
        theta3 = theta3_2

    p4_1 = led_pos_img[0, 3] * math.cos(theta4_1) + led_pos_img[1, 3] * math.sin(theta4_1)
    p4_2 = led_pos_img[0, 3] * math.cos(theta4_2) + led_pos_img[1, 3] * math.sin(theta4_2)
    if abs(p4_1 - (led_pos_img[0, 0] * math.cos(theta4_1) + led_pos_img[1, 0] * math.sin(theta4_1))) < 1e-5:
        p4 = p4_1
        theta4 = theta4_1
    elif abs(p4_2 - (led_pos_img[0, 0] * math.cos(theta4_2) + led_pos_img[1, 0] * math.sin(theta4_2))) < 1e-5:
        p4 = p4_2
        theta4 = theta4_2

    A_line = [f * math.cos(theta1), f * math.cos(theta2), f * math.cos(theta3), f * math.cos(theta4)]  # A12, A23, A34, A41
    B_line = [f * math.sin(theta1), f * math.sin(theta2), f * math.sin(theta3), f * math.sin(theta4)]  # B12, B23, B34, B41
    C_line = [p1, p2, p3, p4]  # C12, C23, C34, C41
    # 求解m、n
    b11 = B_line[0] * C_line[2] - B_line[2] * C_line[0]  # 中间变量
    b12 = C_line[0] * A_line[2] - C_line[2] * A_line[0]
    c1 = A_line[2] * B_line[0] - A_line[0] * B_line[2]

    b21 = B_line[1] * C_line[3] - B_line[3] * C_line[1]  # 中间变量
    b22 = C_line[1] * A_line[3] - C_line[3] * A_line[1]
    c2 = A_line[3] * B_line[1] - A_line[1] * B_line[3]

    A_LS = np.array([[b11, b12], [b21, b22]])
    b_LS = np.array([c1, c2])
    temp1 = np.dot(A_LS.T, A_LS)
    temp2 = np.dot(np.linalg.pinv(temp1), A_LS.T)
    mn = np.dot(temp2, b_LS)
    m = mn[0]
    n = mn[1]
    # 计算LED平面在camera坐标系中法向量
    cos_alpha = m / math.sqrt(m ** 2 + n ** 2 + 1)
    cos_beta = n / math.sqrt(m ** 2 + n ** 2 + 1)
    cos_gama = 1 / math.sqrt(m ** 2 + n ** 2 + 1)
    # 求4个顶点的camera坐标
    WE = np.array([[m, n, 1], [A_line[0], B_line[0], C_line[0]], [A_line[3], B_line[3], C_line[3]]])  # E点的W矩阵
    WF = np.array([[m, n, 1], [A_line[0], B_line[0], C_line[0]], [A_line[1], B_line[1], C_line[1]]])  # F点的W矩阵
    WG = np.array([[m, n, 1], [A_line[1], B_line[1], C_line[1]], [A_line[2], B_line[2], C_line[2]]])  # G点的W矩阵
    WH = np.array([[m, n, 1], [A_line[2], B_line[2], C_line[2]], [A_line[3], B_line[3], C_line[3]]])  # H点的W矩阵
    # 求行列式
    det_WE = np.linalg.det(WE)
    det_WF = np.linalg.det(WF)
    det_WG = np.linalg.det(WG)
    det_WH = np.linalg.det(WH)
    # 代数余子式
    AC_WE = det_WE * np.linalg.pinv(WE)
    AC_WF = det_WF * np.linalg.pinv(WF)
    AC_WG = det_WG * np.linalg.pinv(WG)
    AC_WH = det_WH * np.linalg.pinv(WH)
    # 求q值
    q1_6time = np.array([[AC_WE[0, 0] / det_WE, AC_WE[1, 0] / det_WE, AC_WE[2, 0] / det_WE],
                         [AC_WF[0, 0] / det_WF, AC_WF[1, 0] / det_WF, AC_WF[2, 0] / det_WF],
                         [AC_WG[0, 0] / det_WG, AC_WG[1, 0] / det_WG, AC_WG[2, 0] / det_WG]])
    q1 = 1 / 6 * abs(np.linalg.det(q1_6time))

    q2_6time = np.array([[AC_WF[0, 0] / det_WF, AC_WF[1, 0] / det_WF, AC_WF[2, 0] / det_WF],
                         [AC_WG[0, 0] / det_WG, AC_WG[1, 0] / det_WG, AC_WG[2, 0] / det_WG],
                         [AC_WH[0, 0] / det_WH, AC_WH[1, 0] / det_WH, AC_WH[2, 0] / det_WH]])
    q2 = 1 / 6 * abs(np.linalg.det(q2_6time))

    q3_6time = np.array([[AC_WG[0, 0] / det_WG, AC_WG[1, 0] / det_WG, AC_WG[2, 0] / det_WG],
                         [AC_WH[0, 0] / det_WH, AC_WH[1, 0] / det_WH, AC_WH[2, 0] / det_WH],
                         [AC_WE[0, 0] / det_WE, AC_WE[1, 0] / det_WE, AC_WE[2, 0] / det_WE]])
    q3 = 1 / 6 * abs(np.linalg.det(q3_6time))

    q4_6time = np.array([[AC_WH[0, 0] / det_WH, AC_WH[1, 0] / det_WH, AC_WH[2, 0] / det_WH],
                         [AC_WE[0, 0] / det_WE, AC_WE[1, 0] / det_WE, AC_WE[2, 0] / det_WE],
                         [AC_WF[0, 0] / det_WF, AC_WF[1, 0] / det_WF, AC_WF[2, 0] / det_WF]])
    q4 = 1 / 6 * abs(np.linalg.det(q4_6time))
    # 求k值
    k = math.sqrt(2 * led_area / (3 * (q1 + q2 + q3 + q4) * math.sqrt(m ** 2 + n ** 2 + 1)))

    # 求4个顶点的camera坐标
    p1_xyz = k * (np.dot(np.linalg.pinv(WE), [1, 0, 0]))
    p2_xyz = k * (np.dot(np.linalg.pinv(WF), [1, 0, 0]))
    p3_xyz = k * (np.dot(np.linalg.pinv(WG), [1, 0, 0]))
    p4_xyz = k * (np.dot(np.linalg.pinv(WH), [1, 0, 0]))

    # 绕x、y轴旋转角fai和theta
    theta_1 = math.asin(-cos_alpha)  # fai和theta的范围都是[-pi / 2, pi / 2], asin默认是这个范围不需要后面theta_2等的计算和判断
    fai_1 =  math.asin(cos_beta /  math.cos(theta_1))
    eq1 = cos_gama -  math.cos(fai_1) *  math.cos(theta_1)
    if eq1 < 1e-10:
        theta = theta_1
        fai = fai_1
    # 求未知量cos(psi),sin(psi),tx,ty,tz
    rot_matrix_x_est = np.array([[1, 0, 0], [0, math.cos(fai), -math.sin(fai)], [0, math.sin(fai), math.cos(fai)]])
    rot_matrix_y_est = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
    rox_matrix_yx = np.dot(rot_matrix_y_est, rot_matrix_x_est)
    a1 = rox_matrix_yx[0, 0]
    a2 = rox_matrix_yx[0, 1]
    a3 = rox_matrix_yx[0, 2]
    b1 = rox_matrix_yx[1, 0]
    b2 = rox_matrix_yx[1, 1]
    b3 = rox_matrix_yx[1, 2]
    c1 = rox_matrix_yx[2, 0]
    c2 = rox_matrix_yx[2, 1]
    c3 = rox_matrix_yx[2, 2]
    # 求tz
    cont1 = p1_xyz[0] * c1 + p1_xyz[1] * c2 + p1_xyz[2] * c3
    cont2 = p2_xyz[0] * c1 + p2_xyz[1] * c2 + p2_xyz[2] * c3
    cont3 = p3_xyz[0] * c1 + p3_xyz[1] * c2 + p3_xyz[2] * c3
    cont4 = p4_xyz[0] * c1 + p4_xyz[1] * c2 + p4_xyz[2] * c3
    cont = [cont1, cont2, cont3, cont4]  # 因为tz和LED的高度都是唯一的，因此4个常数都相同, 求均值
    tz = h - np.mean(cont)  # camera的z坐标
    # 利用Ax = b求解
    # 系数矩阵
    A1 = np.array([[a1 * p1_xyz[0] + a2 * p1_xyz[1] + a3 * p1_xyz[2], -b1 * p1_xyz[0] - b2 * p1_xyz[1] - b3 * p1_xyz[2], 1, 0],
                   [b1 * p1_xyz[0] + b2 * p1_xyz[1] + b3 * p1_xyz[2], a1 * p1_xyz[0] + a2 * p1_xyz[1] + a3 * p1_xyz[2], 0, 1]])
    A2 = np.array([[a1 * p2_xyz[0] + a2 * p2_xyz[1] + a3 * p2_xyz[2], -b1 * p2_xyz[0] - b2 * p2_xyz[1] - b3 * p2_xyz[2], 1, 0],
                   [b1 * p2_xyz[0] + b2 * p2_xyz[1] + b3 * p2_xyz[2], a1 * p2_xyz[0] + a2 * p2_xyz[1] + a3 * p2_xyz[2], 0, 1]])
    A3 = np.array([[a1 * p3_xyz[0] + a2 * p3_xyz[1] + a3 * p3_xyz[2], -b1 * p3_xyz[0] - b2 * p3_xyz[1] - b3 * p3_xyz[2], 1, 0],
                   [b1 * p3_xyz[0] + b2 * p3_xyz[1] + b3 * p3_xyz[2], a1 * p3_xyz[0] + a2 * p3_xyz[1] + a3 * p3_xyz[2], 0, 1]])
    A4 = np.array([[a1 * p4_xyz[0] + a2 * p4_xyz[1] + a3 * p4_xyz[2], -b1 * p4_xyz[0] - b2 * p4_xyz[1] - b3 * p4_xyz[2], 1, 0],
                   [b1 * p4_xyz[0] + b2 * p4_xyz[1] + b3 * p4_xyz[2], a1 * p4_xyz[0] + a2 * p4_xyz[1] + a3 * p4_xyz[2], 0, 1]])
    A_para = np.vstack((A1, A2, A3, A4))
    # 右侧常数矩阵
    b1_line = led_corner_gs[[0, 1], 0]
    b2_line = led_corner_gs[[0, 1], 1]
    b3_line = led_corner_gs[[0, 1], 2]
    b4_line = led_corner_gs[[0, 1], 3]

    # b的1, 2列做参数
    M_b1 = np.hstack((b1_line, b2_line))
    M_A1 = np.vstack((A_para[[0, 1], :], A_para[[2, 3], :]))
    temp1 = np.dot(M_A1.T, M_A1)
    temp2 = np.dot(np.linalg.pinv(temp1), M_A1.T)
    x_para_12 = np.dot(temp2, M_b1)

    # b的1, 3列做参数
    M_b2 = np.hstack((b1_line, b3_line))
    M_A2 = np.vstack((A_para[[0, 1], :], A_para[[4, 5], :]))
    temp1 = np.dot(M_A2.T, M_A2)
    temp2 = np.dot(np.linalg.pinv(temp1), M_A2.T)
    x_para_13 = np.dot(temp2, M_b2)

    # b的1, 4列做参数
    M_b3 = np.hstack((b1_line, b4_line))
    M_A3 = np.vstack((A_para[[0, 1], :], A_para[[6, 7], :]))
    temp1 = np.dot(M_A3.T, M_A3)
    temp2 = np.dot(np.linalg.pinv(temp1), M_A3.T)
    x_para_14 = np.dot(temp2, M_b3)

    # b的2, 3列做参数
    M_b4 = np.hstack((b2_line, b3_line))
    M_A4 = np.vstack((A_para[[2, 3], :], A_para[[4, 5], :]))
    temp1 = np.dot(M_A4.T, M_A4)
    temp2 = np.dot(np.linalg.pinv(temp1), M_A4.T)
    x_para_23 = np.dot(temp2, M_b4)

    # b的2, 4列做参数
    M_b5 = np.hstack((b2_line, b4_line))
    M_A5 = np.vstack((A_para[[2, 3], :], A_para[[6, 7], :]))
    temp1 = np.dot(M_A5.T, M_A5)
    temp2 = np.dot(np.linalg.pinv(temp1), M_A5.T)
    x_para_24 = np.dot(temp2, M_b5)

    # b的3, 4列做参数
    M_b6 = np.hstack((b3_line, b4_line))
    M_A6 = np.vstack((A_para[[4, 5], :], A_para[[6, 7], :]))
    temp1 = np.dot(M_A6.T, M_A6)
    temp2 = np.dot(np.linalg.pinv(temp1), M_A6.T)
    x_para_34 = np.dot(temp2, M_b6)

    x_para1 = (x_para_12 + x_para_13+x_para_14+x_para_23+x_para_24+x_para_34) / 6
    # 接收机位置
    receiver_pos_est = np.array([x_para1[2], x_para1[3], tz])

    cos_psi = x_para1[0]+0.1
    if cos_psi > 1:
        cos_psi = 1
    sin_psi = x_para1[1]
    psi1 = math.acos(cos_psi)
    if math.sin(psi1) == sin_psi:
         psi = psi1
    else:
         psi = -psi1
    # 输出结果
    m1 = np.array([theta, fai, psi])
    result_all = np.append(receiver_pos_est, m1)
    return [result_all[0], result_all[1]]
