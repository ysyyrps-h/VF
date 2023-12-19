import numpy as np
import math


def ellipsefit4(ledmark):
    x = ledmark[0]
    y = ledmark[1]
    temp = x ** 2
    temp1 = x * y
    temp2 = y ** 2
    a = np.zeros((5, 1), dtype=float)
    D1 = np.array([temp, temp1, temp2]).T
    D2 = np.array([x, y, np.ones(len(x))]).T
    S1 = np.dot(D1.T, D1)
    S2 = np.dot(D1.T, D2)
    S3 = np.dot(D2.T, D2)
    T = -np.dot(np.linalg.pinv(S3), S2.T)
    M = S1 + np.dot(S2, T)
    C1 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]])
    M = np.dot(np.linalg.pinv(C1), M)
    e, v = np.linalg.eig(M)
    cond = 4 * v[0, :] * v[2, :] - v[1, :] ** 2
    if cond.all() < 0:
        a = np.zeros((1, 5), dtype=float)
    else:
        a1 = v[:, cond > 0]
        a2 = np.vstack((a1, np.dot(T, a1)))
        a = a2[0:5] / a2[5]
    return a


def Ellipse(id_list, contours, flag):
    h = 107  # 实验平台高度
    led_gs = np.array([[0, 30, h], [0, 0, h], [30, 30, h], [30, 0, h], [60, 30, h], [60, 0, h]]).T  # LED的物理坐标
    led_corner_gs = np.zeros((3, 2), dtype=int)
    theta = np.zeros((2, 1), dtype=float)
    fai = np.zeros((2, 1), dtype=float)
    psi = np.zeros((1, 2), dtype=float)
    judgepsi1 = np.zeros((1, 2), dtype=float)
    judgepsi2 = np.zeros((1, 2), dtype=float)
    led_corner_gs[:, 0] = led_gs[:, id_list - 1]
    for i in range(2):  # led投影点的像素坐标，通过相机获取图片后，2x4 double
        led_corner_gs[:, i] = led_gs[:, id_list - 1]  # LED在世界坐标系中的位置坐标
    a = led_corner_gs[:, 1] - led_corner_gs[:, 0]
    b = np.linalg.norm(a)
    G2P_nv_temp = a / b
    rot_angle_cos = math.acos(np.dot([0, 1, 0], G2P_nv_temp) / np.linalg.norm([0, 1, 0]) / np.linalg.norm(G2P_nv_temp))
    if led_corner_gs[0, 1] < led_corner_gs[0, 0]:
        rot_angle_cos = -rot_angle_cos
    rot_mat_z_final = np.array([[math.cos(rot_angle_cos), -math.sin(rot_angle_cos), 0],
                                [math.sin(rot_angle_cos), math.cos(rot_angle_cos), 0], [0, 0, 1]])
    # led_G_gs_final = np.dot(rot_mat_z_final, led_corner_gs)
    #R = 3.2016
    R = 5.5
    f = 0.3462  # 焦距为3000um = 3mm = 0.3cm
    dx = 0.000112
    dy = 0.000112  # 图像中像素点的物理长度352.78 * 352.78um
    fx = f / dx
    fy = f / dy
    if flag == 0:
        u0 = 1560
        v0 = 2080  # 主点
    else:
        u0 = 2080
        v0 = 1560  # 主点
    A = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]])
    led_img_rs_noisy = []
    rot_mat_trans2rs_temp = []
    led_img_trans_temp = []
    led_nv_trans_temp = []
    k_cut_temp = []
    b_led_temp = []
    n_temp = []
    for i in range(2):
        con = np.array(contours[i])
        led_img_rs_noisy_temp = np.zeros((3, len(con[0])), dtype=float)

        for iled in range(len(con[0])):
            temp = np.array([con[0][iled], con[1][iled], 1])
            m = np.linalg.pinv(A)
            temp1 = h * np.dot(m, temp)  # 这里得到camera坐标
            led_img_rs_noisy_temp[:, iled] = temp1[:] / temp1[2] * f
        led_img_rs_noisy.append(led_img_rs_noisy_temp)
        ledmk1 = led_img_rs_noisy_temp
        ledmk = np.array([row[1:] for row in ledmk1])
        Coe_elli_img = ellipsefit4(ledmk)
        Coe_cone_rs_mat = np.array(
            [[Coe_elli_img[0][0] * f ** 2, 0.5 * Coe_elli_img[1][0] * f ** 2, 0.5 * Coe_elli_img[3][0] * f],
             [0.5 * Coe_elli_img[1][0] * f ** 2, Coe_elli_img[2][0] * f ** 2, 0.5 * Coe_elli_img[4][0] * f],
             [0.5 * Coe_elli_img[3][0] * f, 0.5 * Coe_elli_img[4][0] * f, 1]])
        e1, v1 = np.linalg.eig(Coe_cone_rs_mat)
        ind = np.argsort(e1)
        e1 = e1[ind]
        v1 = v1[:, ind]
        rot_mat_trans2rs = v1
        rot_mat_rs2trans = rot_mat_trans2rs.T
        led_img_trans = np.dot(rot_mat_rs2trans, led_img_rs_noisy_temp)
        led_Gi_trans = led_img_trans[:, 0]  # 图像中的led圆心在过渡坐标系中坐标
        temp_sign = np.sign(e1)  # 判断λ的符号正负
        if temp_sign[2] != temp_sign[0] and temp_sign[2] != temp_sign[1]:
            temp_min = min(abs(e1[0:2]))
            B = np.argmin(abs(e1[0:2]))
            temp_max = max(abs(e1[0:2]))
            k_cut1 = math.sqrt(abs(e1[0] - e1[1]) / (temp_min + abs(e1[2])))  # 圆截面AB投影斜率
            k_cut2 = -k_cut1
            k_cut = np.array([k_cut1, k_cut2])
            led_Ge_trans = [0, 0, led_Gi_trans[2]]  # 正底面中心坐标
            b_cut = np.array([led_Ge_trans[2], led_Ge_trans[2]])  # 圆截面AB与z轴的交点，用过渡坐标系下圆切面圆心的坐标代入
            k_cone1 = math.sqrt(temp_max / abs(e1[2]))  # 椭圆锥投影的两条直线
            k_cone2 = -k_cone1
            k_cone = [k_cone1, k_cone2]
            Mc = np.array([b_cut / (k_cone[0] - k_cut), k_cone[0] * b_cut / (k_cone[0] - k_cut)])
            Nc = np.array([b_cut / (k_cone[1] - k_cut), k_cone[1] * b_cut / (k_cone[1] - k_cut)])
            r = np.array([np.linalg.norm(Mc[:, 0] - Nc[:, 0]) / 2, np.linalg.norm(Mc[:, 1] - Nc[:, 1]) / 2])
            b_led = R / r[0] * b_cut  # led平面投影与z轴的交点值
            if B == 1:  # XOZ投影
                led_nv_trans = np.array([[k_cut[0], k_cut[1]], [0, 0], [-1, -1]])
                n = 1
            elif B == 0:  # YOZ投影
                led_nv_trans = np.array([[0, 0], [k_cut[0], k_cut[1]], [-1, -1]])
                n = 2
            kk = 1
        if temp_sign[0] != temp_sign[2] and temp_sign[0] != temp_sign[1]:
            temp_min = min(abs(e1[1:3]))
            B = np.argmin(abs(e1[1:3])) + 1
            temp_max = max(abs(e1[1:3]))
            k_cut1 = math.sqrt(abs(e1[1] - e1[2]) / (temp_min + abs(e1[0])))  # 圆截面AB投影斜率
            k_cut2 = -k_cut1
            k_cut = np.array([k_cut1, k_cut2])
            led_Ge_trans = [led_Gi_trans[0], 0, 0]  # 正底面中心坐标
            b_cut = np.array([led_Ge_trans[0], led_Ge_trans[0]])  # 圆截面AB与z轴的交点，用过渡坐标系下圆切面圆心的坐标代入
            k_cone1 = math.sqrt(temp_max / abs(e1[0]))  # 椭圆锥投影的两条直线
            k_cone2 = -k_cone1
            k_cone = [k_cone1, k_cone2]
            Mc = np.array([b_cut / (k_cone[0] - k_cut), k_cone[0] * b_cut / (k_cone[0] - k_cut)])
            Nc = np.array([b_cut / (k_cone[1] - k_cut), k_cone[1] * b_cut / (k_cone[1] - k_cut)])
            r = np.array([np.linalg.norm(Mc[:, 0] - Nc[:, 0]) / 2, np.linalg.norm(Mc[:, 1] - Nc[:, 1]) / 2])
            b_led = R / r[0] * b_cut  # led平面投影与z轴的交点值
            if B == 2:  # XOZ投影
                led_nv_trans = np.array([[-1, -1], [k_cut[0], k_cut[1]], [0, 0]])
                n = 3
            elif B == 1:  # YOZ投影
                led_nv_trans = np.array([[-1, -1], [0, 0], [k_cut[0], k_cut[1]]])
                n = 4
            kk = 2
        if temp_sign[1] != temp_sign[0] and temp_sign[1] != temp_sign[2]:
            temp2 = np.hstack((e1[0], e1[2]))
            temp_min = min(abs(temp2))
            B = np.argmin(abs(temp2)) * 2 - 1
            temp_max = max(abs(temp2))
            k_cut1 = math.sqrt(abs(e1[0] - e1[2]) / (temp_min + abs(e1[1])))  # 圆截面AB投影斜率
            k_cut2 = -k_cut1
            k_cut = np.array([k_cut1, k_cut2])
            led_Ge_trans = [0, led_Gi_trans[1], 0]  # 正底面中心坐标
            b_cut = np.array([led_Ge_trans[1], led_Ge_trans[1]])  # 圆截面AB与z轴的交点，用过渡坐标系下圆切面圆心的坐标代入
            k_cone1 = math.sqrt(temp_max / abs(e1[1]))  # 椭圆锥投影的两条直线
            k_cone2 = -k_cone1
            k_cone = [k_cone1, k_cone2]
            Mc = np.array([b_cut / (k_cone[0] - k_cut), k_cone[0] * b_cut / (k_cone[0] - k_cut)])
            Nc = np.array([b_cut / (k_cone[1] - k_cut), k_cone[1] * b_cut / (k_cone[1] - k_cut)])
            r = np.array([np.linalg.norm(Mc[:, 0] - Nc[:, 0]) / 2, np.linalg.norm(Mc[:, 1] - Nc[:, 1]) / 2])
            b_led = R / r[0] * b_cut  # led平面投影与z轴的交点值
            if B == 1:  # XOZ投影
                led_nv_trans = np.array([[k_cut[0], k_cut[1]], [-1, -1], [0, 0]])
                n = 5
            elif B == -1:  # YOZ投影
                led_nv_trans = np.array([[0, 0], [-1, -1], [k_cut[0], k_cut[1]]])
                n = 6
            kk = 3
        rot_mat_trans2rs_temp.append(rot_mat_trans2rs)
        led_img_trans_temp.append(led_img_trans)
        led_nv_trans_temp.append(led_nv_trans)
        k_cut_temp.append(k_cut)
        b_led_temp.append(b_led)
        n_temp.append(n)
    # 排除二义性
    led_nv_rs_temp1 = np.dot(rot_mat_trans2rs_temp[0],
                             led_nv_trans_temp[0] / np.linalg.norm(led_nv_trans_temp[0][:, 1]))
    led_nv_rs_temp2 = np.dot(rot_mat_trans2rs_temp[1],
                             led_nv_trans_temp[1] / np.linalg.norm(led_nv_trans_temp[1][:, 1]))
    a = abs(led_nv_rs_temp1)
    b = abs(led_nv_rs_temp2)
    t1 = abs(a[:, 0] - b[:, 0])
    t2 = abs(a[:, 1] - b[:, 1])
    t3 = abs(a[:, 0] - b[:, 1])
    t4 = abs(a[:, 1] - b[:, 0])
    diff = np.array([np.linalg.norm(t1), np.linalg.norm(t2), np.linalg.norm(t3), np.linalg.norm(t4)])
    Bnv = np.argmin(diff)
    rot_mat_trans2rs = rot_mat_trans2rs_temp[0]
    rot_mat_trans2rs1 = rot_mat_trans2rs_temp[1]
    led_img_trans = led_img_trans_temp[0]
    led_img_trans1 = np.dot(rot_mat_trans2rs_temp[0].T, led_img_rs_noisy[1])
    n_case = n_temp[0]
    n_case1 = n_temp[1]
    led_G_gs = led_corner_gs[:, 0]
    led_G1_gs = led_corner_gs[:, 1]
    if Bnv < 2:
        led_nv_rs_est_temp = -np.sign(led_nv_rs_temp1[2, Bnv]) * led_nv_rs_temp1[:, Bnv]
        irr_judge = np.dot(-led_img_rs_noisy[0][:, 0], led_nv_rs_est_temp) / np.linalg.norm(
            -led_img_rs_noisy[0][:, 0]) / np.linalg.norm(led_nv_rs_est_temp)
        led_nv_rs_est = np.sign(irr_judge) * led_nv_rs_est_temp  # 用辐射角判断向量方向是否正确
        led_nv_rs_est_temp1 = -np.sign(led_nv_rs_temp2[2, Bnv]) * led_nv_rs_temp2[:, Bnv]
        irr_judge1 = np.dot(-led_img_rs_noisy[1][:, 0], led_nv_rs_est_temp1) / np.linalg.norm(
            -led_img_rs_noisy[1][:, 0]) / np.linalg.norm(led_nv_rs_est_temp1)
        led_nv_rs_est1 = np.sign(irr_judge1) * led_nv_rs_est_temp1  # 用辐射角判断向量方向是否正确
        k = k_cut_temp[0][Bnv]
        k1 = k_cut_temp[1][Bnv]
        bled = b_led_temp[0][Bnv]
        bled1 = b_led_temp[1][Bnv]
    elif Bnv == 2:
        led_nv_rs_est_temp = -np.sign(led_nv_rs_temp1[2, 0]) * led_nv_rs_temp1[:, 0]
        irr_judge = np.dot(-led_img_rs_noisy[0][:, 0], led_nv_rs_est_temp) / np.linalg.norm(
            -led_img_rs_noisy[0][:, 0]) / np.linalg.norm(led_nv_rs_est_temp)
        led_nv_rs_est = np.sign(irr_judge) * led_nv_rs_est_temp  # 用辐射角判断向量方向是否正确
        led_nv_rs_est_temp1 = -np.sign(led_nv_rs_temp2[2, 1]) * led_nv_rs_temp2[:, 1]
        irr_judge1 = np.dot(-led_img_rs_noisy[1][:, 0], led_nv_rs_est_temp1) / np.linalg.norm(
            -led_img_rs_noisy[1][:, 0]) / np.linalg.norm(led_nv_rs_est_temp1)
        led_nv_rs_est1 = np.sign(irr_judge1) * led_nv_rs_est_temp1  # 用辐射角判断向量方向是否正确
        k = k_cut_temp[0][0]
        k1 = k_cut_temp[1][1]
        bled = b_led_temp[0][0]
        bled1 = b_led_temp[1][1]
    elif Bnv == 3:
        led_nv_rs_est_temp = -np.sign(led_nv_rs_temp1[2, 1]) * led_nv_rs_temp1[:, 1]
        irr_judge = np.dot(-led_img_rs_noisy[0][:, 0], led_nv_rs_est_temp) / np.linalg.norm(
            -led_img_rs_noisy[0][:, 0]) / np.linalg.norm(led_nv_rs_est_temp)
        led_nv_rs_est = np.sign(irr_judge) * led_nv_rs_est_temp  # 用辐射角判断向量方向是否正确
        led_nv_rs_est_temp1 = -np.sign(led_nv_rs_temp2[2, 0]) * led_nv_rs_temp2[:, 0]
        irr_judge1 = np.dot(-led_img_rs_noisy[1][:, 0], led_nv_rs_est_temp1) / np.linalg.norm(
            -led_img_rs_noisy[1][:, 0]) / np.linalg.norm(led_nv_rs_est_temp1)
        led_nv_rs_est1 = np.sign(irr_judge1) * led_nv_rs_est_temp1  # 用辐射角判断向量方向是否正确
        k = k_cut_temp[0][1]
        k1 = k_cut_temp[1][0]
        bled = b_led_temp[0][1]
        bled1 = b_led_temp[1][0]
    if n_case == 1:
        led_G_trans_est = bled / (led_img_trans[2, 0] / led_img_trans[0, 0] - k) * np.array([1,
                                                                                             led_img_trans[1, 0] /
                                                                                             led_img_trans[0, 0],
                                                                                             led_img_trans[2, 0] /
                                                                                             led_img_trans[0, 0]])
        led_G1_trans_est = bled / (led_img_trans1[2, 0] / led_img_trans1[0, 0] - k) * np.array([1,
                                                                                                  led_img_trans1[1, 0] /
                                                                                                  led_img_trans1[0, 0],
                                                                                                  led_img_trans1[2, 0] /
                                                                                                  led_img_trans1[0, 0]])
    elif n_case == 2:
        led_G_trans_est = bled / (led_img_trans[2, 0] / led_img_trans[1, 0] - k) * np.array(
            [led_img_trans[0, 0] / led_img_trans[1, 0], 1,
             led_img_trans[2, 0] / led_img_trans[1, 0]])
        led_G1_trans_est = bled / (led_img_trans1[2, 0] / led_img_trans1[1, 0] - k) * np.array(
            [led_img_trans1[0, 0] / led_img_trans1[1, 0], 1,
             led_img_trans1[2, 0] / led_img_trans1[1, 0]])
    elif n_case == 3:
        led_G_trans_est = bled / (led_img_trans[1, 0] / led_img_trans[0, 0] - k) * np.array([1,
                                                                                             led_img_trans[1, 0] /
                                                                                             led_img_trans[0, 0],
                                                                                             led_img_trans[2, 0] /
                                                                                             led_img_trans[0, 0]])
        led_G1_trans_est = bled / (led_img_trans1[1, 0] / led_img_trans1[0, 0] - k) * np.array([1,
                                                                                                  led_img_trans1[1, 0] /
                                                                                                  led_img_trans1[0, 0],
                                                                                                  led_img_trans1[2, 0] /
                                                                                                  led_img_trans1[0, 0]])
    elif n_case == 4:
        led_G_trans_est = bled / (1 - k * led_img_trans[2, 0] / led_img_trans[0, 0]) * np.array([1,
                                                                                                 led_img_trans[1, 0] /
                                                                                                 led_img_trans[0, 0],
                                                                                                 led_img_trans[2, 0] /
                                                                                                 led_img_trans[0, 0]])
        led_G1_trans_est = bled / (1 - k * led_img_trans1[2, 0] / led_img_trans1[0, 0]) * np.array([1,
                                                                                                      led_img_trans1[
                                                                                                          1, 0] /
                                                                                                      led_img_trans1[
                                                                                                          0, 0],
                                                                                                      led_img_trans1[
                                                                                                          2, 0] /
                                                                                                      led_img_trans1[
                                                                                                          0, 0]])
    elif n_case == 5:
        led_G_trans_est = bled / (1 - k * led_img_trans[1, 0] / led_img_trans[0, 0]) * np.array([1,
                                                                                                 led_img_trans[1, 0] /
                                                                                                 led_img_trans[0, 0],
                                                                                                 led_img_trans[2, 0] /
                                                                                                 led_img_trans[0, 0]])
        led_G1_trans_est = bled / (1 - k * led_img_trans1[1, 0] / led_img_trans1[0, 0]) * np.array([1,
                                                                                                      led_img_trans1[
                                                                                                          1, 0] /
                                                                                                      led_img_trans1[
                                                                                                          0, 0],
                                                                                                      led_img_trans1[
                                                                                                          2, 0] /
                                                                                                      led_img_trans1[
                                                                                                          0, 0]])
    elif n_case == 6:
        led_G_trans_est = bled / (1 - k * led_img_trans[2, 0] / led_img_trans[1, 0]) * np.array(
            [led_img_trans[0, 0] / led_img_trans[1, 0], 1,
             led_img_trans[2, 0] / led_img_trans[1, 0]])
        led_G1_trans_est = bled / (1 - k * led_img_trans1[2, 0] / led_img_trans1[1, 0]) * np.array(
            [led_img_trans1[0, 0] / led_img_trans1[1, 0], 1,
             led_img_trans1[2, 0] / led_img_trans1[1, 0]])
    led_G_rs_est = np.dot(rot_mat_trans2rs, led_G_trans_est)  # led标志点估计坐标，相机坐标系下的
    led_G1_rs_est = np.dot(rot_mat_trans2rs, led_G1_trans_est)
    # led_G_rs_est = np.dot(rot_mat_z_final, led_corner_gs)
    # 求旋转矩阵参数Rx Ry Rz rot_mat_rs2gs_est
    G2P_nv_rs_est = (led_G1_rs_est - led_G_rs_est) / np.linalg.norm(led_G1_rs_est - led_G_rs_est)  # 相机坐标系的led平面的平行向量
    theta[0] = math.asin(led_nv_rs_est[0])  # 求出是[-π / 2, π / 2]间的值
    if theta[0] > 0:
        theta[1] = math.pi - theta[0]
    else:
        theta[1] = -math.pi - theta[0]
    fai[0] = math.asin(-led_nv_rs_est[1] / math.cos(theta[0]))
    fai_sin = -led_nv_rs_est[1] / math.cos(theta[0])
    fai_cos = -led_nv_rs_est[2] / math.cos(theta[0])
    if fai[0] > 0 and fai_cos < 0:
        fai[0] = math.pi - fai[0]
    elif fai[0] < 0 and fai_cos < 0:
        fai[0] = -math.pi - fai[0]
    # 求z轴旋转角psi
    if G2P_nv_rs_est[0] / math.cos(theta[0]) > 1:
        psi[0, 0] = math.pi / 2
    elif G2P_nv_rs_est[0] / math.cos(theta[0]) < -1:
        psi[0, 0] = -math.pi / 2
    else:
        psi[0, 0] = math.asin(G2P_nv_rs_est[0] / math.cos(theta[0]))

    if psi[0, 0] > 0:
        psi[0, 1] = math.pi - psi[0, 0]
    elif psi[0, 0] < 0:
        psi[0, 1] = -math.pi - psi[0, 0]
    judgefai = abs(led_nv_rs_est[2] + math.cos(fai[0]) * math.cos(theta[0]))  # 判别求值是否正确
    judgepsi1[0, 0] = abs(
        math.cos(psi[0, 0]) * math.cos(fai[0]) + math.sin(psi[0, 1]) * math.sin(theta[0]) * math.sin(fai[0]) -
        G2P_nv_rs_est[1])
    judgepsi2[0, 0] = abs(
        -math.cos(psi[0, 0]) * math.sin(fai[0]) + math.sin(psi[0, 1]) * math.sin(theta[0]) * math.cos(fai[0]) -
        G2P_nv_rs_est[2])
    judgepsi1[0, 1] = abs(
        math.cos(psi[0, 1]) * math.cos(fai[0]) + math.sin(psi[0, 1]) * math.sin(theta[0]) * math.sin(fai[0]) -
        G2P_nv_rs_est[1])
    judgepsi2[0, 1] = abs(
        -math.cos(psi[0, 1]) * math.sin(fai[0]) + math.sin(psi[0, 1]) * math.sin(theta[0]) * math.cos(fai[0]) -
        G2P_nv_rs_est[2])
    if judgefai < 1e-7 and judgepsi1[0, 0] < 1e-7 and judgepsi2[0, 0] < 1e-7:  # 全都不满足的情况
        rot_theta = theta[0]
        rot_fai = fai[0]
        rot_psi = psi[0, 0]
    elif judgefai < 1e-7 and judgepsi1[0, 1] < 1e-7 and judgepsi2[0, 1] < 1e-7:
        rot_theta = theta[0]
        rot_fai = fai[0]
        rot_psi = psi[0, 1]
    rot_mat_x_est = np.array(
        [[1, 0, 0], [0, math.cos(rot_fai), -math.sin(rot_fai)], [0, math.sin(rot_fai), math.cos(rot_fai)]])
    rot_mat_y_est = np.array(
        [[math.cos(rot_theta), 0, math.sin(rot_theta)], [0, 1, 0], [-math.sin(rot_theta), 0, math.cos(rot_theta)]])
    rot_mat_z_est = np.array([[math.cos(rot_psi), -math.sin(rot_psi), 0], [math.sin(rot_psi), math.cos(rot_psi), 0],
                              [0, 0, 1]])  # 注意！旋转顺序和generate是相反的，旋转角度正负也有不同。
    rot_mat_rs2gs_est = np.dot(rot_mat_z_est, np.dot(rot_mat_y_est, rot_mat_x_est))
    rot_mat_rs2gs_est1 = np.dot(rot_mat_z_final.T, rot_mat_rs2gs_est)
    mark_GP_temp = np.dot(rot_mat_rs2gs_est1.T, np.vstack((led_G_gs, led_G1_gs)).T)
    tra_mat_rs2gs_est = 0.5 * (sum(mark_GP_temp.T) - sum(np.vstack(([led_G_rs_est, led_G1_rs_est]))))
    ca_gs_est = np.dot(rot_mat_rs2gs_est1, tra_mat_rs2gs_est)
    # ca_gs = np.dot(np.linalg.pinv(rot_mat_z_final), ca_gs_est)
    print(ca_gs_est)
    return [ca_gs_est[0], ca_gs_est[1]]
    # x1 = led_corner_gs[0, 0]
    # x2 = led_corner_gs[0, 1]
    # y1 = led_corner_gs[1, 0]
    # y2 = led_corner_gs[1, 1]
    # x1_pixel = led_pos_pixel[0, 0]
    # x2_pixel = led_pos_pixel[0, 1]
    # y1_pixel = led_pos_pixel[1, 0]
    # y2_pixel = led_pos_pixel[1, 1]
    # u0 = 2080
    # v0 = 1560
    # d1 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    # d2 = math.sqrt((x1_pixel - x2_pixel) ** 2 + (y1_pixel - y2_pixel) ** 2)
    # h1 = h2 * d1 / d2
    # m1 = math.sqrt((x1_pixel - u0) ** 2 + (y1_pixel - v0) ** 2)
    # m2 = math.sqrt((x2_pixel - u0) ** 2 + (y2_pixel - v0) ** 2)
    # R1 = math.sqrt(m1**2*h1**2/h2**2)
    # R2 = math.sqrt(m2**2*h1**2/h2**2)
    # d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    # A = (R1**2 - R2**2 + d**2) / (2 * d)
    # h = math.sqrt(R1**2 - A**2)
    # tmp1 = x1 + A * (x2 - x1) / d
    # tmp2 = y1 + A * (y2 - y1) / d
    # x3 = tmp1 - h * (y2 - y1) / d
    # y3 = tmp2 + h * (x2 - x1) / d
    # x4 = tmp1 + h * (y2 - y1) / d
    # y4 = tmp2 - h * (x2 - x1) / d
    # theta = -math.atan2((y2_pixel-y1_pixel), (x2_pixel-x1_pixel))
    # theta1 = theta/math.pi*180
    # t1 = -(x1_pixel * math.cos(theta) - y1_pixel * math.sin(theta))
    # t2 = -(x1_pixel * math.sin(theta) + y1_pixel * math.cos(theta))
    # a1_p = x1_pixel*math.cos(theta)-y1_pixel*math.sin(theta) + t1
    # b1_p = x1_pixel*math.sin(theta)+y1_pixel*math.cos(theta) + t2
    # a2_p = x2_pixel*math.cos(theta)-y2_pixel*math.sin(theta) + t1
    # b2_p = x2_pixel*math.sin(theta)+y2_pixel*math.cos(theta) + t2
    # ox = u0 * math.cos(theta)-v0*math.sin(theta) + t1
    # oy = u0 * math.sin(theta)+v0*math.cos(theta) + t2
    # if oy > 0:
    #     return [x3, y3]
    # return [x4, y4]
