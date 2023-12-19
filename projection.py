import numpy as np
import cv2
import json
import match
import scipy.io
import VO_position
import try_decoding
import VO_position_auto
def project(R,img1,img2):

    return 0
def ini_process(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 变成灰度图
    blur_image = cv2.blur(gray, (10, 10))  # 滤波
    # cv2.imwrite('blur_image.jpg', blur_image)
    # ret, binary = cv2.threshold(blur_image, 0, 255, cv2.THRESH_OTSU)  # 二值化，阈值要设的小一点
    ret, binary = cv2.threshold(blur_image, 220, 255, cv2.THRESH_BINARY)  # 二值化，阈值要设的小一点
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 找轮廓
    contours1 = []
    for ind, cont in enumerate(contours):
        if len(cont) > 100:
            contours1.append(cont)
    for ind,cont in enumerate(contours1):
        elps = cv2.fitEllipse(cont)
        cv2.ellipse(image, elps, (0, 0, 255))
    contours = contours1
    scipy.io.savemat('resultroll.mat', mdict={'contours': contours})
    print(len(contours))
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


def read_from_jason():
    result = []
    with open('points1/data.txt') as json_file:
        data = json.load(json_file)
        cnt = 0
        while (data.__contains__('KeyPoint_%d' % cnt)):
            pt = cv2.KeyPoint(x=data['KeyPoint_%d' % cnt][0]['x'], y=data['KeyPoint_%d' % cnt][1]['y'], size=data['KeyPoint_%d' % cnt][2]['size'])
            result.append(pt)
            cnt += 1
    return result


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


if __name__ == "__main__":
    # img_1 = cv.imread("/home/ubuntu/users/lirj/wj1e/guominan/VO_task_easyversion_2/dataset/1.png")
    # img_2 = cv.imread("/home/ubuntu/users/lirj/wj1e/guominan/VO_task_easyversion_2/dataset/2.png")
    img_1 =cv2.imread("image/fugai/20230705214154C.jpg")  # 读取图片
    img_2 = cv2.imread("image/fugai/20230705214525C.jpg")  # 读取图片
    # img_1 =cv2.imread("1.jpg")  # 读取图片
    # img_2 = cv2.imread("111.jpg")  # 读取图片
    #img_2 = cv.imread("/home/ubuntu/users/lirj/wj1e/guominan/VO_task_easyversion_2/dataset/0006.jpg")
    [led_boxes1, contours1] = ini_process(img_1)
    [led_boxes2, contours2] = ini_process(img_2)
    img_1=cv2.blur(img_1, (3, 3))  # 滤波
    img_2 = cv2.blur(img_2, (3, 3))  # 滤波
    # img_1 =cv2.imread("1.jpg")  # 读取图片
    # img_2 = cv2.imread("111.jpg")  # 读取图片
    # img_1 =cv2.imread("image/fugai/20230705213018C.jpg")  # 读取图片
    # img_2 = cv2.imread("image/fugai/20230705213935C.jpg")  # 读取图片
    # 图像匹配
    keypoint_1, keypoint_2, matches = match.find_feature_matches(img_1, img_2)
    print("共计匹配点:", len(matches))

    # 预测位姿
    # R, t = poes_estimation_2d2d(keypoint_1, keypoint_2, matches)
    R, t, h = match.poes_estimation_2d2d(keypoint_1, keypoint_2, matches)
    print("共计匹配点:", len(matches))
    result = cv2.drawMatches(img_1, keypoint_1, img_2, keypoint_2, matches, None)
    cv2.namedWindow("result", 0)
    cv2.resizeWindow("result", 816, 612)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    contours=contours1[0]
    contours2 = contours2[0]
    # projection=contours1[0]
    cen=1
    topid1 = 0
    topid2 = 1
    top = np.min(contours[0])
    for i in range(len(contours[0])):
        if cen == 1:
            if contours[0][i] == top:
                cen = 0
            else:
                cen = 1
        else:
            topid1 = i
            break
    cen = 1
    for i in range(len(contours[0])):
        if cen == 1:
            if contours[0][len(contours[0]) - i - 1] == top:
                cen = 0
            else:
                cen = 1
        else:
            topid2 = len(contours[0]) - i - 1
            break
    # top = np.min(contours[0])
    # for i in range(len(contours[0])):
    #     if cen == 1:
    #         if contours[0][i] == top:
    #             cen = 0
    #         else:
    #             cen = 1
    #     else:
    #         topid1 = i
    #         break
    # cen = 1
    # for i in range(len(contours[0])):
    #     if cen == 1:
    #         if contours[0][len(contours[0]) - i - 1] == top:
    #             cen = 0
    #         else:
    #             cen = 1
    #     else:
    #         topid2 = len(contours[0]) - i - 1
    #         break
    interval = round((topid1 + topid2) / 2)
    interval= topid2
    #projection=np.zeros((2,len(contours[0])))
    # for i in range(len(contours[0])):
    #     projection[0][i]=contours[0][i]/len(img_1[0])
    #     projection[1][i] = contours[1][i] / len(img_1)
    x=contours[0][interval]
    y=contours[1][interval]
    projection=[[x],[y],[1]]
    # z=np.ones((1, len(projection[0])))
    # projection=np.row_stack((projection,z))

    projection=np.dot(h,projection)
    projection=np.delete(projection,[2],axis=0)
    # projectioo=[]
    # for i in range(len(projection[0])):
    #     #projectioo.append((round((projection[0][i])*len(img_2[0])),round((projection[1][i])*len(img_2))))
    #     projectioo.append((round(projection[0][i]),  round(projection[1][i])))
    # for point in projectioo:
    #     cv2.circle(img_2, point, 10, (255, 0, 0), 3)  # 画出轮廓，可以省略
    projectioo=[round(projection[0][0]),round(projection[1][0])]
    minstance=np.power((contours2[0][0]-projectioo[0]),2)+np.power((contours2[0][0]-projectioo[0]),2)
    ind=0
    for i in range(len(contours2[0])):
        distance= np.power((contours2[0][i]-projectioo[0]),2)+np.power((contours2[0][i]-projectioo[0]),2)
        if distance>minstance:
            minstance=distance
            ind=i
    projectioo=[2*contours2[0][0]-contours2[0][ind],2*contours2[1][0]-contours2[1][ind]]
    cv2.circle(img_2, projectioo, 10, (255, 0, 0), 3)  # 画出轮廓，可以省略
    # cv2.imwrite('image.jpg', image)
    cv2.namedWindow("img", 0)
    cv2.resizeWindow("img", 816, 612)
    cv2.imshow("img", img_2)
    cv2.waitKey(0)
    id_listn = []  # 最终识别的LED列表
    center_listn = []  # 最终识别的LED中心列表    `
    contours_list = []  # 最终识别的轮廓列表
    data = np.load("points1/data.npz")
    kp1 = read_from_jason()
    id_listn = data["arr_0"]
    contours_listn = data["arr_1"]
    center_listn = data["arr_2"]
    id_list = []
    id_list.append(int(id_listn))
    center_list = []
    contours_list = []
    center_list.append(center_listn[0])
    # center_list.append(center_listn[0])
    center_list.append(center_listn[1])
    contours_list.append(contours_listn)
    # center_list.append(center_listn[1])
    # point_xy=data["arr_3"]
    pointsdata = np.load("points1/pointsdata.npz")
    des1 = pointsdata["arr_0"]
    err_corr_list1 = [0] * len(led_boxes1)  # 没有成功识别的LED列表
    flag1 = 0  # 手机横竖标志位 1表示横屏拍摄，0表·    示竖屏拍摄
    flag2 = 0  # 倾斜角度标志位 1表示180° 0表示0°
    # ################### 自适应阈值 #######################
    threshold1 = adap_threshold(led_boxes1, img_2)  # 自适应二值化阈值计算
    # ################### 解码 #######################
    #try_decoding.try_decode(img1, led_boxes1, threshold1, err_corr_list1, id_listn[0], center_list1, contours_list1, flag1,contours1)  # 尝试解码,找到旋转方向

    #    VO_position_auto.Vo_Frames(5, contours_list[0], center_list, flag1, 5, projection, [projection[0][0],projection[0][1]], flag2, R,[x,y], projectioo)
    VO_position_auto.Vo_Frames(5, contours_list[0], center_list, flag1, 5, contours2, [contours2[0][0],contours2[0][1]], flag2, R,[x,y], projectioo)