# coding: utf-8
# Author：QiTang
# Date ：23/10/2022

import os
import cv2
import math
import numpy as np


def calibrate_location(image):
    # 定义需要返回的参数
    mouse_params = {'x': None, 'width': None, 'height': None,
                    'y': None, 'patch': None}
    cv2.namedWindow('image')
    # 鼠标框选操作函数
    cv2.setMouseCallback('image', on_mouse, mouse_params)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    return [mouse_params['x'], mouse_params['y'], mouse_params['width'],
            mouse_params['height']], mouse_params['patch']


def on_mouse(event, x, y, flags, param):
    global image, point
    img_tmp = image.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point = (x, y)
        cv2.circle(img_tmp, point, 10, (0, 255, 0), 5)
        cv2.imshow('image', img_tmp)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img_tmp, point, (x, y), (255, 0, 0), 5)
        cv2.imshow('image', img_tmp)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point_ = (x, y)
        cv2.rectangle(img_tmp, point, point_, (0, 0, 255), 5)
        cv2.imshow('image', img_tmp)
        # 返回框选矩形左上角点的坐标、矩形宽度、高度以及矩形包含的图像
        param['x'] = min(point[0], point_[0])
        param['y'] = min(point[1], point_[1])
        param['width'] = abs(point[0] - point_[0])
        param['height'] = abs(point[1] - point_[1])
        param['patch'] = image[param['y']:param['y'] + param['height'],
                         param['x']:param['x'] + param['width']]


def main():
    global image
    image = cv2.imread('data/Car_Data/car001.bmp')
    # 框选目标并返回相应信息
    location, patch = calibrate_location(image)
    (height, width, c) = patch.shape
    center = [height / 2, width / 2]
    # 计算目标图像的权值矩阵
    weight = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            distance = (i - center[0]) ** 2 + (j - center[1]) ** 2
            weight[i, j] = 1 - distance / (center[0] ** 2 + center[1] ** 2)
    # 计算目标权值直方图
    C = 1 / sum(sum(weight))
    hist1 = np.zeros(16 ** 3)
    for i in range(height):
        for j in range(width):
            q_b = math.floor(float(patch[i, j, 0]) / 16)
            q_g = math.floor(float(patch[i, j, 1]) / 16)
            q_r = math.floor(float(patch[i, j, 2]) / 16)
            q_temp1 = q_r * 256 + q_g * 16 + q_b
            hist1[int(q_temp1)] = hist1[int(q_temp1)] + weight[i, j]
    hist1 = hist1 * C
    # 读取视频并进行目标跟踪
    for pic_i in range(2, 101):
        if pic_i < 10:
            pic_name = "car00%d.bmp" % pic_i
        elif 100 > pic_i >= 10:
            pic_name = "car0%d.bmp" % pic_i
        else:
            pic_name = "car%d.bmp" % pic_i
        path = os.path.join(r"data/Car_Data/", pic_name)
        frame = cv2.imread(path)
        num = 0
        offset = [1, 1]
        # mean shift 迭代
        while (np.sqrt(offset[0] ** 2 + offset[1] ** 2) > 0.5) & (num < 20):
            num = num + 1
            # 计算候选区域直方图
            patch2 = frame[int(location[1]):int(location[1] + location[3]),
                     int(location[0]):int(location[0] + location[2])]
            hist2 = np.zeros(16 ** 3)
            q_temp2 = np.zeros((height, width))
            for i in range(height):
                for j in range(width):
                    q_b = math.floor(float(patch2[i, j, 0]) / 16)
                    q_g = math.floor(float(patch2[i, j, 1]) / 16)
                    q_r = math.floor(float(patch2[i, j, 2]) / 16)
                    q_temp2[i, j] = q_r * 256 + q_g * 16 + q_b
                    hist2[int(q_temp2[i, j])] = hist2[int(q_temp2[i, j])] + weight[i, j]
            hist2 = hist2 * C
            w = np.zeros(16 ** 3)
            for i in range(16 ** 3):
                if hist2[i] != 0:
                    w[i] = math.sqrt(hist1[i] / hist2[i])
                else:
                    w[i] = 0
            sum_w = 0
            sum_xw = [0, 0]
            for i in range(height):
                for j in range(width):
                    sum_w = sum_w + w[int(q_temp2[i, j])]
                    sum_xw = sum_xw + w[int(q_temp2[i, j])] * np.array([i - center[0], j - center[1]])
            offset = sum_xw / sum_w
            # 位置更新
            location[0] = location[0] + offset[1]
            location[1] = location[1] + offset[0]
        x = int(location[0])
        y = int(location[1])
        width = int(location[2])
        height = int(location[3])
        pt1 = (x, y)
        pt2 = (x + width, y + height)

        image_with_rect = cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
        cv2.imshow('Car Tracker', image_with_rect)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break


if __name__ == '__main__':
    main()
