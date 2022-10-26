# coding: utf-8
# Author：QiTang
# Date ：24/10/2022

import os
import cv2
import math
import numpy as np
import time as T
from sklearn.svm import SVC
from skimage.feature import hog
import matplotlib.pyplot as plt

global location


def calibrate_location(image):
    # 定义需要返回的参数
    mouse_params = {'x': None, 'width': None, 'height': None,
                    'y': None, 'patch': None}
    cv2.namedWindow('image')
    # 鼠标框选操作函数
    cv2.setMouseCallback('image', on_mouse, mouse_params)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    return (mouse_params['y'], mouse_params['x'], mouse_params['height'],
            mouse_params['width']), mouse_params['patch']


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

        padding = 12
        y = param['y'] - padding
        x = param['x'] - padding
        height = param['height'] + 2 * padding
        width = param['width'] + 2 * padding
        param['patch'] = image[y:y + height, x:x + width]


def hog_descriptor(image, pixels_per_cell, visualize):
    if visualize:
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=pixels_per_cell, cells_per_block=(1, 1),
                            feature_vector=False, visualize=True, channel_axis=-1)
        return fd, hog_image
    else:
        fd = hog(image, orientations=8, pixels_per_cell=pixels_per_cell, cells_per_block=(1, 1), feature_vector=False)
        return fd


def calculate_features(image, pixels_per_cell):
    feature_dims = 11
    image_copy = image.copy()
    b, g, r = cv2.split(image_copy)
    height, width, channel = image_copy.shape
    # print("height,width", height, width)
    strides = image_copy.itemsize * np.array([width, 1, height, 1])
    new_height = height - (pixels_per_cell - 1)
    new_width = width - (pixels_per_cell - 1)
    # print("new_height,new_width", new_height, new_width)

    image_slides = np.lib.stride_tricks.as_strided(image_copy,
                                                   shape=(new_height, new_width, pixels_per_cell, pixels_per_cell),
                                                   strides=strides)
    # print(image_slides.shape)
    features = np.zeros((new_height, new_width, feature_dims))
    for i in range(0, new_height):
        for j in range(0, new_width):
            # gray_img = cv2.cvtColor(image_slides[i, j], cv2.COLOR_BGR2GRAY)
            fd = hog_descriptor(image_slides[i, j], (pixels_per_cell, pixels_per_cell), False)
            feature = np.concatenate((fd[0][0][0][0], np.array([r[i + 2, j + 2], g[i + 2, j + 2], b[i + 2, j + 2]])),
                                     axis=0)

            features[i, j] = np.array(feature)

    return features


def generate_labels(features, top_y, top_x, n_rows, n_cols):
    h, w, d = features.shape
    labels = np.zeros((h, w, 1))

    for i in range(0, h):
        for j in range(0, w):
            if i < 10 or j < 10:
                labels[i, j] = -1
            elif i >= n_rows + 10 or j >= n_cols + 10:
                labels[i, j] = -1
            else:
                labels[i, j] = 1

    return labels


def calculate_err(w, h, y):
    return np.sum(w * np.abs(h - y))


def svm(features, labels):
    clf = SVC()
    clf.fit(features, labels)
    return clf


def weak_classifier(features, labels, w):
    features_flatten = np.reshape(features, (-1, 11))
    labels_flatten = labels.ravel()

    clf = svm(features_flatten, labels_flatten)

    err = calculate_err(w, clf.predict(features_flatten), labels_flatten)
    # print("err", err)
    alpha = 0.5 * math.log(10, abs(1 - err) / err)

    w = w * np.exp(alpha * np.abs(clf.predict(features_flatten) - labels_flatten))

    return clf, alpha, w


def adaboost(features, labels, top_y, top_x, n_rows, n_cols):
    global location
    num_of_classifiers = 5
    classifiers = []
    alpha_array = []
    h_array = []
    err_array = []

    w0 = np.full(labels.ravel().shape, 1 / (np.reshape(features, (-1, 11))).shape[0])
    w = w0.copy()

    for index in range(0, num_of_classifiers):
        clf, alpha, w = weak_classifier(features, labels, w)
        print("alpha", alpha)
        classifiers.append(clf)
        alpha_array.append(alpha)

    total_time = 0
    for pic_i in range(2, 127):
        start = T.time()

        if pic_i < 10:
            pic_name = "00%d.bmp" % pic_i
        elif 100 > pic_i >= 10:
            pic_name = "0%d.bmp" % pic_i
        else:
            pic_name = "%d.bmp" % pic_i

        path = os.path.join(r"./data/football/", pic_name)
        image = cv2.imread(path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if pic_i == 2:
            padding = 12
            top_y -= padding
            top_x -= padding
            n_rows += 2 * padding
            n_cols += 2 * padding
        image_cut = image[top_y:top_y + n_rows, top_x:top_x + n_cols]

        features = calculate_features(image_cut, 5)
        labels = generate_labels(features, *location)

        features_flatten = np.reshape(features, (-1, 11))
        labels_flatten = labels.ravel()

        prediction = np.zeros(labels_flatten.shape)
        h_array.clear()
        for _, clf in enumerate(classifiers):
            h = clf.predict(features_flatten)
            h_array.append(h)
            prediction += alpha_array[_] * h

        prediction[prediction < 0] = 0
        prediction = (prediction - np.min(prediction)) / np.ptp(prediction)
        height, width, channel = features.shape
        c_map = np.reshape(prediction * 255, (height, width))

        cv2.imwrite("./c_map/" + str(pic_i) + ".png", c_map)

        # cv2.imshow("confidence map", c_map)
        # cv2.waitKey(0)

        (top_y, top_x, n_rows, n_cols) = mean_shift(image, c_map, *location)

        location = (top_y, top_x, n_rows, n_cols)

        image_with_rect = cv2.rectangle(image, (location[1], location[0]),
                                        (location[1] + location[3], location[0] + location[2]), (0, 0, 255), 2)
        cv2.imwrite("./save_football/" + str(pic_i) + ".png", image_with_rect)

        padding = 12
        top_y -= padding
        top_x -= padding
        n_rows += 2 * padding
        n_cols += 2 * padding
        image_cut = image[top_y:top_y + n_rows, top_x:top_x + n_cols]

        features = calculate_features(image_cut, 5)
        labels = generate_labels(features, *location)

        labels_flatten = labels.ravel()

        del classifiers[0]
        del alpha_array[0]
        del h_array[0]

        err_array.clear()
        for h in h_array:
            err_array.append(calculate_err(w0, h, labels_flatten))

        w = w0.copy()
        while len(err_array) > 0:
            idx = np.argmin(np.array(err_array))
            alpha = 0.5 * math.log(10, abs(1 - err_array[idx]) / err_array[idx])
            w = w * np.exp(alpha * np.abs(h_array[idx] - labels_flatten))

            alpha_array[idx] = alpha
            del err_array[idx]

        clf, alpha, w = weak_classifier(features, labels, w)
        print(pic_i, "alpha", alpha)
        classifiers.append(clf)
        alpha_array.append(alpha)

        cost = T.time() - start
        total_time += cost

        print("单张耗时:", total_time / (pic_i - 1), "每秒处理:", (pic_i - 1) / (total_time))


def mean_shift(image, cur_frame, top_y, top_x, n_rows, n_cols):
    temp = cur_frame[10:n_rows + 10, 10:n_cols + 10]
    (height, width) = temp.shape
    center = [height / 2, width / 2]

    # 计算目标图像的权值矩阵
    weight = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            z = (i - center[0]) ** 2 + (j - center[1]) ** 2
            weight[i, j] = 1 - z / (center[0] ** 2 + center[1] ** 2)

    # 计算目标权值直方图
    C = 1 / sum(sum(weight))
    hist1 = np.zeros(16 ** 3)
    for i in range(height):
        for j in range(width):
            q_temp1 = math.floor(float(temp[i, j]) / 16)
            hist1[int(q_temp1)] = hist1[int(q_temp1)] + weight[i, j]
    hist1 = hist1 * C

    rect = [10, 10, width, height]

    num = 0
    offset = [1, 1]

    # mean shift迭代
    while (np.sqrt(offset[0] ** 2 + offset[1] ** 2) > 0.5) & (num < 20):
        num = num + 1

        # 计算候选区域直方图
        temp2 = cur_frame[int(rect[1]):int(rect[1] + rect[3]), int(rect[0]):int(rect[0] + rect[2])]
        hist2 = np.zeros(16 ** 3)
        q_temp2 = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                # print("=" * 20, q_temp2.shape, temp2.shape)
                if i >= temp2.shape[0]:
                    break
                if j >= temp2.shape[1]:
                    break
                q_temp2[i, j] = math.floor(float(temp2[i, j]) / 16)
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
        rect[0] = rect[0] + offset[1]
        rect[1] = rect[1] + offset[0]

    x = int(rect[0] + top_x)
    y = int(rect[1] + top_y)
    width = int(rect[2])
    height = int(rect[3])

    # show_result(image, [x, y], height, width)

    return (y, x, height, width)


def show_result(img, point_top, n_rows, n_cols):
    plt.imshow(img, cmap='gray')
    ax = plt.gca()
    ax.add_patch(plt.Rectangle((point_top[0], point_top[1]), n_cols, n_rows, color="blue", fill=False, linewidth=1))

    plt.show()
    plt.pause(0.2)
    plt.close()


def main():
    global image, location
    image = cv2.imread('data/football/050.bmp')
    location, patch = calibrate_location(image)

    image_with_rect = cv2.rectangle(image, (location[1], location[0]),
                                    (location[1] + location[3], location[0] + location[2]), (0, 0, 255), 2)
    cv2.imwrite("./save_football/" + str(1) + ".png", image_with_rect)

    features = calculate_features(patch, 5)
    labels = generate_labels(features, *location)
    adaboost(features, labels, *location)


if __name__ == "__main__":
    main()
