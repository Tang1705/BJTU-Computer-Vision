# coding: utf-8
# Author：Qi Tang
# Date ：26/09/2022

import os
import cv2
import time
import copy
import numpy as np
from PIL import Image
from gabor import runGabor
import matplotlib.pyplot as plt
from glcm_features import glcm_features
from mpl_toolkits.mplot3d import Axes3D


# 判断是否为图片格式的文件
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


# 读取图片
def load_image(path):
    if is_image_file(path):
        return np.array(Image.open(path), dtype=float)
    else:
        print("Image file format error.")


# 直方图均衡化
def histogram_equalization(image):
    image = image.astype(np.uint8)
    return cv2.equalizeHist(image).astype(np.float64)


# 图像灰度值量化
def image_requantize(image, gray_level):
    bins = np.linspace(0, 255, gray_level + 1)
    image = np.digitize(image, bins)
    image[image > gray_level] = gray_level
    image = image - 1
    return image


# 图像归一化
def image_normalization(image):
    return np.uint8(255.0 * (image - np.min(image)) / (np.max(image) - np.min(image)))


# 展示图像
def image_show(image):
    plt.imshow(image, cmap='gray')
    plt.show()


# glcm 统计特征三维可视化
def glcm_feature_show(x, y, z, img, title):
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
    ax.contourf(x, y, img, zdir='z', offset=np.max(z) + 0.5, cmap='gray')
    ax.set_title(title)
    plt.show()


# 纹理特征可视化
def texture_vector_show(image, texture, feat_keys):
    texture = texture.transpose(2, 0, 1)
    plt.figure(figsize=(10, 4.5))

    row = 2
    col = int((texture.shape[0] + 2) / row)
    offset = 1

    plt.subplot(row, col, offset)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.title('Texture Image')
    offset += 1

    for name in feat_keys:
        plt.subplot(row, col, offset)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.axis('off')
        plt.imshow(texture[offset - 2], cmap='gray')
        plt.title(name)
        offset += 1

    plt.tight_layout(pad=0.5)

    plt.show()


# 纹理图像分割结果可视化及保存
def cluster_result_show(image_name, image):
    image_save = image.copy()
    image_save = image_save / np.max(image) * 255.0
    if not os.path.exists("./segmentation_results/"):
        os.mkdir("./segmentation_results/")
    cv2.imwrite("./segmentation_results/" + image_name, image_save)
    plt.imshow(image, cmap='gray')
    plt.show()


# 均值滤波
def mean_filter(array, ker_s=10):
    re = np.zeros_like(array)
    W, H, f_dim = array.shape
    padding = (ker_s - 1) // 2
    array = np.pad(array, ((padding, padding), (padding, padding), (0, 0)), "edge")
    for w in range(padding, W + padding):
        for h in range(padding, H + padding):
            re[w - padding][h - padding] = np.mean(
                array[w - padding:w + padding, h - padding:h + padding, :].reshape((-1, f_dim)), axis=0)
    return re


# 计算灰度共生矩阵
def calculate_glcm(input_win, step_x, step_y, gray_level):
    w, h = input_win.shape
    ret = np.zeros([gray_level, gray_level])
    for index_h in range(h):
        for index_w in range(w):
            try:
                row = int(input_win[index_h][index_w])
                col = int(input_win[index_h + step_y][index_w + step_x])
                ret[row, col] += 1
            except:
                continue
    return ret


# 提取纹理特征
def extract_texture_representation(win_size, image, stride=(1, 1), gray_level=16, feat_keys=[]):
    print('................Calculate GLCM...................')
    feature_dim = len(feat_keys)
    width, height = image.shape
    glcm_width = (width - (win_size - 1)) // stride[0]
    glcm_height = (height - (win_size - 1)) // stride[1]

    strides = image.itemsize * np.array([width * stride[1], stride[0], width, 1])
    windowed_image = np.lib.stride_tricks.as_strided(image, shape=(glcm_width, glcm_height, win_size, win_size),
                                                     strides=strides)

    windowed_image = image_requantize(windowed_image, gray_level)

    print("sliding window:", windowed_image.shape)

    glcm_matrix = np.zeros((glcm_width, glcm_height, gray_level, gray_level))
    # image_measurement = np.zeros((glcm_width, glcm_height, 4))

    print('................Calculate Feature.................')
    start = time.time()
    matrix = glcm_width * glcm_height
    for index in range(matrix):
        w = index // glcm_width
        h = index % glcm_height
        mean_glcm = (calculate_glcm(input_win=windowed_image[w][h], step_x=1, step_y=0, gray_level=gray_level) +
                     calculate_glcm(input_win=windowed_image[w][h], step_x=0, step_y=1, gray_level=gray_level) +
                     calculate_glcm(input_win=windowed_image[w][h], step_x=1, step_y=1, gray_level=gray_level) +
                     calculate_glcm(input_win=windowed_image[w][h], step_x=-1, step_y=1,
                                    gray_level=gray_level)) / 4
        mean_glcm = mean_glcm / np.sum(mean_glcm)
        glcm_matrix[w][h] = mean_glcm

        # local_mean = np.mean(windowed_image[w][h])
        # local_variance = np.var(windowed_image[w][h])
        # sc = np.mean((windowed_image[w][h] - local_mean) ** 3)  # 计算偏斜度
        # ku = np.mean((windowed_image[w][h] - local_mean) ** 4) / pow(local_variance, 2)  # 计算峰度

        # image_measurement[w][h] = np.array([local_mean, local_variance,], dtype="float")

    glcm_matrix = glcm_matrix.transpose((2, 3, 0, 1))

    glcm_features_calc = glcm_features(glcm_matrix, gray_level)
    mean = glcm_features_calc.calculate_glcm_mean()
    variance = glcm_features_calc.calculate_glcm_variance()
    contrast = glcm_features_calc.calculate_glcm_contrast()
    dissimilarity = glcm_features_calc.calculate_glcm_dissimilarity()
    entropy = glcm_features_calc.calculate_glcm_entropy()
    energy = glcm_features_calc.calculate_glcm_energy()
    inertia = glcm_features_calc.calculate_glcm_inertia()
    corr = glcm_features_calc.calculate_glcm_correlation()
    auto_correlation = glcm_features_calc.calculate_glcm_auto_correlation()

    feat_dict = {"Mean": mean, "Variance": variance, "Contrast": contrast, "Dissimilarity": dissimilarity,
                 "Entropy": entropy, "Energy": energy, "Inertia": inertia, "Correlation": corr,
                 "Autocorrelation": auto_correlation}

    feat_list = []
    for name in feat_keys:
        feat_list.append(feat_dict[name])

    # glcm_texture_feats = np.array([mean, variance, entropy, energy, inertia, corr, auto_correlation], dtype="float")
    glcm_texture_feats = np.array(feat_list, dtype="float")
    glcm_texture_feats = glcm_texture_feats.transpose((1, 2, 0))
    glcm_texture_feats = mean_filter(glcm_texture_feats)
    print("glcm_texture_feats:", glcm_texture_feats.shape)
    means = np.mean(glcm_texture_feats.reshape((-1, feature_dim)), axis=0).reshape((1, 1, feature_dim)).repeat(
        glcm_width, 0).repeat(glcm_height, 1)
    stds = np.std(glcm_texture_feats.reshape((-1, feature_dim)), axis=0).reshape((1, 1, feature_dim)).repeat(glcm_width,
                                                                                                             0).repeat(
        glcm_height, 1)

    glcm_texture_feats = (glcm_texture_feats - means) / stds
    end = time.time()
    print('Code run time:', end - start)

    # for i in range(feature_dim):
    #     glcm_feature_show(np.arange(glcm_width), np.arange(glcm_height), glcm_texture_feats[:, :, i],
    #                       np.var(windowed_image.reshape((glcm_width, glcm_height, -1)) / np.max(windowed_image),
    #                              axis=-1),
    #                       title=feat_keys[i])

    # glcm_texture_feats = np.concatenate((image_measurement, glcm_texture_feats), axis=2)
    return glcm_texture_feats


# 计算欧氏距离
def calculate_distance(position, centers, image):
    pixel_features = image[position[0]][position[1]]
    distance = []
    for index in range(len(centers)):
        diff = (pixel_features - centers[index]) ** 2
        distance.append(np.sum(diff))
    return np.array(distance)


# 判断簇中心是否发生变化
def cluster_center_change(origin, current):
    return np.sum((origin - current) ** 2)


# K 均值聚类
def k_means(matrix, k):
    w, h, c = matrix.shape
    centers = np.array(
        [matrix[np.random.randint(low=0, high=w)][[np.random.randint(low=0, high=h)]][0] for i in range(k)])
    origin_centers = np.zeros_like(centers)
    clusters = np.zeros((w, h))
    while cluster_center_change(centers, origin_centers) != 0:
        for i in range(w):
            for j in range(h):
                distance = calculate_distance([i, j], centers, matrix)
                cluster = np.argmin(distance)
                clusters[i][j] = cluster
        origin_centers = copy.deepcopy(centers)
        for k in range(k):
            points = []
            for i in range(w):
                for j in range(h):
                    if clusters[i][j] == k:
                        points.append(matrix[i][j])
            centers[k] = np.mean(np.array(points), axis=0)
    return clusters


def main():
    default_algorithm = "glcm"
    image_cluster = {"Texture_mosaic_1": 2, "Texture_mosaic_2": 3, "Texture_mosaic_3": 4, "Texture_mosaic_4": 5}

    glcm_statistics = {
        "Texture_mosaic_1": ["Mean", "Variance", "Entropy", "Energy", "Inertia", "Correlation"],
        "Texture_mosaic_2": ["Mean", "Variance", "Entropy", "Energy", "Correlation", "Inertia"],
        "Texture_mosaic_3": ["Variance", "Entropy", "Inertia", "Autocorrelation", "Contrast", "Dissimilarity"],
        "Texture_mosaic_4": ["Mean", "Variance", "Inertia", "Autocorrelation", "Contrast", "Dissimilarity"]
    }

    image_dir = "./Texture_mosaic__data/"
    for directory, subdirectory, filenames in os.walk(image_dir):
        for image_name in filenames:
            if is_image_file(image_name):
                image_key = image_name.split(".")[0]
                cluster_num = image_cluster[image_key]
                image = load_image(image_dir + image_name)

                image = image_normalization(image)
                # equalization_image = histogram_equalization(image)
                # stack_image = np.hstack([image, equalization_image])
                image_show(image)

                if default_algorithm == "glcm":
                    glcm_texture_feats = extract_texture_representation(win_size=19, image=image, stride=(1, 1),
                                                                        gray_level=16,
                                                                        feat_keys=glcm_statistics[image_key])
                    texture_vector_show(image, glcm_texture_feats, glcm_statistics[image_key])

                    result = k_means(glcm_texture_feats, k=cluster_num)
                    cluster_result_show(image_name, result)
                else:
                    runGabor(image_dir + image_name, "./segmentation_results/" + image_name, cluster_num)


if __name__ == "__main__":
    main()
