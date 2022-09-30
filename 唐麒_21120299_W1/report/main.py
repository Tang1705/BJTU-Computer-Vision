# coding: utf-8
# Author：QiTang
# Date ：25/09/2022

import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


# 判断是否为图片文件
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


# 加载图片
def load_image(path):
    if is_image_file(path):
        return cv2.imread(path)
    else:
        print("Image file format error.")


# 绘制灰度直方图并去噪
def show_gray_level_histogram(image, bins=256, smooth_win=64):
    n, bins, patches = plt.hist(image.flatten(), bins=bins, density=True)
    plt.title("Gray Level Histogram of Origin Image")
    plt.show()

    bins_center = []
    for index in range(len(bins) - 1):
        bins_center.append((bins[index] + bins[index + 1]) / 2)

    smooth_result = oned_gaussian_filter(n / np.sum(n), smooth_win)

    plt.plot(smooth_result)
    plt.title("Smoothed Hist")
    plt.show()
    return smooth_result, np.array(bins_center)


# 一维高斯滤波
def oned_gaussian_filter(sequence, win_size):
    sigma = 128
    r = range(-int(win_size / 2), int(win_size / 2) + 1)
    gauss = [1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-float(x) ** 2 / (2 * sigma ** 2)) for x in r]
    return np.convolve(sequence, gauss, 'same')


# 对图像进行二值化处理
def segmentation_by_thresholding(image, thresholding):
    thresholding_image = image.copy()
    thresholding_image[image > thresholding] = 255
    thresholding_image[image <= thresholding] = 0
    return thresholding_image


# 定义高斯模型
def gaussian(x, mu, theta):
    theta = theta + np.array(1e-10)  # 避免 0 做除数
    return (1 / ((2 * np.pi) ** 0.5 * theta)) * np.exp(-(x - mu) ** 2 / (2 * theta ** 2))


# 根据阈值预测混合高斯分布
def mixture_gaussian(thresholding, n, bins_center, smooth_win):
    if thresholding > np.max(bins_center) or thresholding < 0:
        print("Invalid Value of Thresholding:", thresholding)
        return
    bins_of_fo = bins_center[bins_center <= thresholding]
    po = n[bins_center <= thresholding] / np.sum(n[bins_center <= thresholding])
    mu_of_fo = np.sum(bins_of_fo * po)
    theta_of_fo = np.mean((bins_of_fo - mu_of_fo) ** 2 * po)

    bins_of_fg = bins_center[bins_center > thresholding]
    pg = n[bins_center > thresholding] / np.sum(n[bins_center > thresholding])
    mu_of_fg = np.sum(bins_of_fg * pg)
    theta_of_fg = np.mean((bins_of_fg - mu_of_fg) ** 2 * pg)

    po_t = np.sum(n[bins_center <= thresholding])
    pb_t = 1 - po_t

    predicted_model = []
    for bin_center in bins_center:
        predicted_model.append(po_t * gaussian(bin_center, mu_of_fo, theta_of_fo)
                               + pb_t * gaussian(bin_center, mu_of_fg, theta_of_fg))
    return oned_gaussian_filter(predicted_model, smooth_win)


# 计算散度
def calculate_divergence(true_distribution, fit_distribution):
    eps = np.array(1e-10)  # 避免 0 做除数
    fit_distribution = fit_distribution + eps
    return np.mean(true_distribution * np.log(true_distribution / fit_distribution + eps))  # 避免 log(0)


# 循环获得使散度最小的阈值
def get_auto_threshold(n, bins_center, smooth_win):
    divergences = []
    mixture_gaussians = []
    # for potential_thresholding in range(0, math.ceil(np.max(bin_center))):
    for potential_thresholding in range(50, 200):
        predicted_model = mixture_gaussian(potential_thresholding, n, bins_center, smooth_win)
        # if len(predicted_model) == 0:
        #     continue
        mixture_gaussians.append(predicted_model)
        divergences.append(calculate_divergence(n, predicted_model))
    # print(divergences)
    plt.plot(divergences)
    plt.title("Curve of Thresholding-Divergence")
    plt.show()
    plt.bar(bins_center, mixture_gaussians[int(np.argmin(divergences))])
    plt.title("Distribution under Auto-Thresholding")
    plt.show()
    return np.argmin(divergences) + 50


# 绘制原始图像及随机阈值的分割结果
def show_segmentation_result_by_random_thresholding(image):
    ax0 = plt.subplot(2, 2, 1)
    ax0.set_title("origin_image")
    plt.axis("off")
    plt.imshow(image, cmap='gray')

    ax1 = plt.subplot(2, 2, 2)
    ax1.set_title("thresholding=125")
    plt.axis("off")
    plt.imshow(segmentation_by_thresholding(image, 125), cmap='gray')

    ax2 = plt.subplot(2, 2, 3)
    ax2.set_title("thresholding=99")
    plt.axis("off")
    plt.imshow(segmentation_by_thresholding(image, 99), cmap='gray')

    ax3 = plt.subplot(2, 2, 4)
    ax3.set_title("thresholding=156")
    plt.axis("off")
    plt.imshow(segmentation_by_thresholding(image, 156), cmap='gray')

    plt.show()


# 绘制自适应阈值的分割结果
def show_segmentation_result_by_auto_thresholding(image, thresholding):
    image = segmentation_by_thresholding(image, thresholding)
    plt.imshow(image, cmap='gray')
    plt.show()


def main():
    image_dir = "./Segmentation_data/"
    for directory, subdirectory, filenames in os.walk(image_dir):
        for image_name in filenames:
            image = load_image(image_dir + image_name)
            show_segmentation_result_by_random_thresholding(image)
            n, bins = show_gray_level_histogram(image, 256, 96)
            thresholding = get_auto_threshold(n, bins, 96)
            if thresholding != -1:
                show_segmentation_result_by_auto_thresholding(image, thresholding)
            print(image_name, "Auto-Thresholding:", thresholding)


if __name__ == "__main__":
    main()
