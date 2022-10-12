# coding: utf-8
# Author：Qi Tang
# Date ：05/10/2022

import os
import cv2
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 判断是否为图片格式的文件
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


# 读取图片
def load_image(path):
    if is_image_file(path):
        return np.array(Image.open(path), dtype=float).astype(np.uint8)
    else:
        print("Image file format error.")


def image_show(image, title=""):
    plt.imshow(image)
    plt.title(title)
    plt.show()


# Hausdorff 距离三维可视化
def hausdorff_distance_show(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()


# 计算相关
def calculate_correlation(m, n):
    return np.sum(m * n) / np.sqrt(np.sum(m ** 2) * np.sum(n ** 2))


# 计算 hausdorff 距离
def calculate_hausdorff_distance(a, b):
    h_a_b = np.min((abs(a.reshape([-1, 1]).repeat(b.shape[0], 1) - b)).reshape([len(a.flatten()), -1]), axis=-1)
    h_b_a = np.min((abs(b.reshape([-1, 1]).repeat(a.shape[0], 1) - a)).reshape([len(b.flatten()), -1]), axis=-1)
    return np.max((np.max(h_a_b), np.max(h_b_a)))


# 计算距离变换
def calculate_hausdorff_distance_transform(scene):
    row, col = scene.shape
    distance = np.ones((row, col))
    max = np.max(scene)
    r, c = np.where(scene == max)
    for i in range(row):
        for j in range(col):
            distance[i, j] = np.min(abs(r - i) + abs(c - j))
    return distance


# 获取图像中响应最大的位置
def max_value_position(array):
    h, w = array.shape
    max_idx = np.argmax(array)
    return max_idx // w, max_idx % w


# 获取图像中响应最小的位置
def min_value_position(array):
    h, w = array.shape
    min_dix = np.argmin(array.flatten())
    return min_dix // w, min_dix % w


# 相关匹配
def correlation_match(template, scene):
    scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template = template[2:template.shape[0] - 2, 2:template.shape[1] - 2]

    max_template = np.max(template)
    min_template = np.min(template)

    max_scene = np.max(scene)
    min_scene = np.min(scene)

    template = (template - min_template) / (max_template - min_template)
    scene = (scene - min_scene) / (max_scene - min_scene)

    template_h, template_w = template.shape
    scene_h, scene_w = scene.shape
    print(template_h, template_w, scene_h, scene_w)
    # 分窗
    strides = scene.itemsize * np.array([scene_w, 1, scene_w, 1])
    new_h = scene_h - (template_h - 1)
    new_w = scene_w - (template_w - 1)
    scene_slides = np.lib.stride_tricks.as_strided(scene, shape=(new_h, new_w, template_h, template_w),
                                                   strides=strides)
    print(scene_slides.shape)
    correlation = np.zeros((new_h, new_w))
    for h in range(new_h):
        for w in range(new_w):
            correlation[h][w] = calculate_correlation(template, scene_slides[h][w])
    position = max_value_position(correlation)

    return [*position, *template.shape]


# hausdorff 距离匹配
def hausdorff_match(idx, template, scene):
    t_dict = {"0": [150, 250], "1": [50, 250]}
    t = t_dict[str(idx)]
    template = cv2.Canny(template[2:template.shape[0] - 2, 2:template.shape[1] - 2], t[0], t[1])
    if idx == 0:
        scene = cv2.Canny(scene, 150, 250)
        scene_edge = scene.copy()
        image_show(scene)

    image_show(template)

    stride = 3
    template_h, template_w = template.shape
    scene_h, scene_w = scene.shape
    strides = scene.itemsize * np.array([scene_w * stride, 1 * stride, scene_w, 1])
    new_h = (scene_h - template_h) // stride + 1
    new_w = (scene_w - template_w) // stride + 1
    scene_slides = np.lib.stride_tricks.as_strided(scene, shape=(new_h, new_w, template_h, template_w),
                                                   strides=strides)
    template_x, template_y = np.where(template > 0)
    distance = np.zeros((new_h, new_w))
    # print(scene_slides.shape)
    for h in range(new_h):
        for w in range(new_w):
            win_x, win_y = np.where(scene_slides[h][w] > 0)
            distance[h][w] = np.max(
                [calculate_hausdorff_distance(template_x, win_x), calculate_hausdorff_distance(template_y, win_y)])
    position = min_value_position(distance)
    # print(distance[0, 0])
    if idx == 0:
        return [(position[0] - 1) * stride, (position[1] - 1) * stride, *template.shape], scene_edge
    else:
        return [(position[0] - 1) * stride, (position[1] - 1) * stride, *template.shape]


# 距离变换匹配
def distance_transform_match(idx, template, scene):
    t_dict = {"0": [150, 250], "1": [50, 250]}
    t = t_dict[str(idx)]
    template = cv2.Canny(template[2:template.shape[0] - 2, 2:template.shape[1] - 2], t[0], t[1])
    if idx == 0:
        scene = cv2.Canny(scene, 150, 250)
        scene = calculate_hausdorff_distance_transform(scene)
        image_show(scene)
        scene_edge = scene.copy()

    stride = 1
    template_h, template_w = template.shape
    scene_h, scene_w = scene.shape
    strides = scene.itemsize * np.array([scene_w * stride, 1 * stride, scene_w, 1])
    new_h = (scene_h - template_h) // stride + 1
    new_w = (scene_w - template_w) // stride + 1
    scene_slides = np.lib.stride_tricks.as_strided(scene, shape=(new_h, new_w, template_h, template_w),
                                                   strides=strides)
    distance = np.zeros((new_h, new_w))
    # print(scene_slides.shape)
    for h in range(new_h):
        for w in range(new_w):
            distance[h][w] = np.mean(template * scene_slides[h][w])
    hausdorff_distance_show(np.arange(new_w), np.arange(new_h), distance)
    position = min_value_position(distance)
    if idx == 0:
        return [(position[0] - 1) * stride, (position[1] - 1) * stride, *template.shape], scene_edge
    else:
        return [(position[0] - 1) * stride, (position[1] - 1) * stride, *template.shape]


# 绘制模版匹配结果
def draw_rectangle(locations, scene, title=""):
    scene_copy = scene.copy()

    masks = [np.zeros(scene.shape, np.uint8), np.zeros(scene.shape, np.uint8)]
    names = ["Template_1", "Template_2"]
    point_color = [[255, 0, 0], [0, 255, 0]]
    thickness = 2
    lineType = 4

    for idx, local in enumerate(locations):
        point_left_top = (local[1], local[0])
        point_right_bottom = (local[1] + local[3], local[0] + local[2])
        cv2.rectangle(masks[idx], point_left_top, point_right_bottom, point_color[idx], thickness, lineType)
        cv2.rectangle(masks[idx], (point_left_top[0] - 1, point_left_top[1]), (local[1] + local[3] + 1, local[0] - 15),
                      point_color[idx], -1, lineType)
        cv2.putText(masks[idx], names[idx], (local[1] + 2, local[0] - 4), cv2.FONT_HERSHEY_COMPLEX, .3, (0, 0, 0), 1)

    mask = np.array(masks[0] + masks[1])
    scene_copy = cv2.addWeighted(scene_copy, 1.0, mask, 0.5, 0)
    cv2.imwrite(title + ".png", scene_copy)

    plt.imshow(scene_copy, cmap='gray')
    plt.title(title)
    plt.show()


# 模版匹配
def localize(templates, scene, function, title):
    scene_r = scene.shape[0]
    locations = []
    for idx, template in enumerate(templates):
        if title == "correlation":
            location = function(template, scene)
        else:
            if idx == 0:
                location, scene_edge = function(idx, template, scene)
            else:
                location = function(idx, template, scene_edge)
        locations.append(location)
    print(locations)
    print("左下角为(0,0)情况下,模版1和2目标框的左上角坐标分别为:\n",
          [[scene_r - item[0], item[1]] for item in locations])
    draw_rectangle(locations, scene, title=title)


def main():
    image_dir = r"./Template_matching_data/"
    template_1 = load_image(os.path.join(image_dir, "Template_1.jpg"))
    template_2 = load_image(os.path.join(image_dir, "Template_2.jpg"))
    scene = load_image(os.path.join(image_dir, "Scene.jpg"))

    start = time.time()
    localize([template_1, template_2], scene, correlation_match, "correlation")
    print("ncc耗时:", time.time() - start)

    start = time.time()
    localize([template_1, template_2], scene, hausdorff_match, "hausdorff")
    print("hausdorff耗时:", time.time() - start)

    start = time.time()
    localize([template_1, template_2], scene, distance_transform_match, "distance_transform")
    print("distance_transform+hausdorff耗时:", time.time() - start)


if __name__ == "__main__":
    main()
