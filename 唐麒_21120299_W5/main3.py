# coding: utf-8
# Author：QiTang
# Date ：31/10/2022

import os
import cv2
import time


def main():
    # 第一步：cv2.getStructuringElement构造形态学使用的kernel
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 第二步：构造高斯混合模型
    model = cv2.createBackgroundSubtractorMOG2(history=109, varThreshold=100, detectShadows=True)

    total_time = 0
    num_frame = 0

    result_save_path = "./results/OpenCV/"
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)

    for pic_i in range(0, 201):
        if pic_i < 10:
            pic_name = "000%d.jpg" % pic_i
        elif 100 > pic_i >= 10:
            pic_name = "00%d.jpg" % pic_i
        else:
            pic_name = "0%d.jpg" % pic_i

        file_path = os.path.join("./Scene_Data/", pic_name)

        # 第三步：读取视频中的图片，并使用高斯模型进行拟合
        frame = cv2.imread(file_path)
        # 运用高斯模型进行拟合，在两个标准差内设置为0，在两个标准差外设置为255
        if pic_i >= 109:
            num_frame += 1
            start_time = time.time()
        fgmk = model.apply(frame)
        # 第四步：使用形态学的开运算做背景的去除
        fgmk = cv2.morphologyEx(fgmk, cv2.MORPH_OPEN, kernel)
        cv2.morphologyEx(fgmk, cv2.MORPH_CLOSE, kernel, iterations=3)
        if pic_i > 109:
            cost = time.time() - start_time
            total_time += cost
        # 第五步：进行图片的展示
        cv2.imshow('Scene', fgmk)
        if pic_i >= 109:
            cv2.imwrite(result_save_path + str(pic_i) + ".png", fgmk)

        k = cv2.waitKey(150) & 0xff
        if k == 27:
            break
    print("单张耗时:", total_time / num_frame, "每秒处理:", 1 / total_time)


if __name__ == "__main__":
    main()