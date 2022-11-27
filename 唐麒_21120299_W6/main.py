# coding: utf-8
# Author：QiTang
# Date ：20/11/2022

import cv2
import os
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

winSize = (94, 34)
blockSize = (8, 8)
blockStride = (2, 2)
cellSize = (4, 4)
winStride = (0, 0)
nbins = 11
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)


# 获取图片文件名列表
def list_image_file(dir):
    image_filenames = []
    for directory, subdirectory, filenames in os.walk(dir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.bmp':
                image_filenames.append(os.path.join(dir, filename))
    return image_filenames


# 加载训练数据集
def dataloader(data_dir):
    train_dir = os.path.join(data_dir, "train_34x94")
    train_pos = list_image_file(os.path.join(train_dir, 'pos'))
    train_neg = list_image_file(os.path.join(train_dir, 'neg'))
    pos_label = np.ones(len(train_pos))
    neg_label = np.zeros(len(train_neg))
    train_path = train_neg + train_pos

    features = []
    for train_dir in train_path:
        image = cv2.imread(train_dir)
        feature = hog.compute(image, winStride)
        features.append(np.squeeze(feature))
    labels = np.append(neg_label, pos_label)

    return features, labels


# 非极大值抑制
def apply_nms(candidate_bboxes, threshold=.5):
    if len(candidate_bboxes) == 0:
        return []
    candidate_bboxes = sorted(candidate_bboxes, key=lambda detections: detections[2],
                              reverse=True)
    bboxes = [candidate_bboxes[0]]
    del candidate_bboxes[0]
    for index, candidate_bbox in enumerate(candidate_bboxes):
        for bbox in bboxes:
            if calculate_IOU(candidate_bbox, bbox) > threshold:
                del candidate_bboxes[index]
                break
        else:
            bboxes.append(candidate_bbox)
            del candidate_bboxes[index]
    return bboxes


# 计算 IOU
def calculate_IOU(bbox1, bbox2):
    x_1, y_1 = bbox1[0], bbox1[1]
    x_2, y_2 = bbox2[0], bbox2[1]
    x1_br = bbox1[0] + bbox1[3]
    x2_br = bbox2[0] + bbox2[3]
    y1_br = bbox1[1] + bbox1[4]
    y2_br = bbox2[1] + bbox2[4]
    x_overlap = max(0, min(x1_br, x2_br) - max(x_1, x_2))
    y_overlap = max(0, min(y1_br, y2_br) - max(y_1, y_2))
    intersection = x_overlap * y_overlap
    bbox1_area = bbox1[3] * bbox2[4]
    bbox2_area = bbox2[3] * bbox2[4]
    union = bbox1_area + bbox2_area - intersection
    return intersection / float(union)


if __name__ == '__main__':
    home = './data/Dataset for Car/'
    time_start = time.time()
    data, label = dataloader(home)

    train_data, valid_data, train_label, valid_label = train_test_split(data, label, test_size=0.2, shuffle=True)
    print("数据划分结果，训练集：", np.shape(train_data), "验证集：", np.shape(valid_data))

    # PCA
    pca = PCA(n_components=300)
    pca.fit(train_data)
    train_data = pca.transform(train_data)
    valid_data = pca.transform(valid_data)

    # Logistic
    clf = LogisticRegression()
    clf.fit(train_data, train_label)

    # SVM
    # clf = SVC(probability=True)
    # clf.fit(train_data, train_label)

    # Adaboost
    # clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
    #                          learning_rate=1.0, n_estimators=100, random_state=0)
    # clf = AdaBoostClassifier()
    # clf.fit(train_data, train_label)

    print("分类器训练结果:")
    print(classification_report(clf.predict(valid_data), valid_label, target_names=["no car", "car"]))

    # 测试集
    test_path = os.path.join(home, "test")
    test_file = list_image_file(test_path)
    min_wdw_sz = (94, 34)
    num = 0
    for idx in range(len(test_file)):
        im = cv2.imread(test_file[idx])
        bboxes = []
        cd = []
        step = 18
        step_size = (im.shape[1] // step, im.shape[0] // step)
        y = 0
        while y <= (im.shape[0] - step_size[1]):
            x = 0
            while x <= (im.shape[1] - step_size[0]):
                im_window = im[y:y + min_wdw_sz[1], x:x + min_wdw_sz[0]]
                if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                    break
                fd = hog.compute(im_window, winStride)
                fd = np.squeeze(fd).tolist()
                fd = [fd]
                pred = clf.predict_proba(pca.transform(fd))[0]
                if pred[1] > 0.1:
                    step_size = (3, 3)
                else:
                    step_size = (im.shape[1] // step, im.shape[0] // step)
                if pred[0] < pred[1]:
                    bboxes.append((x, y, clf.decision_function(pca.transform(fd)),
                                   int(min_wdw_sz[0]),
                                   int(min_wdw_sz[1])))
                    cd.append(bboxes[-1])
                clone = im.copy()
                for x1, y1, _, _, _ in cd:
                    cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                                                    im_window.shape[0]), (255, 0, 0), thickness=2)
                cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                                              im_window.shape[0]), (255, 0, 0), thickness=2)
                cv2.imshow("image", clone)
                x += step_size[0]
                cv2.waitKey(1)

            y += step_size[1]
        clone = im.copy()
        threshold = 0.01
        # 非极大值抑制
        bboxes = apply_nms(bboxes, threshold)
        for (x_tl, y_tl, _, w, h) in bboxes:
            # Draw the detections
            num += 1
            cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (255, 0, 0), thickness=2)
        cv2.imwrite(os.path.join(home, 'result', '{0}.bmp'.format(idx)), clone)
    time_end = time.time()
    print('用时：', time_end - time_start, 's')
    print("平均每张图耗时:", (time_end - time_start) / (idx + 1), "s")
    print(num)
