# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import cv2
import csv
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def create_img_result(file_index, cols, rows, name_i):


    if file_index==0:
        img1_class1 = np.load('img0_class1_compress_aug.npy')
        img1_class2 = np.load('img0_class2_compress_aug.npy')
        img1_class3 = np.load('img0_class3_compress_aug.npy')
        img1_class4 = np.load('img0_class4_compress_aug.npy')

    if file_index==1:
        img1_class1 = np.load('img1_class1_compress_aug.npy')
        img1_class2 = np.load('img1_class2_compress_aug.npy')
        img1_class3 = np.load('img1_class3_compress_aug.npy')
        img1_class4 = np.load('img1_class4_compress_aug.npy')

    if file_index==2:
        img1_class1 = np.load('img2_class1_compress_aug.npy')
        img1_class2 = np.load('img2_class2_compress_aug.npy')
        img1_class3 = np.load('img2_class3_compress_aug.npy')
        img1_class4 = np.load('img2_class4_compress_aug.npy')

    # 裁剪
    img1_class1 = img1_class1[0:rows,0:cols,0]
    img1_class2 = img1_class2[0:rows,0:cols,0]
    img1_class3 = img1_class3[0:rows,0:cols,0]
    img1_class4 = img1_class4[0:rows,0:cols,0]

    # ============================================================
    # 判断类别

    results_4class = np.zeros((rows, cols, 4), dtype=np.float32)

    #植被
    # img1_class1[img1_class1 > 0.8] = 1
    # img1_class1[img1_class1 <= 0.8] = 0
    #
    # # 建筑
    # img1_class2[img1_class2 > 0.5] = 1
    # img1_class2[img1_class2 <= 0.5] = 0
    #
    # # 水体
    # img1_class3[img1_class3 > 0.5] = 1
    # img1_class3[img1_class3 <= 0.5] = 0
    #
    # # 道路
    # img1_class4[img1_class4 > 0.2] = 1
    # img1_class4[img1_class4 <= 0.2] = 0

    results_4class[:, :, 0] = img1_class1
    results_4class[:, :, 1] = img1_class2
    results_4class[:, :, 2] = img1_class3
    results_4class[:, :, 3] = img1_class4

    # 关键的一步,找到最大索引
    max_index = np.argmax(results_4class, axis=2) + 1
    max_index = max_index.astype('uint8')

    #==============================================================
    # 写入CSV文件

    results_img1 = np.zeros((rows * cols), dtype=np.uint8)

    for i in range(cols):
        for j in range(rows):
            results_img1[i * rows + j] = max_index[j,i]

    filename = str(name_i) + '.csv'
    f = open(filename, 'w')
    writer = csv.writer(f)
    writer.writerow(results_img1)
    f.close()


    #==============================================================
    # 生成图片

    label_trans = np.zeros((max_index.shape[0], max_index.shape[1], 3), dtype=np.uint8)

    label_trans[max_index == 0] = (0, 0, 0)
    label_trans[max_index == 1] = (255, 0, 0)
    label_trans[max_index == 2] = (0, 0, 255)
    label_trans[max_index == 3] = (255, 255, 0)
    label_trans[max_index == 4] = (255, 255, 255)

    label_trans = array_to_img(label_trans)
    label_trans.save("y-" + str(name_i) + ".png")

    # ============================================================


# 第一张图片在npy数组中的第2维
create_img_result(file_index = 0,cols = 5664,rows = 5142,name_i = 1)
print("Done one...")

# 第二张图片在npy数组中的第1维
create_img_result(file_index = 1,cols = 4011,rows = 2470,name_i = 2)
print("Done one...")

# 第三张图片在npy数组中的第0维
create_img_result(file_index = 2,cols = 3356,rows = 6116,name_i = 3)
print("Done one...")