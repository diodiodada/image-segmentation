# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import cv2
import csv
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img




def inspect3(class_result):
    if not os.path.lexists("results-binary"):
        os.mkdir("results-binary")

    threshold = 0.2

    imgs = np.load('class'+str(class_result)+'_results.npy')
    for img_index in range(imgs.shape[0]):
        # 每一个numpy数组包含3张图片的结果
        img = imgs[img_index,:,:,:]

        img[img > threshold] = 1
        img[img <= threshold] = 0

        img = array_to_img(img)
        img.save("results-binary/"+ 'class' + str(class_result)+  'image' + str(img_index) + ".jpg" )


def inspect1():
    if not os.path.lexists("results-binary"):
        os.mkdir("results-binary")

    rows = [5142,2470,6116]
    cols = [5664,4011,3356]

    #            植被     建筑   水体     道路
    threshold = [0.5,    0.5,   0.5,    0.3]

    for img_index in range(3):
        for class_index in [1,2,3,4]:

            img = np.load("img" +str(img_index)+ "_class" +str(class_index)+ "_compress_aug.npy")

            img = img[0:rows[img_index],0:cols[img_index],:]

            img[img > threshold[class_index-1]] = 1
            img[img <= threshold[class_index-1]] = 0

            img = array_to_img(img)
            img.save("results-binary/img" +str(img_index)+ "_class" +str(class_index)+ "_compress_aug.jpg" )

            print("done_"+str(img_index)+"_"+str(class_index) )



def create_one_result(file_index, cols, rows, name_i):
    # 读取同一个维度的4个不同数组获得同一个图片的4个分类
    # 选取最大值为最终分类

    class4_threshold = 0.00001

    class1 = np.load('class1_results_aug.npy')
    class2 = np.load('class2_results_aug.npy')
    class3 = np.load('class3_results_aug.npy')
    class4 = np.load('class4_results_aug.npy')

    print(class1.shape)
    print(class2.shape)
    print(class3.shape)
    print(class4.shape)



    results_4class = np.zeros((rows, cols, 4), dtype=np.float32)

    results_4class[:, :, 0] = class1[file_index, 0:rows, 0:cols, 0]
    results_4class[:, :, 1] = class2[file_index, 0:rows, 0:cols, 0]
    results_4class[:, :, 2] = class3[file_index, 0:rows, 0:cols, 0]
    results_4class[:, :, 3] = class4[file_index, 0:rows, 0:cols, 0]


    class4_only = class4[file_index, 0:rows, 0:cols, 0]

    max_index = np.argmax(results_4class, axis=2) + 1

    # 对道路进行特殊处理
    max_index[class4_only > class4_threshold] = 4



    #===========================================================================

    max_index = max_index.astype('uint8')

    label_trans = np.zeros((max_index.shape[0], max_index.shape[1], 3), dtype=np.uint8)

    label_trans[max_index == 0] = (0, 0, 0)
    label_trans[max_index == 1] = (255, 0, 0)
    label_trans[max_index == 2] = (0, 0, 255)
    label_trans[max_index == 3] = (255, 255, 0)
    label_trans[max_index == 4] = (255, 255, 255)

    label_trans = array_to_img(label_trans)
    label_trans.save("y-" + str(name_i) +"-"+ str(class4_threshold) +".png")

    #===========================================================================


def create_class4_result(file_index, cols, rows, name_i):
    # 读取同一个维度的4个不同数组获得同一个图片的4个分类
    # 选取最大值为最终分类



    class4_threshold = 0.00001

    class1 = np.load('class1_results_aug.npy')
    class2 = np.load('class2_results_aug.npy')
    class3 = np.load('class3_results_aug.npy')
    class4 = np.load('class4_results_aug.npy')

    print(class1.shape)
    print(class2.shape)
    print(class3.shape)
    print(class4.shape)

    class4_only = class4[file_index, 0:rows, 0:cols, 0]

    max_index = np.ones((rows, cols), dtype=np.uint8)

    # 对道路进行特殊处理
    max_index[class4_only > class4_threshold] = 4

    #===========================================================================

    label_trans = np.zeros((max_index.shape[0], max_index.shape[1], 3), dtype=np.uint8)

    label_trans[max_index == 0] = (0, 0, 0)
    label_trans[max_index == 1] = (255, 0, 0)
    label_trans[max_index == 2] = (0, 0, 255)
    label_trans[max_index == 3] = (255, 255, 0)
    label_trans[max_index == 4] = (255, 255, 255)

    label_trans = array_to_img(label_trans)
    label_trans.save("y-" + str(name_i) +"-"+ str(class4_threshold) +".png")

    #===========================================================================



# # 第一张图片在npy数组中的第0维
# create_class4_result(file_index = 0,cols = 5664,rows = 5142,name_i = 1)
# print("Done one...")
#
# # 第二张图片在npy数组中的第1维
# create_class4_result(file_index = 1,cols = 4011,rows = 2470,name_i = 2)
# print("Done one...")
#
# # 第三张图片在npy数组中的第2维
# create_class4_result(file_index = 2,cols = 3356,rows = 6116,name_i = 3)
# print("Done one...")



inspect1()