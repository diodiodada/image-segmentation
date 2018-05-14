# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.np_utils import to_categorical
import numpy as np
import os
import glob
import cv2

imgs_test = np.load("/media/data_2t/zj/class4_results_120.npy")

for img_index in [0,1,2]:

    img_40 = imgs_test[img_index*40 : img_index*40+40]
    np.save("/media/data_2t/zj/img" + str(img_index) + "_40.npy", img_40)

    print("create "+ str(img_index)+ "file")






