# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import cv2
import csv
import pandas as pd
import numpy as np





def merge_aug(from_file,to_file):
    class1 = np.load(from_file)
    print(class1.shape)

    for degree in [0,1,2,3]:
        for flip in [0,1]:
            for method in [0,1,2,3,4]:

                if (degree == 1 or degree == 3):
                    tmp = class1[degree * 10 + flip * 5 + method, :, :, 0][0:cols, 0:rows]
                else:
                    tmp = class1[ degree*10 + flip*5 + method,:,:,0][0:rows,0:cols]

                if flip == 1:
                    tmp = tmp[::-1]

                # 顺时针旋转
                rotate_times = degree
                while rotate_times > 0:
                    tmp = map(list, zip(*tmp[::-1]))
                    rotate_times -= 1

                class1[degree * 10 + flip * 5 + method, 0:rows, 0:cols, 0] = tmp
                print(" done one!!")

    print("ratate done!!")
    compress = np.sum(class1, axis=0) / 40.0


    np.save(to_file, compress)

cols = 5664
rows = 5142

merge_aug('img0_class4_40.npy','img0_class4_compress_aug.npy')

cols = 4011
rows = 2470

merge_aug('img1_class4_40.npy','img1_class4_compress_aug.npy')

cols = 3356
rows = 6116

merge_aug('img2_class4_40.npy','img2_class4_compress_aug.npy')




