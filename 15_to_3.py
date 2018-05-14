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

	imgdatas1 = np.ndarray(( 3,class1.shape[1],class1.shape[2],1), dtype=np.float32)

	# 把15维的数据降到3维

	for img_index in [0,1,2]:
		class1a = class1[img_index*5:img_index*5+5]
		# imgdatas1[img_index] = np.max(class1a, axis=0)
		imgdatas1[img_index] = np.sum(class1a,axis=0)/5.0
		print(imgdatas1[img_index].shape)

	np.save(to_file, imgdatas1)



merge_aug('class1_results.npy','class1_results_aug.npy')
merge_aug('class2_results.npy','class2_results_aug.npy')
merge_aug('class3_results.npy','class3_results_aug.npy')
merge_aug('class4_results.npy','class4_results_aug.npy')



