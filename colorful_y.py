from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2

label = load_img("data/train-y-3.png",grayscale = True)
label = img_to_array(label)

label = label.astype('uint8')

label_trans = np.zeros( (label.shape[0],label.shape[1],3), dtype=np.uint8)

for i in range(label.shape[0]):
	for j in range(label.shape[1]):
		if label[i,j,0] == 0:
			label_trans[i,j] = (0,0,0);
		elif label[i,j,0] == 1:
			label_trans[i,j] = (255,0,0);
		elif label[i,j,0] == 2:
			label_trans[i,j] = (0,0,255);
		elif label[i,j,0] == 3:
			label_trans[i,j] = (255,255,0);
		elif label[i,j,0] == 4:
			label_trans[i,j] = (255,255,255);

label_trans = array_to_img(label_trans)
label_trans.save("y-3.jpg")