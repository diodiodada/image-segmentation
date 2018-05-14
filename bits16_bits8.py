# -*- coding: utf-8 -*-
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# im = Image.open('data/train1.png')
#
# img = img_to_array(im)
#
#
# im_array = np.array(im)
# 也可以用 np.asarray(im) 区别是 np.array() 是深拷贝，np.asarray() 是浅拷贝


filename = "data/test/testing3.png"
filename_save = "data/test/testing3_8bits.png"

im = mpimg.imread(filename) # 这里读入的数据是 float32 型的，范围是0-1

im = Image.fromarray(np.uint8(im*65535)) # 乘以65536.转为int类型，再转为图像格式

im.save(filename_save)

