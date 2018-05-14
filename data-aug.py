# -*- coding: UTF-8 -*-
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
from PIL import Image
from PIL import ImageEnhance
import os
import random
import matplotlib.pyplot as pyplot

def enhance(image, method, label):


    if label == True:
        return image

    elif method == "bright":
        enh_bri = ImageEnhance.Brightness(image)
        brightness = 1.5
        tmp = enh_bri.enhance(brightness)

    elif method == "color":
        enh_col = ImageEnhance.Color(image)
        color = 1.5
        tmp = enh_col.enhance(color)

    elif method == "contrast":
        enh_con = ImageEnhance.Contrast(image)
        contrast = 1.5
        tmp = enh_con.enhance(contrast)

    elif method == "sharp":
        enh_sha = ImageEnhance.Sharpness(image)
        sharpness = 3.0
        tmp = enh_sha.enhance(sharpness)

    elif method == "none":
        tmp = image

    return tmp


def rotate_flip_enhance(image, file_index, sava_folder, label, degree, flip, method):

    # 旋转
    if degree == 0:
        tmp = image.rotate(0, expand=1)
    elif degree == 1:
        tmp = image.rotate(90, expand=1)
    elif degree == 2:
        tmp = image.rotate(180, expand=1)
    elif degree == 3:
        tmp = image.rotate(270, expand=1)

    # 翻转
    if flip == 0:
        tmp = tmp.transpose(Image.FLIP_TOP_BOTTOM)
    elif flip == 1:
        tmp = tmp

    # 增强
    if method == 1:
        tmp = enhance(tmp, "bright", label)
    elif method == 2:
        tmp = enhance(tmp, "color", label)
    elif method == 3:
        tmp = enhance(tmp, "contrast", label)
    elif method == 4:
        tmp = enhance(tmp, "sharp", label)
    elif method == 0:
        tmp = enhance(tmp, "none", label)

    name = file_index + "_" +str(degree) + "_" + str(flip) + "_" + str(method)
    tmp.save(sava_folder + "/" + name + ".png")


class myAugmentation(object):

    def __init__(self,
                 train_path="/home1/zj/segmentation/x",
                 label_path="/home1/zj/segmentation/y",

                 aug_train_path="/home1/zj/segmentation/aug-x",
                 aug_label_path="/home1/zj/segmentation/aug-y",

                 npy_path = "/home1/zj/segmentation/npydata",
                 img_type="png"):

        self.train_imgs = glob.glob(train_path + "/*." + img_type)
        self.label_imgs = glob.glob(label_path + "/*." + img_type)

        self.train_path = train_path
        self.label_path = label_path

        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path

        self.npy_path = npy_path

        self.slices = len(self.train_imgs)
        self.img_type = img_type

        self.aug_class_index = 3

        if not os.path.lexists(aug_train_path):
            os.mkdir(aug_train_path)

        if not os.path.lexists(aug_label_path):
            os.mkdir(aug_label_path)

    def Augmentation(self):
        # 每种比例的样本的图片数量
        num_per_percentage = 200

        # 根据对哪一类的样本做数据增强来加载不同的统计数据
        class_frequency = np.load( self.npy_path + "/class"+ str(self.aug_class_index) +"_frequency_512.npy")

        trains = self.train_imgs
        labels = self.label_imgs

        augfoldx = self.aug_train_path
        augfoldy = self.aug_label_path

        if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
            print "trains can't match labels"
            return 0

        i = 0
        for filename_y in labels:

            # 根据当前样本前景像素点的比例决定数据增强的倍数
            # 对样本y做数据增强
            img = load_img(filename_y, grayscale=True)  # 读y采用灰度模式

            filename_x = filename_y.replace('y', 'x')
            img_x = Image.open(filename_x)

            img_array = img_to_array(img)
            img_array = img_array.astype('uint8')
            this_img_percentage = (img_array == self.aug_class_index).sum() / float(512*512) * 100 # 得到这张图片的前景百分比
            this_percentage_num = class_frequency[int(this_img_percentage)] # 得到这类百分比的图片一共有多少张

            if this_percentage_num > num_per_percentage:
                # 以一定的概率选取这张图像
                choose_probability = float(num_per_percentage)/float(this_percentage_num)
                if random.random() < choose_probability:

                    rotate_flip_enhance(img, str(i), augfoldy, True, 0, 0, 0)

                    rotate_flip_enhance(img_x, str(i), augfoldx, False, 0, 0, 0)
            else:
                # 以一定的倍数增强这张图像
                enhance_mul = num_per_percentage // this_percentage_num
                # rotate_flip_enhance(image, file_index, sava_folder, label, degree, flip, method)
                rand_list = np.random.permutation(40)

                for enhance_i in range(int(enhance_mul)):
                    choose = rand_list[ enhance_i%40 ]
                    # 生成degree
                    degree = choose // 10
                    choose = choose % 10
                    # 生成flip
                    flip = choose // 5
                    choose = choose % 5
                    # 生成method
                    method = choose // 1


                    rotate_flip_enhance(img, str(i), augfoldy, True, degree, flip, method)

                    rotate_flip_enhance(img_x, str(i), augfoldx, False, degree, flip, method)

            i += 1
            print('Done: {0}/{1} images'.format(i, self.slices))




if __name__ == "__main__":

    # aug = myAugmentation()
    # aug.Augmentation()

    if not os.path.lexists("data/test_aug"):
        os.mkdir("data/test_aug")

    img_t = Image.open("data/test/testing1_8bits.png")
    for degree in [0,1,2,3]:
        for flip in [0,1]:
            for method in [0,1,2,3,4]:
                rotate_flip_enhance(img_t, "1", "data/test_aug", False, degree, flip, method)
                print("Done one!!")

    img_t = Image.open("data/test/testing2_8bits.png")
    for degree in [0,1,2,3]:
        for flip in [0,1]:
            for method in [0, 1, 2, 3, 4]:
                rotate_flip_enhance(img_t, "2", "data/test_aug", False, degree, flip, method)
                print("Done one!!")

    img_t = Image.open("data/test/testing3_8bits.png")
    for degree in [0,1,2,3]:
        for flip in [0,1]:
            for method in [0, 1, 2, 3, 4]:
                rotate_flip_enhance(img_t, "3", "data/test_aug", False, degree, flip, method)
                print("Done one!!")