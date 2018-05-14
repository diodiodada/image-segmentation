# -*- coding: utf-8 -*-
import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import glob
import cv2
import shutil


def splitimage(src, dstpath, width, hight, rownum, colnum):

    img = Image.open(src)
    w, h = img.size
    if hight <= h and width <= w:

        print('Original image info: %sx%s, %s, %s' % (w, h, img.format, img.mode))

        s = os.path.split(src)

        fn = s[1].split('.')
        basename = fn[0]       # 文件名
        ext = fn[-1]           # 文件后缀

        num = 0
        step_y = (h-hight)//(rownum-1)
        step_x = (w-width)//(colnum-1)


        y = 0

        for r in range(rownum):  # 竖着平移

            x = 0

            for c in range(colnum):  # 横着平移
                box = (x,
                       y,
                       x + width,
                       y + hight
                       )

                img.crop(box).save(os.path.join(dstpath, basename + '_' + str(num) + '.' + ext), ext)
                num = num + 1
                x = x + step_x

            y = y + step_y
            # print('处理完一行。')

        print('图片切割完毕，共生成 %s 张小图片。' % num)
    else:
        print('不合法的行列切割参数！')

def do_split(width, hight):

	width = width
	hight = hight

	if not os.path.lexists("data/train/y"):
		os.makedirs("data/train/y")

	rownum = 5142//hight+1
	colnum = 5664//width+1
	# 第一张
	src = "data/train-y-1.png"
	dstpath = "data/train/y"
	print("开始处理"+src)
	print("%d 行, %d 列" %(rownum,colnum))
	splitimage(src=src,dstpath=dstpath,width=width,hight=hight,rownum=rownum,colnum=colnum)


	rownum = 2470//hight+1
	colnum = 4011//width+1
	# 第二张
	src = "data/train-y-2.png"
	dstpath = "data/train/y"
	print("开始处理"+src)
	print("%d 行, %d 列" % (rownum, colnum))
	splitimage(src=src, dstpath=dstpath,width=width,hight=hight,rownum=rownum,colnum=colnum)

	rownum = 6116//hight+1
	colnum = 3357//width+1
	# 第三张
	src = "data/train-y-3.png"
	dstpath = "data/train/y"
	print("开始处理"+src)
	print("%d 行, %d 列" % (rownum, colnum))
	splitimage(src=src, dstpath=dstpath,width=width,hight=hight,rownum=rownum,colnum=colnum)

class dataProcess(object):

	def __init__(self,
				 rows,
				 cols,
				 label_path = "data/train/y",
				 npy_path = "npydata",
				 img_type = "png"):
		self.out_rows = rows
		self.out_cols = cols
		self.label_path = label_path
		self.img_type = img_type
		self.npy_path = npy_path

		if not os.path.lexists(npy_path):
			os.mkdir(npy_path)

	def create_train_data(self):
		i = 0

		imgs = glob.glob(self.label_path+"/*."+self.img_type)
		print(len(imgs))

		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)

		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			label = load_img(self.label_path + "/" + midname,grayscale = True)		# 读y采用灰度模式
			label = img_to_array(label)
			imglabels[i] = label
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1

		print('loading done')
		print("train data y size:" + str(imglabels.shape))
		np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
		print('Saving to .npy files done.')

	def count(self):
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
		imgs_mask_train = imgs_mask_train.astype('uint8')

		# ==============================================================================================
		# 这很重要，更改这里的值来更改分类目标
		# 根据分类目标对5类数据进行二值化
		imgs_mask_train[imgs_mask_train == 0] = 0
		imgs_mask_train[imgs_mask_train == 1] = 0
		imgs_mask_train[imgs_mask_train == 2] = 1
		imgs_mask_train[imgs_mask_train == 3] = 0
		imgs_mask_train[imgs_mask_train == 4] = 0
		# ==============================================================================================

		# 初始化index
		index = np.zeros(11)

		# 解决样本分类不均问题
		for i in range(imgs_mask_train.shape[0]):
			if imgs_mask_train[i].sum() > (imgs_mask_train.shape[1] * imgs_mask_train.shape[2]) * 0.9:
				index[10] += 1
			elif imgs_mask_train[i].sum() > (imgs_mask_train.shape[1] * imgs_mask_train.shape[2]) * 0.8:
				index[9] += 1
			elif imgs_mask_train[i].sum() > (imgs_mask_train.shape[1] * imgs_mask_train.shape[2]) * 0.7:
				index[8] += 1
			elif imgs_mask_train[i].sum() > (imgs_mask_train.shape[1] * imgs_mask_train.shape[2]) * 0.6:
				index[7] += 1
			elif imgs_mask_train[i].sum() > (imgs_mask_train.shape[1] * imgs_mask_train.shape[2]) * 0.5:
				index[6] += 1
			elif imgs_mask_train[i].sum() > (imgs_mask_train.shape[1] * imgs_mask_train.shape[2]) * 0.4:
				index[5] += 1
			elif imgs_mask_train[i].sum() > (imgs_mask_train.shape[1] * imgs_mask_train.shape[2]) * 0.3:
				index[4] += 1
			elif imgs_mask_train[i].sum() > (imgs_mask_train.shape[1] * imgs_mask_train.shape[2]) * 0.2:
				index[3] += 1
			elif imgs_mask_train[i].sum() > (imgs_mask_train.shape[1] * imgs_mask_train.shape[2]) * 0.1:
				index[2] += 1
			elif imgs_mask_train[i].sum() > 0:
				index[1] += 1
			elif imgs_mask_train[i].sum() == 0:
				index[0] += 1


		print("index shape:" + str(index.shape))
		print("index.sum:" + str(index.sum()))
		print(index)



shutil.rmtree('data/train')
print("删除了data/train")

shutil.rmtree('npydata')
print("删除了npydata")

size = 1024

do_split(size,size)

mydata = dataProcess(size,size)
mydata.create_train_data()
mydata.count()


