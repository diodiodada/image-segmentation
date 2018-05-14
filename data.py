# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.np_utils import to_categorical
import numpy as np 
import os
import glob
import cv2
#from libtiff import TIFF

class dataProcess(object):

	def __init__(self,
				 out_rows,
				 out_cols,
				 data_path = "/home1/zj/segmentation/aug-x",
				 label_path = "/home1/zj/segmentation/aug-y",
				 test_path = "data/test",
				 npy_path = "/home1/zj/segmentation/npydata",
				 img_type = "png"):

		self.out_rows = out_rows
		self.out_cols = out_cols

		self.data_path = data_path
		self.label_path = label_path
		self.img_type = img_type
		self.test_path = test_path
		self.test_path_aug = "data/test_aug"
		self.npy_path = npy_path

		self.imgs = glob.glob(self.data_path + "/*." + self.img_type)

		if not os.path.lexists(npy_path):
			os.mkdir(npy_path)

	def create_train_data(self):
		i = 0
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		imgs = glob.glob(self.data_path+"/*."+self.img_type)
		print(len(imgs))
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for imgname in imgs:

			# midname = imgname[imgname.rindex("/")+1:]
			# img = load_img(self.data_path + "/" + midname) 							# 读x采用RGB模式
			# label = load_img(self.label_path + "/" + midname,grayscale = True)		# 读y采用灰度模式

			img = load_img(imgname)  # 读x采用RGB模式
			label = load_img(imgname.replace('x','y'), grayscale=True)  # 读y采用灰度模式

			img = img_to_array(img)
			label = img_to_array(label)
			imgdatas[i] = img
			imglabels[i] = label
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
		print('loading done')
		print("train data x size:" + str(imgdatas.shape))
		print("train data y size:" + str(imglabels.shape))
		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
		print('Saving to .npy files done.')

	def create_test_data(self):
		padding = 64
		i = 0
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		imgs = glob.glob(self.test_path+"/*."+self.img_type)
		print(len(imgs))

		imgdatas = np.zeros((len(imgs),1024*6+padding*2,1024*6+padding*2,3), dtype=np.uint8)

		imgdatas_2 = np.zeros((len(imgs)*6*6, 1024+padding*2, 1024+padding*2, 3), dtype=np.uint8)


		for imgname in imgs:

			img = load_img(imgname)							# 读x采用RGB模式
			img = img_to_array(img)
			imgdatas[i, 0+padding : img.shape[0]+padding, 0+padding : img.shape[1]+padding ] = img
			i += 1

		# 把 imgdatas 重整为 imgdatas_2
		for i in range(len(imgs)):
			for row in range(6):
				for col in range(6):
					imgdatas_2[i*36+row*6+col,:,:,:] = imgdatas[i, 
															row*1024+padding-padding:(row+1)*1024+padding+padding, 
															col*1024+padding-padding:(col+1)*1024+padding+padding, 
															:]


		print('loading done')
		print("test data x size:" + str(imgdatas_2.shape))
		np.save(self.npy_path + '/imgs_test.npy', imgdatas_2)
		print('Saving to imgs_test.npy files done.')





	def create_test_data_aug(self):
		padding = 64
		i = 0
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		imgs = glob.glob(self.test_path_aug+"/*."+self.img_type)
		print(len(imgs))

		imgdatas = np.zeros((len(imgs),1024*6+padding*2,1024*6+padding*2,3), dtype=np.uint8)

		imgdatas_2 = np.zeros((len(imgs)*6*6, 1024+padding*2, 1024+padding*2, 3), dtype=np.uint8)


		# for imgname in imgs:
		#
		# 	img = load_img(imgname)							# 读x采用RGB模式
		# 	img = img_to_array(img)
		# 	imgdatas[i, 0+padding : img.shape[0]+padding, 0+padding : img.shape[1]+padding ] = img
		# 	i += 1

		for file_index in [1,2,3]:
			for degree in [0, 1, 2, 3]:
				for flip in [0, 1]:
					for method in [0, 1, 2, 3, 4]:
						name = self.test_path_aug + "/" + str(file_index) + "_" + str(degree) + "_" + str(flip) + "_" + str(method)+".png"
						img = load_img(name)
						img = img_to_array(img)
						imgdatas[i, 0 + padding: img.shape[0] + padding, 0 + padding: img.shape[1] + padding] = img
						i += 1


		# 把 imgdatas 重整为 imgdatas_2
		for i in range(len(imgs)):
			for row in range(6):
				for col in range(6):
					imgdatas_2[i*36+row*6+col,:,:,:] = imgdatas[i,
															row*1024+padding-padding:(row+1)*1024+padding+padding,
															col*1024+padding-padding:(col+1)*1024+padding+padding,
															:]


		print('loading done')
		print("test data x size:" + str(imgdatas_2.shape))
		np.save(self.npy_path + '/imgs_test_aug.npy', imgdatas_2)
		print('Saving to imgs_test_aug.npy files done.')



	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)

		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")

		# imgs_train 		= imgs_train.astype('uint8')
		imgs_mask_train = imgs_mask_train.astype('uint8')

		# ==============================================================================================
		# 这很重要，更改这里的值来更改分类目标
		# 根据分类目标对5类数据进行二值化
		imgs_mask_train[imgs_mask_train == 0] = 0
		imgs_mask_train[imgs_mask_train == 1] = 0
		imgs_mask_train[imgs_mask_train == 2] = 0
		imgs_mask_train[imgs_mask_train == 3] = 1
		imgs_mask_train[imgs_mask_train == 4] = 0
		# ==============================================================================================

		# 初始化index
		# index = np.zeros(imgs_train.shape[0])

		# 解决样本分类不均问题
		# for i in range(imgs_train.shape[0]):
		# 	if imgs_mask_train[i].sum() > (imgs_mask_train.shape[1]) * (imgs_mask_train.shape[2]) * 0.05:
		# 		index[i] = 1

		# imgs_train_reduce 		= imgs_train[index>0]
		# imgs_mask_train_reduce 	= imgs_mask_train[index > 0]
		# print("index shape:" + str(index.shape))
		# print("index.sum:" + str(index.sum()))
		
		imgs_train = imgs_train.astype('float32')
		imgs_train /= 255   # 这个千万别忘了
		return imgs_train,imgs_mask_train

	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		return imgs_test

	def load_test_data_aug(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test_aug.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		return imgs_test

	def generate_batch_data(self, batch_size):

		data_len = int( len(self.imgs) * 0.8 )
		num_batch = data_len // batch_size

		imgdatas = np.ndarray((batch_size, self.out_rows, self.out_cols, 3), dtype=np.uint8)
		imglabels = np.ndarray((batch_size, self.out_rows, self.out_cols, 1), dtype=np.uint8)

		while(True): # 无限循环

			# 生成每一个batch的随机位置
			rand_list = np.random.permutation(num_batch)
			for batch_i in rand_list: # 每循环一次，完成一个epoch

				file_i = 0
				file_name = self.imgs[batch_i * batch_size:(batch_i + 1) * batch_size]
				for imgname in file_name:# 每循环一次，完成一个batch

					midname = imgname[imgname.rindex("/") + 1:]

					#img = load_img(self.data_path + "/" + midname)  # 读x采用RGB模式
					#label = load_img(self.label_path + "/" + midname, grayscale=True)  # 读y采用灰度模式

					img = load_img(imgname)  # 读x采用RGB模式
					label = load_img(imgname.replace('x','y'), grayscale=True)  # 读y采用灰度模式

					img = img_to_array(img)
					label = img_to_array(label)
					imgdatas[file_i] = img
					imglabels[file_i] = label
					file_i += 1

				imgdatas = imgdatas.astype('float32')
				imgdatas /= 255

				imglabels = imglabels.astype('uint8')
				imglabels[imglabels == 0] = 0
				imglabels[imglabels == 1] = 0
				imglabels[imglabels == 2] = 1
				imglabels[imglabels == 3] = 0
				imglabels[imglabels == 4] = 0

				# imglabels_category = to_categorical(imglabels, num_classes=5)
				# imglabels_category = np.reshape(imglabels_category,(batch_size, self.out_rows, self.out_cols, 5))

				yield imgdatas, imglabels

	def get_valid_data(self, batch_size):

		data_begin = int(len(self.imgs) * 0.8)
		data_len = len(self.imgs)-data_begin
		num_batch = data_len // batch_size

		imgs = self.imgs[data_begin:-1]

		imgdatas = np.ndarray((batch_size, self.out_rows, self.out_cols, 3), dtype=np.uint8)
		imglabels = np.ndarray((batch_size, self.out_rows, self.out_cols, 1), dtype=np.uint8)

		while (True):  # 无限循环

			# 生成每一个batch的随机位置
			rand_list = np.random.permutation(num_batch)
			for batch_i in rand_list:  # 每循环一次，完成一个epoch

				file_i = 0
				file_name = imgs[batch_i * batch_size:(batch_i + 1) * batch_size]
				for imgname in file_name:  # 没循环一次，完成一个batch

					midname = imgname[imgname.rindex("/") + 1:]

					#img = load_img(self.data_path + "/" + midname)  # 读x采用RGB模式
					#label = load_img(self.label_path + "/" + midname, grayscale=True)  # 读y采用灰度模式

					img = load_img(imgname)  # 读x采用RGB模式
					label = load_img(imgname.replace('x', 'y'), grayscale=True)  # 读y采用灰度模式

					img = img_to_array(img)
					label = img_to_array(label)
					imgdatas[file_i] = img
					imglabels[file_i] = label
					file_i += 1

				imgdatas = imgdatas.astype('float32')
				imgdatas /= 255

				imglabels = imglabels.astype('uint8')
				imglabels[imglabels == 0] = 0
				imglabels[imglabels == 1] = 0
				imglabels[imglabels == 2] = 1
				imglabels[imglabels == 3] = 0
				imglabels[imglabels == 4] = 0

				# imglabels_category = to_categorical(imglabels, num_classes=5)
				# imglabels_category = np.reshape(imglabels_category, (batch_size, self.out_rows, self.out_cols, 5))

				yield imgdatas, imglabels

	def count_frequency(self):

		class1_frequency = np.zeros((101))
		class2_frequency = np.zeros((101))
		class3_frequency = np.zeros((101))
		class4_frequency = np.zeros((101))

		out_rows = 512
		out_cols = 512

		imgs = glob.glob("/home1/zj/segmentation/aug-x/*.png")

		pixel_sum = float(out_rows * out_cols)

		for imgname in imgs:# 每循环一次，完成一个batch

			label = load_img(imgname.replace('x', 'y'), grayscale=True)  # 读y采用灰度模式

			label = img_to_array(label)

			assert label.shape[0]==out_rows ,'图片高不符'
			assert label.shape[1]==out_cols ,'图片宽不符'

			label = label.astype('uint8')

			class1_percentage = (label==1).sum() / pixel_sum * 100
			class2_percentage = (label==2).sum() / pixel_sum * 100
			class3_percentage = (label==3).sum() / pixel_sum * 100
			class4_percentage = (label==4).sum() / pixel_sum * 100

			class1_frequency[int(class1_percentage)] += 1
			class2_frequency[int(class2_percentage)] += 1
			class3_frequency[int(class3_percentage)] += 1
			class4_frequency[int(class4_percentage)] += 1

		np.save(self.npy_path + '/class1_frequency_512_aug.npy', class1_frequency)
		np.save(self.npy_path + '/class2_frequency_512_aug.npy', class2_frequency)
		np.save(self.npy_path + '/class3_frequency_512_aug.npy', class3_frequency)
		np.save(self.npy_path + '/class4_frequency_512_aug.npy', class4_frequency)



if __name__ == "__main__":

	mydata = dataProcess(512,512)
	#mydata.create_train_data()
	mydata.create_test_data_aug()
	
	#mydata.count_frequency()
