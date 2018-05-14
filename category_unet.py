# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, BatchNormalization,Activation
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

class myUnet(object):

	def __init__(self, img_rows, img_cols):

		self.img_rows = img_rows
		self.img_cols = img_cols


	def get_unet(self):

		inputs = Input((self.img_rows, self.img_cols,3))

		conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
		conv1 = BatchNormalization()(conv1)
		conv1 = Activation(activation = 'relu')(conv1)
		conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
		conv1 = BatchNormalization()(conv1)
		conv1 = Activation(activation = 'relu')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)


		conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
		conv2 = BatchNormalization()(conv2)
		conv2 = Activation(activation = 'relu')(conv2)
		conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
		conv2 = BatchNormalization()(conv2)
		conv2 = Activation(activation = 'relu')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


		conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
		conv3 = BatchNormalization()(conv3)
		conv3 = Activation(activation = 'relu')(conv3)
		conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
		conv3 = BatchNormalization()(conv3)
		conv3 = Activation(activation = 'relu')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


		conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = BatchNormalization()(conv4)
		conv4 = Activation(activation = 'relu')(conv4)
		conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
		conv4 = BatchNormalization()(conv4)
		conv4 = Activation(activation = 'relu')(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = BatchNormalization()(conv5)
		conv5 = Activation(activation = 'relu')(conv5)
		conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)
		conv5 = BatchNormalization()(conv5)
		conv5 = Activation(activation = 'relu')(conv5)

		up6 = Conv2D(512, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
		up6 = BatchNormalization()(up6)
		up6 = Activation(activation = 'relu')(up6)
		merge6 = merge([conv4,up6], mode = 'concat', concat_axis = 3)
		conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = BatchNormalization()(conv6)
		conv6 = Activation(activation = 'relu')(conv6)
		conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
		conv6 = BatchNormalization()(conv6)
		conv6 = Activation(activation = 'relu')(conv6)

		up7 = Conv2D(256, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		up7 = BatchNormalization()(up7)
		up7 = Activation(activation = 'relu')(up7)
		merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
		conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = BatchNormalization()(conv7)
		conv7 = Activation(activation = 'relu')(conv7)
		conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
		conv7 = BatchNormalization()(conv7)
		conv7 = Activation(activation = 'relu')(conv7)

		up8 = Conv2D(128, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		up8 = BatchNormalization()(up8)
		up8 = Activation(activation = 'relu')(up8)
		merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
		conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = BatchNormalization()(conv8)
		conv8 = Activation(activation = 'relu')(conv8)
		conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
		conv8 = BatchNormalization()(conv8)
		conv8 = Activation(activation = 'relu')(conv8)

		up9 = Conv2D(64, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		up9 = BatchNormalization()(up9)
		up9 = Activation(activation = 'relu')(up9)
		merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
		conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = BatchNormalization()(conv9)
		conv9 = Activation(activation = 'relu')(conv9)
		conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = BatchNormalization()(conv9)
		conv9 = Activation(activation = 'relu')(conv9)
		conv10 = Conv2D(5, 3, activation = 'softmax', padding = 'same', kernel_initializer = 'he_normal')(conv9)


		print "conv1 shape:", conv1.shape
		print "pool1 shape:", pool1.shape
		print "conv2 shape:", conv2.shape
		print "pool2 shape:", pool2.shape
		print "conv3 shape:", conv3.shape
		print "pool3 shape:", pool3.shape

		model = Model(input = inputs, output = conv10)

		model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

		return model


	def train(self):

		mydata = dataProcess(self.img_rows, self.img_cols)
		batch_size = 3
		steps_per_epoch =  int(len(mydata.imgs) * 0.8) // batch_size
		validation_steps = int(len(mydata.imgs) * 0.2) // batch_size


		# 加载模型
		print("loading data done")
		model = self.get_unet()
		print("got unet")

		# 开始训练
		model_checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')

		model.fit_generator( generator = mydata.generate_batch_data(batch_size),
							 steps_per_epoch = steps_per_epoch,
							 epochs=10,
							 verbose=1,
							 callbacks=[model_checkpoint],
							 validation_data=mydata.get_valid_data(batch_size),
							 validation_steps=validation_steps
							 )


if __name__ == '__main__':
	myunet = myUnet(512, 512)
	myunet.train()




