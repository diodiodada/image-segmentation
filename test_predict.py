# -*- coding: utf-8 -*-
from unet0 import *
from data import *

padding = 64

test_size = 1024+padding*2

#if not os.path.lexists("results"):
	#os.mkdir("results")

mydata = dataProcess(test_size,test_size)


imgs_test = mydata.load_test_data_aug()
print(imgs_test.shape)
myunet = myUnet(test_size,test_size)
model = myunet.get_unet()


def test(weight_name,results_name):
	model.load_weights(weight_name)

	imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

	#========================================================================
	imgdatas = np.zeros((120, 1024 * 6, 1024 * 6, 1), dtype=np.float32)

	for i in range(120):
		for row in range(6):
			for col in range(6):
				imgdatas[i,
						row * 1024:(row + 1) * 1024,
						col * 1024:(col + 1) * 1024,
						:] = imgs_mask_test[i*36 + row*6 + col,
											padding:1024+padding,
											padding:1024+padding,
											:]

	np.save(results_name, imgdatas) 


test("1.hdf5","class1_results")
test("2.hdf5","class2_results")
test("3.hdf5","class3_results")
test("4.hdf5","class4_results")



