# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

img_size = 512

#class1_frequency = np.load("npydata/class1_frequency_" + str(img_size) + "_aug.npy")
#class2_frequency = np.load("npydata/class2_frequency_" + str(img_size) + "_aug.npy")
#class3_frequency = np.load("npydata/class3_frequency_" + str(img_size) + "_aug.npy")
#class4_frequency = np.load("npydata/class4_frequency_" + str(img_size) + "_aug.npy")

class1_frequency = np.load("/home1/zj/segmentation/npydata/class1_frequency_" + str(img_size) + "_aug.npy")
class2_frequency = np.load("/home1/zj/segmentation/npydata/class2_frequency_" + str(img_size) + "_aug.npy")
class3_frequency = np.load("/home1/zj/segmentation/npydata/class3_frequency_" + str(img_size) + "_aug.npy")
class4_frequency = np.load("/home1/zj/segmentation/npydata/class4_frequency_" + str(img_size) + "_aug.npy")


c1 = (class1_frequency > 0).sum()
c2 = (class2_frequency > 0).sum()
c3 = (class3_frequency > 0).sum()
c4 = (class4_frequency > 0).sum()

plt.subplot(221)
plt.bar(range(len(class1_frequency)), class1_frequency)
plt.title('class1_frequency_'+ str(c1))

plt.subplot(222)
plt.bar(range(len(class2_frequency)), class2_frequency)
plt.title('class2_frequency_'+ str(c2))

plt.subplot(223)
plt.bar(range(len(class3_frequency)), class3_frequency)
plt.title('class3_frequency_'+ str(c3))

plt.subplot(224)
plt.bar(range(len(class4_frequency)), class4_frequency)
plt.title('class4_frequency_'+ str(c4))

plt.show()



# """
# 绘制多个子图
# 一个Figure对象可以包含多个子图（Axes），在matplotlib中用Axes对象表示一个绘图区域，称为子图，可以使用subplot()快速绘制包含多个子图的图表，它的调用形式如下：
# subplot(numRows,numCols,plotNum)
# 图表的整个绘图区域被等分为numRows行和numCols列，然后按照从左到下的顺序对每个区域进行编号，左上区域的编号为1。plotNum参数指定创建的Axes对象所在的区域
# """
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(1)#创建图表1
# plt.figure(2)#创建图表2
# ax1=plt.subplot(211)#在图表2中创建子图1
# ax2=plt.subplot(212)#在图表2中创建子图2
# x=np.linspace(0,3,100)
# for i in xrange(5):
#     plt.figure(1)
#     plt.plot(x,np.exp(i*x/3))
#
#     plt.sca(ax1)
#     plt.plot(x,np.sin(i*x))
#
#     plt.sca(ax2)
#     plt.plot(x,np.cos(i*x))
# plt.show()
















