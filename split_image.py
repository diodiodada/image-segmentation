# -*- coding: utf-8 -*-
import os
from PIL import Image


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

        for r in range(rownum): #竖着平移

            x = 0

            for c in range(colnum): #横着平移
                box = (x,
                       y,
                       x + width,
                       y + hight
                       )

                img.crop(box).save(os.path.join(dstpath, basename + '_' + str(num) + '.' + ext), ext)
                num = num + 1
                x = x + step_x

            y = y + step_y
            print('处理完一行。')

        print('图片切割完毕，共生成 %s 张小图片。' % num)
    else:
        print('不合法的行列切割参数！')



width = 512
hight = width

out_x = "/home1/zj/segmentation/x"
out_y = "/home1/zj/segmentation/y"

if not os.path.lexists(out_x):
    os.makedirs(out_x)

if not os.path.lexists(out_y):
    os.makedirs(out_y)

rownum = (5142//hight)*2
colnum = (5664//width)*2
# 第一张
print("%d 行, %d 列" % (rownum, colnum))
src = "data/train-x-1.png"
dstpath = out_x
print("开始处理"+src)
splitimage(src=src,dstpath=dstpath, width=width,hight=hight,rownum=rownum,colnum=colnum)

src = "data/train-y-1.png"
dstpath = out_y
print("开始处理"+src)
splitimage(src=src,dstpath=dstpath,width=width,hight=hight,rownum=rownum,colnum=colnum)


rownum = (2470//hight+1)*2
colnum = (4011//width+1)*2
# 第二张
print("%d 行, %d 列" % (rownum, colnum))
src = "data/train-x-2.png"
dstpath = out_x
print("开始处理"+src)
splitimage(src=src,dstpath=dstpath,width=width,hight=hight,rownum=rownum,colnum=colnum)

src = "data/train-y-2.png"
dstpath = out_y
print("开始处理"+src)
splitimage(src=src, dstpath=dstpath,width=width,hight=hight,rownum=rownum,colnum=colnum)

rownum = (6116//hight)*2
colnum = (3357//width)*2
# 第三张
print("%d 行, %d 列" % (rownum, colnum))
src = "data/train-x-3.png"
dstpath = out_x
print("开始处理"+src)
splitimage(src=src,dstpath=dstpath,width=width,hight=hight,rownum=rownum,colnum=colnum)

src = "data/train-y-3.png"
dstpath = out_y
print("开始处理"+src)
splitimage(src=src, dstpath=dstpath,width=width,hight=hight,rownum=rownum,colnum=colnum)
