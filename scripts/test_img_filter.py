import cv2
import os

#载入图片
# img_path = "/home/speed_kai/yeyunfan_phd/COCO_datasets/COCO_COCO_2017_Val_images/val2017/000000289229.jpg"
img_path = "/home/speed_kai/yeyunfan_phd/COCO_datasets/COCO_COCO_2017_Val_images/val2017/image/000000000872.jpg"
img_data=cv2.imread(img_path)
# img_original = cv2.GaussianBlur(img_original, (5, 5), 1, 0)
#设置窗口
cv2.namedWindow('biteral')
#定义回调函数
def nothing(x):
    pass
#创建两个滑动条，分别控制radius, threshold1，threshold2
# cv2.createTrackbar('radius','biteral', 0,100, nothing)
cv2.createTrackbar('sigmaColor','biteral',0,200,nothing)
cv2.createTrackbar('sigmaSpace','biteral',0,200,nothing)
while(1):
    #返回滑动条所在位置的值

    #radius = cv2.getTrackbarPos('radius', 'biteral')
    threshold1 = cv2.getTrackbarPos('sigmaColor','biteral')
    threshold1 = cv2.getTrackbarPos('sigmaSpace','biteral')

    img_blur = cv2.bilateralFilter(img_data, 15, threshold1, threshold1)
    #显示图片
    cv2.imshow('original', img_data)
    cv2.imshow('bilateralFilter', img_blur)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()