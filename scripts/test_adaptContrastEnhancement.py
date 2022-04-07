# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
from tqdm import tqdm
np.seterr(divide='ignore',invalid='ignore')


def getVarianceMean(scr, winSize):
    if scr is None or winSize is None:
        print("The input parameters of getVarianceMean Function error")
        return -1
    if winSize % 2 == 0:
        print("The window size should be singular")
        return -1
    copyBorder_map = cv2.copyMakeBorder(scr, winSize // 2, winSize // 2, winSize // 2, winSize // 2,
                                        cv2.BORDER_REPLICATE)
    shape = np.shape(scr)
    local_mean = np.zeros_like(scr)
    local_std = np.zeros_like(scr)

    for i in range(shape[0]):
        for j in range(shape[1]):
            temp = copyBorder_map[i:i + winSize, j:j + winSize]
            local_mean[i, j], local_std[i, j] = cv2.meanStdDev(temp)
            if local_std[i, j] <= 0:
                local_std[i, j] = 1e-8
    return local_mean, local_std


def adaptContrastEnhancement(scr, winSize, maxCg):
    if scr is None or winSize is None or maxCg is None:
        print("The input parameters of ACE Function error")
        return -1

    YUV_img = cv2.cvtColor(scr, cv2.COLOR_BGR2YUV)  ##转换通道
    Y_Channel = YUV_img[:, :, 0]
    shape = np.shape(Y_Channel)
    meansGlobal = cv2.mean(Y_Channel)[0]

    ### 这里提供使用boxfilter 计算局部均质和方差的方法
    localMean_map=cv2.boxFilter(Y_Channel,-1,(winSize,winSize),normalize=True)
    localVar_map=cv2.boxFilter(np.multiply(Y_Channel,Y_Channel),-1,(winSize,winSize),normalize=True)-np.multiply(localMean_map,localMean_map)
    greater_Zero=localVar_map>0
    localVar_map=localVar_map*greater_Zero+1e-8
    localStd_map = np.sqrt(localVar_map)

    # localMean_map, localStd_map = getVarianceMean(Y_Channel, winSize)
    for i in range(shape[0]):
        for j in range(shape[1]):
            cg = 0.2 * meansGlobal / localStd_map[i, j];
            if cg > maxCg:
                cg = maxCg
            elif cg < 1:
                cg = 1
            temp = Y_Channel[i, j].astype(float)
            temp = max(0, min(localMean_map[i, j] + cg * (temp - localMean_map[i, j]), 255))

            #            Y_Channel[i,j]=max(0,min(localMean_map[i,j]+cg*(Y_Channel[i,j]-localMean_map[i,j]),255))
            Y_Channel[i, j] = temp
    YUV_img[:, :, 0] = Y_Channel
    dst = cv2.cvtColor(YUV_img, cv2.COLOR_YUV2BGR)
    return dst


def get_imgs_list(imgs_dir):
    return [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if
            f.endswith('.jpg') or f.endswith('.png') or f.endswith('.pgm')]


imgs_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/datasets/" \
            "BIPED/image"
target_dir = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/datasets/" \
             "BIPED/image_filters_results/images_contrast"
imgs = get_imgs_list(imgs_path)
for i in tqdm(range(len(imgs))):
    if i == 200:
        break
    img_name = os.path.basename(imgs[i])
    img_data = cv2.imread(imgs[i])
    dstimg = adaptContrastEnhancement(img_data, 15, 15)
    cv2.imwrite(os.path.join(target_dir, img_name), dstimg)

