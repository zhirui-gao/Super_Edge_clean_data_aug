import os
import cv2
import numpy as np
import shutil
import torch
import torch.nn as nn
from tqdm import tqdm
import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utils import create_pseudo_label_with_canny
from utils.canny import filter_canny_connectivity


def get_imgs_list(imgs_dir):
    return [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if
            f.endswith('.jpg') or f.endswith('.png') or f.endswith('.pgm')]


def fit_img_postfix(img_path):
    if not os.path.exists(img_path) and img_path.endswith(".jpg"):
        img_path = img_path[:-4] + ".png"
    if not os.path.exists(img_path) and img_path.endswith(".png"):
        img_path = img_path[:-4] + ".jpg"
    return img_path


def get_uncertainty(preds):
    new_preds = []
    for each_pred in preds:
        if len(each_pred.shape) == 2:
            each_pred = np.expand_dims(each_pred, axis=0)
        new_preds.append(each_pred)
    cat = np.concatenate(new_preds, axis=0)     # cat.shape: (3, 720, 1280)
    std = cat.std(axis=0)                       # std.shape: (720, 1280)
    return std


def use_more_canny(imgs, t_min, t_max):
    # (h, w) = imgs[0].shape[:2]
    # canny_final = np.zeros((h, w), dtype=np.float64)
    # for each_img in imgs:
    #     canny_temp = cv2.Canny(each_img, t_min, t_max)
    #     canny_final = canny_final + canny_temp
    for index, each_img in enumerate(imgs):
        if index == 0:
            canny_final = cv2.Canny(each_img, t_min, t_max)
        else:
            canny_tmp = cv2.Canny(each_img, t_min, t_max)
            canny_final = cv2.add(canny_final, canny_tmp)
    return canny_final


imgs_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/datasets/" \
            "BIPED/image"

base_dir = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/" \
           "checkpoints/BIPED_10%_no_iterative/24/results_scale512"
preds_path = os.path.join(base_dir, "avg")
preds = get_imgs_list(preds_path)

imgs_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/datasets/" \
            "BIPED/image"
target_dir = os.path.join(imgs_path, "adaptiveThreshold")
imgs = get_imgs_list(imgs_path)
for i in tqdm(range(len(imgs))):
    if i == 200:
        break
    img_name = os.path.basename(imgs[i])
    img_data = cv2.imread(imgs[i], 0)

    img_binary = cv2.adaptiveThreshold(img_data, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)
    img_binary = cv2.bitwise_not(img_binary)
    img_binary = filter_canny_connectivity(img_binary, min_thresh=100)
    cv2.imwrite(os.path.join(target_dir, img_name), img_binary)

# output_path_bi = preds_path + "_pred_data_binary"
# if not os.path.exists(output_path_bi):
#     os.mkdir(output_path_bi)
# output_path = os.path.join(base_dir, "avg_merge_canny_test")
# if not os.path.exists(output_path):
#     os.mkdir(output_path)
#
# for i in tqdm(range(len(preds))):
#     if i == 20:
#         break
#     img_name = os.path.basename(preds[i])
#     img_data = cv2.imread(fit_img_postfix(os.path.join(imgs_path, img_name)))
#     pred_data = cv2.imread(preds[i], 0)
#
#     # for _ in range(2):
#     #     img_data = cv2.bilateralFilter(img_data, 15, 50, 50)
#     # img_data = cv2.GaussianBlur(img_data, (5, 5), 0)
#
#     pred_data_binary = cv2.adaptiveThreshold(pred_data, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 20)
#
#     pred_data_binary = cv2.bitwise_not(pred_data_binary)
#     pred_data_binary = filter_canny_connectivity(pred_data_binary, min_thresh=100)
#
#
#     cv2.imwrite(os.path.join(output_path_bi, img_name), pred_data_binary)
#     pred_data_binary[pred_data_binary == 255] = 1
#     (h, w) = pred_data.shape[:2]
#     merge_result = np.zeros((h, w), dtype=np.float64)
#     canny_ranges = [(20, 40), (40, 80), (80, 120), (120, 160)]
#
#     # canny_ranges = [(40, 80)]
#     for j, canny_range in enumerate(canny_ranges):
#         edge_canny = cv2.Canny(img_data, canny_range[0], canny_range[1])
#         merge_tmp = pred_data_binary * edge_canny
#         # cv2.imwrite(os.path.join(output_path, img_name[:-4]+"_"+str(j)+".png"), merge_tmp)
#         merge_result = cv2.add(merge_result, merge_tmp / len(canny_ranges))
#
#
#     # merge_result = cv2.dilate(merge_result, kernel)
#     # merge_result = cv2.dilate(merge_result, kernel)
#     # merge_result = cv2.erode(merge_result, kernel)
#     # merge_result = cv2.erode(merge_result, kernel)
#     # merge_result = filter_canny_connectivity(merge_result, min_thresh=5)
#     merge_result = filter_canny_connectivity(merge_result, min_thresh=10)
#
#     # merge_result = create_pseudo_label_with_canny(img_data, pred_data)
#     cv2.imwrite(os.path.join(output_path, img_name), merge_result)
