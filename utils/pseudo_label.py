import numpy as np
import cv2
import os
from tqdm import tqdm
from .functions import get_imgs_list, fit_img_postfix
from .canny import filter_canny_connectivity
# from skimage import measure


def create_pseudo_label_with_canny(img_data, pred_data, blur=True):
    # use bilateralFilter to img_data for clearer and connected edge
    if blur:
        # for _ in range(2):
        #     img_data = cv2.bilateralFilter(img_data, 15, 50, 50)
        img_data = cv2.GaussianBlur(img_data, (5, 5), 0)
    # --------------------------------------------------------------
    pred_data_binary = cv2.adaptiveThreshold(pred_data, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 20)
    pred_data_binary = cv2.bitwise_not(pred_data_binary)
    pred_data_binary = filter_canny_connectivity(pred_data_binary, min_thresh=100)
    pred_data_binary[pred_data_binary == 255] = 1
    (h, w) = pred_data.shape[:2]
    merge_result = np.zeros((h, w), dtype=np.float64)
    # canny_ranges = [(10, 20), (20, 40), (30, 60), (40, 80), (50, 100)]
    canny_ranges = [(20, 40)]
    for j, canny_range in enumerate(canny_ranges):
        edge_canny = cv2.Canny(img_data, canny_range[0], canny_range[1])
        merge_tmp = pred_data_binary * edge_canny
        merge_result = cv2.add(merge_result, merge_tmp / len(canny_ranges))
    merge_result = filter_canny_connectivity(merge_result, min_thresh=15)
    return merge_result


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


def create_pseudo_label_with_more_blur_canny(img_data, pred_data):
    pred_data_binary = cv2.adaptiveThreshold(pred_data, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 20)
    pred_data_binary = cv2.bitwise_not(pred_data_binary)
    pred_data_binary = filter_canny_connectivity(pred_data_binary, min_thresh=100)
    pred_data_binary[pred_data_binary == 255] = 1
    (h, w) = pred_data.shape[:2]

    img_blur_g = cv2.GaussianBlur(img_data, (5, 5), 0)
    img_blur_g_b1 = cv2.bilateralFilter(img_blur_g, 15, 50, 50)
    img_blur_g_b2 = cv2.bilateralFilter(img_blur_g_b1, 15, 50, 50)
    imgs_data = [img_blur_g_b1, img_blur_g_b2]

    merge_result = np.zeros((h, w), dtype=np.float64)
    canny_ranges = [(20, 40)]
    for j, canny_range in enumerate(canny_ranges):
        edge_canny = use_more_canny(imgs_data, canny_range[0], canny_range[1])
        merge_tmp = pred_data_binary * edge_canny
        merge_result = cv2.add(merge_result, merge_tmp / len(canny_ranges))
    merge_result = filter_canny_connectivity(merge_result, min_thresh=30)
    return merge_result


def create_pseudo_label_with_l0_blur_canny(img_data, pred_data):
    pred_data_binary = cv2.adaptiveThreshold(pred_data, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 20)
    pred_data_binary = cv2.bitwise_not(pred_data_binary)
    pred_data_binary = filter_canny_connectivity(pred_data_binary, min_thresh=100)
    pred_data_binary[pred_data_binary == 255] = 1
    (h, w) = pred_data.shape[:2]

    img_blur = cv2.GaussianBlur(img_data, (5, 5), 0)
    img_blur = cv2.ximgproc.l0Smooth(img_blur, kappa=2)
    imgs_data = [img_blur]

    merge_result = np.zeros((h, w), dtype=np.float64)
    canny_ranges = [(20, 40)]
    for j, canny_range in enumerate(canny_ranges):
        edge_canny = use_more_canny(imgs_data, canny_range[0], canny_range[1])
        merge_tmp = pred_data_binary * edge_canny
        merge_result = cv2.add(merge_result, merge_tmp / len(canny_ranges))
    merge_result = filter_canny_connectivity(merge_result, min_thresh=30)
    return merge_result


def create_pseudo_label_with_more_blur_canny_new(img_data, pred_data):
    pred_data_binary = cv2.adaptiveThreshold(pred_data, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 20)
    pred_data_binary = cv2.bitwise_not(pred_data_binary)
    pred_data_binary = filter_canny_connectivity(pred_data_binary, min_thresh=100)
    pred_data_binary[pred_data_binary == 255] = 1
    (h, w) = pred_data.shape[:2]

    merge_result = np.zeros((h, w), dtype=np.float64)
    img_blur_g = cv2.GaussianBlur(img_data, (5, 5), 0)
    img_blur_g_b1 = cv2.bilateralFilter(img_blur_g, 15, 50, 50)
    img_blur_g_b2 = cv2.bilateralFilter(img_blur_g_b1, 15, 50, 50)
    imgs_data = [img_blur_g_b1, img_blur_g_b2]

    canny_ranges = [(20, 40)]
    for j, canny_range in enumerate(canny_ranges):
        edge_canny = use_more_canny(imgs_data, canny_range[0], canny_range[1])
        merge_tmp = pred_data_binary * edge_canny
        merge_result = cv2.add(merge_result, merge_tmp / len(canny_ranges))
    # merge_result = filter_canny_connectivity(merge_result, min_thresh=30)
    # --------------------new filter_canny_connectivity--------------------
    labels = measure.label(merge_result, connectivity=2)  # connectivity=2是8连通区域标记，等于1则是4连通
    props = measure.regionprops(labels)
    min_thresh = int(len(props) / 10)

    for each_area in props:
        if each_area.area <= min_thresh:
            for each_point in each_area.coords:
                merge_result[each_point[0]][each_point[1]] = 0

    # for each_area in props:
    #     if each_area.area <= int(min_thresh / 3):
    #         for each_point in each_area.coords:
    #             merge_result[each_point[0]][each_point[1]] = 0
    #     elif each_area.area <= int(min_thresh / 3 * 2):
    #         for each_point in each_area.coords:
    #             merge_result[each_point[0]][each_point[1]] *= 1 / 4
    #     elif each_area.area <= min_thresh:
    #         for each_point in each_area.coords:
    #             merge_result[each_point[0]][each_point[1]] *= 1 / 2
    return merge_result
