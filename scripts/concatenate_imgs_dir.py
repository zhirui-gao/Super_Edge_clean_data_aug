import os
import cv2
import numpy as np
import math
from tqdm import tqdm


def get_imgs_list(imgs_dir):
    return [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.pgm')]


def concatenate_images(final_result, row, column, interval):
    (h, w, _) = final_result[0].shape  # 默认所有的图都是一个尺寸啊

    # 把数量不够的内容用黑图补上
    for i in range(row * column - len(final_result)):
        final_result.append(np.zeros((h, w, 3), dtype=np.uint8))

    # 先创建一个全黑大图，然后再一个个重新覆盖进去
    concat_result = np.zeros((row * h + interval * (row - 1), column * w + interval * (column - 1), 3), dtype=np.uint8)

    for r in range(row):
        for c in range(column):
            index = r * column + c
            # print(index)
            range_h1 = r * h + r * interval
            range_h2 = (r + 1) * h + r * interval
            range_w1 = c * w + c * interval
            range_w2 = (c + 1) * w + c * interval
            concat_result[range_h1: range_h2, range_w1: range_w2] = final_result[index]
    return concat_result


def concatenate_images(final_result, row, column, interval, strategy="left2right"):
    (h, w, _) = final_result[0].shape  # 默认所有的图都是一个尺寸啊

    # 把数量不够的内容用黑图补上
    for _ in range(row * column - len(final_result)):
        final_result.append(np.zeros((h, w, 3), dtype=np.uint8))

    # 先创建一个全黑大图，然后再一个个重新覆盖进去
    concat_result = np.zeros((row * h + interval * (row - 1), column * w + interval * (column - 1), 3), dtype=np.uint8)

    for r in range(row):
        for c in range(column):
            if strategy == "left2right":
                index = r * column + c
            if strategy == "top2down":
                index = c * row + r
            # print(index)
            if len(final_result[index].shape) != 3:
                final_result[index] = cv2.cvtColor(final_result[index], cv2.COLOR_GRAY2BGR)
            range_h1 = r * h + r * interval
            range_h2 = (r + 1) * h + r * interval
            range_w1 = c * w + c * interval
            range_w2 = (c + 1) * w + c * interval
            concat_result[range_h1: range_h2, range_w1: range_w2] = final_result[index]
    return concat_result

# ---------------------------------------------------------------------------------------------------------
# imgs_path = "/home/speed_kai/yeyunfan_phd/COCO_datasets/COCO_COCO_2017_Val_images/val2017_pseudo_label_strict/image"
# gts_path = "/home/speed_kai/yeyunfan_phd/COCO_datasets/COCO_COCO_2017_Val_images/val2017_pseudo_label_strict/edge"
# preds_author = "/home/speed_kai/yeyunfan_phd/COCO_datasets/COCO_COCO_2017_Val_images/val2017_pseudo_label_strict/preds/fused"


# imgs_path = "/home/speed_kai/yeyunfan_phd/edge_detection/datasets/BIPED/edges/imgs/test/rgbr"
# gts_path = "/home/speed_kai/yeyunfan_phd/edge_detection/datasets/BIPED/edges/edge_maps/test/rgbr"
# preds_author = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/edges-eva/results/biped_biped_epoch19/edges_pred"
# base = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/edges-eva/results/biped_render/edges_pred"
#
# coco_1 = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/edges-eva/results/biped_coco_epoch1/edges_pred"
# coco_3 = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/edges-eva/results/biped_coco_epoch3/edges_pred"
# coco_large1 = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/edges-eva/results/biped_coco_large_epoch1/edges_pred"
# coco_large3 = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/edges-eva/results/biped_coco_large_epoch3/edges_pred"
#
# finetune5_epoch_1 = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/edges-eva/results/biped_coco_large_finetune5%_epoch1/edges_pred"
# finetune5_epoch_5 = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/edges-eva/results/biped_coco_large_finetune5%_epoch5/edges_pred"
# finetune5_epoch_10 = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/edges-eva/results/biped_coco_large_finetune5%_epoch10/edges_pred"
# finetune5_epoch_20 = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/edges-eva/results/biped_coco_large_finetune5%_epoch20/edges_pred"
#
# finetune10_epoch_1 = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/edges-eva/results/biped_coco_large_finetune10%_epoch1/edges_pred"
# finetune10_epoch_5 = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/edges-eva/results/biped_coco_large_finetune10%_epoch5/edges_pred"
# finetune10_epoch_10 = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/edges-eva/results/biped_coco_large_finetune10%_epoch10/edges_pred"
# finetune10_epoch_20 = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/edges-eva/results/biped_coco_large_finetune10%_epoch20/edges_pred"
#
#
# target_dir = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge/result/BIPED/visualize_4*4"
# if not os.path.exists(target_dir):
#     os.mkdir(target_dir)
# list_dirs = [imgs_path, gts_path, preds_author, base,
#              coco_1, coco_3, coco_large1, coco_large3,
#              finetune5_epoch_1, finetune5_epoch_5, finetune5_epoch_10, finetune5_epoch_20,
#              finetune10_epoch_1, finetune10_epoch_5, finetune10_epoch_10, finetune10_epoch_20]
# list_names = ["raw img", "gt", "preds_author", "preds_base",
#               "coco_1", "coco_3", "coco_large1", "coco_large3",
#               "5%_1", "5%_5", "5%_10", "5%_20",
#               "10%_1", "10%_5", "10%_10", "10%_20"]
# assert len(list_dirs) == len(list_names)
# row = 4
# column = 4
# ---------------------------------------------------------------------------------------------------------

# # # -------------------------------------------------------------------------------------------------
# imgs_path = "/home/speed_kai/yeyunfan_phd/edge_detection/datasets/BIPED/edges/imgs/test/rgbr"
# gts_path = "/home/speed_kai/yeyunfan_phd/edge_detection/datasets/BIPED/edges/edge_maps/test/rgbr"
#
# base_dir = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge/result/BIPED"
#
# pred_from_author = os.path.join(base_dir, "biped_epoch19", "fused")
# preds1 = os.path.join(base_dir, "base_detector", "fused")
# preds3 = os.path.join(base_dir, "coco_epoch3", "fused")
# preds4 = os.path.join(base_dir, "coco_epoch10", "fused")
# preds5 = os.path.join(base_dir, "coco_epoch20", "fused")
# preds6 = os.path.join(base_dir, "coco_large_epoch3", "fused")
#
# list_dirs = [imgs_path, gts_path, pred_from_author, preds1, preds3, preds4, preds5, preds6]
# list_names = ["raw_img", "gt", "from_author", "base_detector", "coco_epoch3", "coco_epoch10", "coco_epoch20", "coco_large_epoch3"]
# row = 2
# column = 4
#
# target_dir = os.path.join(base_dir, "visualize")
# if not os.path.exists(target_dir):
#     os.mkdir(target_dir)
# assert len(list_dirs) == len(list_names)


# # -------------------------------------------------------------------------------------------------
base_dir = "D:/applications/baidu/BaiduNetdiskDownload/val2017_iterative_post_blur"
target_dir = os.path.join(base_dir, "visualize")
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
dir_names = ["start", "after_round1", "after_round2", "after_round3", "after_round4"]
# dir_names = ["start", "after_round1", "after_round2", "after_round3", "after_round4"]

imgs_path = os.path.join(base_dir, dir_names[0], "avg")
imgs = get_imgs_list(imgs_path)
for i in tqdm(range(len(imgs))):
    if i == 5000:
        break
    img = cv2.imread(imgs[i])
    final_result = []
    img_name = os.path.basename(imgs[i])
    for j in range(len(dir_names)):
        # path_temp = os.path.join(dir_names[j], img_name)
        dir_avg = os.path.join(base_dir, dir_names[j], "avg")
        dir_edges = os.path.join(base_dir, dir_names[j], "edges")

        img_avg = os.path.join(dir_avg, img_name)[:-4] + ".png"
        img_edges = os.path.join(dir_edges, img_name)[:-4] + ".png"

        img_data_avg = cv2.imread(img_avg)
        img_data_edges = cv2.imread(img_edges)
        # print(img_data_avg.shape, img_data_edges.shape)

        # print(img_tmp.shape, path_temp)
        cv2.putText(img_data_avg, dir_names[j], (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
        final_result.append(img_data_avg)
        final_result.append(img_data_edges)
    row = 2
    column = math.ceil(len(final_result) / row)
    concat_result = concatenate_images(final_result, row, column, interval=10, strategy="top2down")
    cv2.imwrite(os.path.join(target_dir, img_name), concat_result)




# ====================================================================================================
# imgs_path = "/home/speed_kai/yeyunfan_phd/edge_detection/datasets/BIPED/edges/imgs/test/rgbr"
# gts_path = "/home/speed_kai/yeyunfan_phd/edge_detection/datasets/BIPED/edges/edge_maps/test/rgbr"
# base_dir = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge/result/BIPED"
#
# pred_from_author = os.path.join(base_dir, "biped_epoch19", "fused")
# preds1 = os.path.join(base_dir, "base_detector", "fused")
# preds3 = os.path.join(base_dir, "coco_epoch3", "fused")
# preds4 = os.path.join(base_dir, "coco_epoch10", "fused")
# preds5 = os.path.join(base_dir, "coco_epoch20", "fused")
# preds6 = os.path.join(base_dir, "coco_large_epoch3", "fused")
#
# list_dirs = [imgs_path, gts_path, pred_from_author, preds1, preds3, preds4, preds5, preds6]
# list_names = ["raw_img", "gt", "from_author", "base_detector", "coco_epoch3", "coco_epoch10", "coco_epoch20", "coco_large_epoch3"]
# row = 2
# column = 4
#
# target_dir = os.path.join(base_dir, "visualize")
# if not os.path.exists(target_dir):
#     os.mkdir(target_dir)
# assert len(list_dirs) == len(list_names)
#
#
# imgs = get_imgs_list(imgs_path)
# for i in tqdm(range(len(imgs))):
#     if i == 100:
#         break
#
#     img = cv2.imread(imgs[i])
#     final_result = []
#
#     img_name = os.path.basename(imgs[i])
#     for j in range(len(list_dirs)):
#         path_temp = os.path.join(list_dirs[j], img_name)
#         if not os.path.exists(path_temp) and path_temp.endswith(".jpg"):
#             path_temp = path_temp[:-4] + ".png"
#         if not os.path.exists(path_temp) and path_temp.endswith(".png"):
#             path_temp = path_temp[:-4] + ".jpg"
#         img_tmp = cv2.imread(path_temp)
#         if "gt" in list_names[j]:
#             img_tmp = cv2.bitwise_not(img_tmp)
#         # print(img_tmp.shape, path_temp)
#         cv2.putText(img_tmp, list_names[j], (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 2)
#         final_result.append(img_tmp)
#     concat_result = concatenate_images(final_result, row, column, interval=10, strategy="left2right")
#     cv2.imwrite(os.path.join(target_dir, img_name), concat_result)






