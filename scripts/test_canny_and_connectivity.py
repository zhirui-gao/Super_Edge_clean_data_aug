import numpy as np
from skimage import measure
import cv2
from tqdm import tqdm
import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utils.canny import filter_canny_connectivity

def get_imgs_list(imgs_dir):
    return [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if
            f.endswith('.jpg') or f.endswith('.png') or f.endswith('.pgm')]


imgs_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/datasets/" \
            "BIPED/image_filters_results/images_l0smooth"

target_dir = os.path.join(imgs_path, "canny")
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

imgs = get_imgs_list(imgs_path)
for i in tqdm(range(len(imgs))):
    if i == 200:
        break
    img_name = os.path.basename(imgs[i])
    img_data = cv2.imread(imgs[i])
    canny = cv2.Canny(img_data, 20, 40)
    canny = filter_canny_connectivity(canny, min_thresh=15)
    cv2.imwrite(os.path.join(target_dir, img_name), canny)





# # min_thresh = 10
# canny_edges_dir = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/datasets/BIPED/pseudo_label_conf_with_blur/after_round5/edges"
# output_dir = canny_edges_dir + "_filter"
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)
# for i, edge_name in enumerate(os.listdir(canny_edges_dir)):
#     if i == 200:
#         break
#
#     edge_path = os.path.join(canny_edges_dir, edge_name)
#     canny_edge = cv2.imread(edge_path)
#     '''
#     canny_gray = cv2.cvtColor(canny_edge.copy(), cv2.COLOR_BGR2GRAY)
#     canny_gray[canny_gray > 20] = 255
#     # print(canny_edge.shape)
#
#     labels = measure.label(canny_gray, connectivity=2)  # connectivity=2是8连通区域标记，等于1则是4连通
#     props = measure.regionprops(labels)
#     print("total area:", len(props))
#
#     num_15 = 0
#     num_20 = 0
#     num_30 = 0
#     num_40 = 0
#     num_50 = 0
#     for each_area in props:
#         if each_area.area <= 50:
#             num_50 += 1
#             for each_point in each_area.coords:
#                 canny_edge[each_point[0]][each_point[1]] = (0, 0, 255)
#         if each_area.area <= 40:
#             num_40 += 1
#             for each_point in each_area.coords:
#                 canny_edge[each_point[0]][each_point[1]] = (0, 0, 255)
#         if each_area.area <= 30:
#             num_30 += 1
#             for each_point in each_area.coords:
#                 canny_edge[each_point[0]][each_point[1]] = (255, 255, 0)
#         if each_area.area <= 20:
#             num_20 += 1
#             for each_point in each_area.coords:
#                 canny_edge[each_point[0]][each_point[1]] = (255, 0, 0)
#         if each_area.area <= 15:
#             num_15 += 1
#             for each_point in each_area.coords:
#                 canny_edge[each_point[0]][each_point[1]] = (122, 122, 122)
#     text = "<50: " + str(num_50) + ", < 40: " + str(num_40) + ", < 30: " + str(num_30) + ", < 20: " + str(num_20) + ", < 15: " + str(num_15)
#     cv2.putText(canny_edge, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.putText(canny_edge, "total: " + str(len(props)), (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
#     '''
#     cv2.imwrite(os.path.join(output_dir, edge_name), canny_edge)



