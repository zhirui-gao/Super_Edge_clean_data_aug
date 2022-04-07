import cv2
import os
import numpy as np
import math
from skimage import measure

def get_imgs_list(imgs_dir):
    return [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.pgm')]



edges_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/data_aug/" \
             "COCO_val2017_iterative_with_blur_consistency_mse0.001_batch4_size400_filter30_after_round4/edge/real"

target_path = edges_path + "_connectivity_vis"
if not os.path.exists(target_path):
    os.mkdir(target_path)

edges = get_imgs_list(edges_path)

c_100 = 0
for i, edge_path in enumerate(edges):
    if i == 5000:
        break

    edge_name = os.path.basename(edge_path)
    # print(edge_name)
    edge_data = cv2.imread(edge_path, 0)

    labels = measure.label(edge_data, connectivity=2)  # connectivity=2是8连通区域标记，等于1则是4连通
    props = measure.regionprops(labels)
    # print("total area:", len(props))
    if len(props) <= 100:
        c_100 += 1


        num_50 = 0
        for each_area in props:
            if each_area.area <= 50:
                num_50 += 1
        # print(num_50)

        edge_data = cv2.cvtColor(edge_data, cv2.COLOR_GRAY2BGR)
        text = "total: " + str(len(props))+ ", <=50: " + str(num_50)
        cv2.putText(edge_data, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(target_path, edge_name), edge_data)
    # break

print(c_100)
