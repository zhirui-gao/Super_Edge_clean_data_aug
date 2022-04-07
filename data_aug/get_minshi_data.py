import json
import os
import cv2
import numpy as np
import colorsys
import random


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors


def get_imgs_list(imgs_dir):
    return [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if
            f.endswith('.bmp') or f.endswith('.jpg') or f.endswith('.png')]


def vis_labelme(json_path, img_path, writed_img_path, writed_edge_path):
    data = json.load(open(json_path))
    img = cv2.imread(img_path)

    current_labels = []
    for shape in data['shapes']:
        if shape['label'] not in current_labels:
            current_labels.append(shape['label'])
    colors = ncolors(len(current_labels))

    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        poly = np.array(points, dtype=np.int32)
        ellipse = cv2.fitEllipse(poly)
        edge_img = np.zeros([img.shape[0], img.shape[1]], np.uint8)
        axesSize = (int(ellipse[1][0]/2), int(ellipse[1][1]/2))
        rotateAngle = ellipse[2]
        startAngle = 0
        endAngle = 360
        point_color = (255, 255, 255)  # BGR
        thickness = 1
        lineType = 4
        cv2.ellipse(edge_img, (int(ellipse[0][0]), int(ellipse[0][1])),axesSize, rotateAngle, startAngle, endAngle, point_color, thickness, lineType)
    cv2.imwrite(writed_edge_path, edge_img)
    cv2.imwrite(writed_img_path, img)


base_dir = "E:/deep_laerning_dataset/initial_labeled_imgs"
img_dir = os.path.join(base_dir,'bot_small_circle','train_raw')
imgs = get_imgs_list(img_dir)

target_path = os.path.join(base_dir, 'visualize')
if not os.path.exists(target_path):
    os.makedirs(target_path)
cnt = 568
for i, img_path in enumerate(imgs):
    json_path = img_path[:-4] + ".json"
    writed_img_path = os.path.join(base_dir, 'data_for_edge_detection', 'image', str(cnt)+'.png')
    writed_edge_path = os.path.join(base_dir, 'data_for_edge_detection', 'edge', str(cnt)+'.png')
    if os.path.exists(json_path):
        vis_labelme(json_path, img_path, writed_img_path, writed_edge_path)
    cnt = cnt + 1
print(cnt)