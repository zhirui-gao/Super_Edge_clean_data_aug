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


def vis_labelme(json_path, img_path, writed_path):
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
        x_axis = []
        y_axis = []
        for point in points:
            x_axis.append(point[0])
            y_axis.append(point[1])
        x = round(x_axis[y_axis.index(min(y_axis))])
        y = round(min(y_axis))
        chosen_color = colors[current_labels.index(label)]
        cv2.putText(img, shape['label'], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        # print(points)
        cv2.fillPoly(img, [np.array(points, "int32")], chosen_color)

    cv2.imwrite(writed_path, img)
    # cv2.imshow('imshow',lab_ok)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


base_dir = "E:/deep_laerning_dataset/initial_labeled_imgs/bot_small_circle/train_raw"
imgs = get_imgs_list(base_dir)

target_path = os.path.join(base_dir, 'visualize')
if not os.path.exists(target_path):
    os.makedirs(target_path)
for i, img_path in enumerate(imgs):
    if i == 100:
        break
    json_path = img_path[:-4] + ".json"
    writed_path = os.path.join(base_dir, 'visualize', os.path.basename(img_path))
    # print(json_path)
    # print(img_path)
    # print(writed_path)
    if os.path.exists(json_path):
        vis_labelme(json_path, img_path, writed_path)
