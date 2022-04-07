import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import os
from skimage import measure

def aug_single_img(img, edge_map, augmenter):
    images = img[np.newaxis, :, :]
    # edge_map[edge_map == 255] = 1
    segmaps = edge_map[np.newaxis, :, :, np.newaxis]
    # print("images and segmaps:", images.shape, segmaps.shape)   # (1, 384, 544, 3) (1, 384, 544, 1)
    images_aug, segmaps_aug = augmenter(images=images, segmentation_maps=segmaps)
    # print("each image and segmap:", images_aug[0].shape, segmaps_aug[0].shape)    # (384, 544, 3) (384, 544, 1)
    segmaps_aug = np.concatenate((segmaps_aug[0], segmaps_aug[0], segmaps_aug[0]), axis=-1)
    # print(segmaps_aug.shape)   # (384, 544, 3)
    return images_aug[0], segmaps_aug


# img_path = "/home/speed_kai/yeyunfan_phd/edge_detection/render_datasets/output/all_dense_aug/coco_data/rgb_0001_aug_0001.png"
img_path = "test/000000000724.jpg"
img_data = cv2.imread(img_path)

edge_path = "test/000000000724.png"
edge_data = np.array(cv2.imread(edge_path, 0), dtype=np.int32)


height, width = img_data.shape[:2]


aug_crop_mid = iaa.Sequential([iaa.Crop(percent=0.2, keep_size=False)])
img_aug_mid, edge_aug_mid = aug_single_img(img_data, edge_data, aug_crop_mid)

cut_ratio_max = 0.7
if height <= width:
    cut_ratio = min(1 - height / width, cut_ratio_max)
    # 如果percent=是一个4个元素的tuple,那么4个元素分别代表(top, right, bottom, left)
    aug_crop_p1 = iaa.Sequential([iaa.Crop(percent=(0, cut_ratio, 0, 0), keep_size=False)])
    img_aug_p1, edge_aug_p1 = aug_single_img(img_data, edge_data, aug_crop_p1)
    aug_crop_p2 = iaa.Sequential([iaa.Crop(percent=(0, 0, 0, cut_ratio), keep_size=False)])
    img_aug_p2, edge_aug_p2 = aug_single_img(img_data, edge_data, aug_crop_p2)
else:
    cut_ratio = min(1 - width / height, cut_ratio_max)
    aug_crop_p1 = iaa.Sequential([iaa.Crop(percent=(cut_ratio, 0, 0, 0), keep_size=False)])
    img_aug_p1, edge_aug_p1 = aug_single_img(img_data, edge_data, aug_crop_p1)

    aug_crop_p2 = iaa.Sequential([iaa.Crop(percent=(0, 0, cut_ratio, 0), keep_size=False)])
    img_aug_p2, edge_aug_p2 = aug_single_img(img_data, edge_data, aug_crop_p2)

cv2.imwrite("test/000000000724_aug_mid.jpg", img_aug_mid)
cv2.imwrite("test/000000000724_aug_mid.png", edge_aug_mid)

cv2.imwrite("test/000000000724_aug_p1.jpg", img_aug_p1)
cv2.imwrite("test/000000000724_aug_p1.png", edge_aug_p1)

cv2.imwrite("test/000000000724_aug_p2.jpg", img_aug_p2)
cv2.imwrite("test/000000000724_aug_p2.png", edge_aug_p2)

print(np.arange(0, 360, 22.5))

