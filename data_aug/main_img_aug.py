import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import os
from tqdm import tqdm
import math
import shutil
# augmenter = iaa.SomeOf((4, None), [
# augmenter = iaa.Sequential([
#     iaa.Crop(percent=(0, 0.4), keep_size=False),
#     iaa.Fliplr(p=0.5),
#     iaa.Flipud(p=0.5),
#     # iaa.Dropout(p=(0, 0.1)),
#     # iaa.CoarseDropout(0.02, size_percent=0.5),
#     iaa.GaussianBlur(sigma=(0, 3.0)),
#     # iaa.PerspectiveTransform(scale=(0, 0.1), keep_size=True),
#     # iaa.ElasticTransformation(alpha=(10, 20), sigma=(5, 10)),
#     iaa.Affine(
#         scale={"x": (0.5, 1.0), "y": (0.5, 1.0)},
#         # translate_percent={"x": (0, 0.2), "y": (0, 0.2)},
#         # translate_px={"x": -30},
#         #rotate=(-30, 30),
#         # shear=(-10, 10),
#         # fit_output=True,
#         #cval=71
#     )
# ], random_order=True)


def get_imgs_list(imgs_dir):
    return [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if
            f.endswith('.jpg') or f.endswith('.png') or f.endswith('.pgm')]


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


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_innder_square(raw_data_shape, img_aug, edge_aug, a):
    assert a > 0
    if a == 90 or a == 180 or a == 270 or a == 360:
        return img_aug, edge_aug
    while a > 90:
        a = a - 90
    min_wh = min(raw_data_shape[0], raw_data_shape[1])
    radian = math.radians(a)
    # print(radian)
    r = min_wh / (2 * math.cos(radian) * (1 + math.tan(radian)))
    center_h = img_aug.shape[0] / 2
    center_w = img_aug.shape[1] / 2
    h_min = math.ceil(center_h - r)
    h_max = math.floor(center_h + r)
    w_min = math.ceil(center_w - r)
    w_max = math.floor(center_w + r)
    img_aug_innder = img_aug[h_min: h_max, w_min: w_max]
    edge_aug_innder = edge_aug[h_min: h_max, w_min: w_max]
    return img_aug_innder, edge_aug_innder


def rotate_aug_dir(aug_imgs_dir, aug_edges_dir, rot_interval=90):
    wait_aug_lists = os.listdir(aug_imgs_dir)
    print("waiting rotate list:", wait_aug_lists)
    rot_interval = rot_interval
    for aug_list_dir in wait_aug_lists:
        if "rot" in aug_list_dir:
            continue
        print("processing " + aug_list_dir + " ... ...")
        imgs = get_imgs_list(os.path.join(aug_imgs_dir, aug_list_dir))
        # print(imgs)
        for index in tqdm(range(len(imgs))):
            img_data = cv2.imread(imgs[index])
            img_name = os.path.basename(imgs[index])
            # print(img_name)
            gt_path = os.path.join(aug_edges_dir, aug_list_dir, img_name)
            # print(gt_path)
            if not os.path.exists(gt_path) and gt_path.endswith(".jpg"):
                gt_path = gt_path[:-4] + ".png"
            if not os.path.exists(gt_path) and gt_path.endswith(".png"):
                gt_path = gt_path[:-4] + ".jpg"
            gt_data = np.array(cv2.imread(gt_path, 0), dtype=np.int32)

            for angle in np.arange(0, 360, rot_interval):
                if angle == 0:
                    continue
                # print("processing angle" + str(angle) + " ... ...")
                tmp_img_path = os.path.join(aug_imgs_dir, aug_list_dir + "_rot" + str(angle))
                create_dir(tmp_img_path)
                tmp_edge_path = os.path.join(aug_edges_dir, aug_list_dir + "_rot" + str(angle))
                create_dir(tmp_edge_path)

                aug_rotate = iaa.Sequential([iaa.Affine(rotate=angle, fit_output=True)])
                img_aug, edge_aug = aug_single_img(img_data, gt_data, aug_rotate)
                img_aug_innder, edge_aug_innder = get_innder_square(img_data.shape, img_aug, edge_aug, angle)

                img_name = img_name[:-4] + ".png"
                cv2.imwrite(os.path.join(tmp_img_path, img_name), img_aug_innder)
                cv2.imwrite(os.path.join(tmp_edge_path, img_name), edge_aug_innder)


def flip_aug_dir(aug_imgs_dir, aug_edges_dir):
    wait_aug_lists = os.listdir(aug_imgs_dir)
    print("waiting flip list:", wait_aug_lists)
    for aug_list_dir in wait_aug_lists:
        if "flip" in aug_list_dir:
            continue
        print("processing " + aug_list_dir + " ... ...")
        imgs = get_imgs_list(os.path.join(aug_imgs_dir, aug_list_dir))
        # print(imgs)
        for index in tqdm(range(len(imgs))):
            img_data = cv2.imread(imgs[index])
            img_name = os.path.basename(imgs[index])
            gt_path = os.path.join(aug_edges_dir, aug_list_dir, img_name)
            if not os.path.exists(gt_path) and gt_path.endswith(".jpg"):
                gt_path = gt_path[:-4] + ".png"
            if not os.path.exists(gt_path) and gt_path.endswith(".png"):
                gt_path = gt_path[:-4] + ".jpg"
            gt_data = np.array(cv2.imread(gt_path, 0), dtype=np.int32)

            tmp_img_path = os.path.join(aug_imgs_dir, aug_list_dir + "_flip")
            create_dir(tmp_img_path)
            tmp_edge_path = os.path.join(aug_edges_dir, aug_list_dir + "_flip")
            create_dir(tmp_edge_path)

            # --------------------------here is the main augmentation part-------------------------
            aug_flip = iaa.Sequential([iaa.Fliplr(p=1.0)])
            img_aug, edge_aug = aug_single_img(img_data, gt_data, aug_flip)
            # -------------------------------------------------------------------------------------
            img_name = img_name[:-4] + ".png"
            cv2.imwrite(os.path.join(tmp_img_path, img_name), img_aug)
            cv2.imwrite(os.path.join(tmp_edge_path, img_name), edge_aug)


def CoarseDropout_aug_dir(aug_imgs_dir, aug_edges_dir):
    wait_aug_lists = os.listdir(aug_imgs_dir)
    print("waiting drop list:", wait_aug_lists)
    for aug_list_dir in wait_aug_lists:
        if "drop" in aug_list_dir:
            continue
        print("processing " + aug_list_dir + " ... ...")
        imgs = get_imgs_list(os.path.join(aug_imgs_dir, aug_list_dir))

        for each_img in imgs:
            img_data = cv2.imread(each_img)
            img_name = os.path.basename(each_img)
            gt_path = os.path.join(aug_edges_dir, aug_list_dir, img_name)
            if not os.path.exists(gt_path) and gt_path.endswith(".jpg"):
                gt_path = gt_path[:-4] + ".png"
            if not os.path.exists(gt_path) and gt_path.endswith(".png"):
                gt_path = gt_path[:-4] + ".jpg"
            gt_data = np.array(cv2.imread(gt_path, 0), dtype=np.int32)

            tmp_img_path = os.path.join(aug_imgs_dir, aug_list_dir + "_drop")
            create_dir(tmp_img_path)
            tmp_edge_path = os.path.join(aug_edges_dir, aug_list_dir + "_drop")
            create_dir(tmp_edge_path)
            # --------------------------here is the main augmentation part-------------------------
            # aug_drop = iaa.Sequential([iaa.CoarseDropout(0.03, size_percent=0.5)])
            aug_drop = iaa.Sequential([iaa.CoarseDropout(0.03, size_percent=0.5)])
            img_aug, edge_aug = aug_single_img(img_data, gt_data, aug_drop)
            # -------------------------------------------------------------------------------------
            img_name = img_name[:-4] + ".png"
            cv2.imwrite(os.path.join(tmp_img_path, img_name), img_aug)
            cv2.imwrite(os.path.join(tmp_edge_path, img_name), edge_aug)


def GaussianBlur_aug_dir(aug_imgs_dir, aug_edges_dir):
    wait_aug_lists = os.listdir(aug_imgs_dir)
    print("waiting flip list:", wait_aug_lists)
    for aug_list_dir in wait_aug_lists:
        if "blur" in aug_list_dir:
            continue
        print("processing " + aug_list_dir + " ... ...")
        imgs = get_imgs_list(os.path.join(aug_imgs_dir, aug_list_dir))
        # print(imgs)
        for each_img in imgs:
            img_data = cv2.imread(each_img)
            img_name = os.path.basename(each_img)
            gt_path = os.path.join(aug_edges_dir, aug_list_dir, img_name)
            if not os.path.exists(gt_path) and gt_path.endswith(".jpg"):
                gt_path = gt_path[:-4] + ".png"
            if not os.path.exists(gt_path) and gt_path.endswith(".png"):
                gt_path = gt_path[:-4] + ".jpg"
            gt_data = np.array(cv2.imread(gt_path, 0), dtype=np.int32)

            tmp_img_path = os.path.join(aug_imgs_dir, aug_list_dir + "_blur")
            create_dir(tmp_img_path)
            tmp_edge_path = os.path.join(aug_edges_dir, aug_list_dir + "_blur")
            create_dir(tmp_edge_path)

            # --------------------------here is the main augmentation part-------------------------
            img_aug = cv2.GaussianBlur(img_data, (15, 15), 0)
            edge_aug = gt_data
            # -------------------------------------------------------------------------------------
            img_name = img_name[:-4] + ".png"
            cv2.imwrite(os.path.join(tmp_img_path, img_name), img_aug)
            cv2.imwrite(os.path.join(tmp_edge_path, img_name), edge_aug)

def Sharpen_aug_dir(aug_imgs_dir, aug_edges_dir):
    wait_aug_lists = os.listdir(aug_imgs_dir)
    print("waiting flip list:", wait_aug_lists)
    for aug_list_dir in wait_aug_lists:
        if "sharpen" in aug_list_dir:
            continue

        print("processing " + aug_list_dir + " ... ...")
        imgs = get_imgs_list(os.path.join(aug_imgs_dir, aug_list_dir))
        # print(imgs)
        for each_img in imgs:
            img_data = cv2.imread(each_img)
            img_name = os.path.basename(each_img)
            gt_path = os.path.join(aug_edges_dir, aug_list_dir, img_name)
            if not os.path.exists(gt_path) and gt_path.endswith(".jpg"):
                gt_path = gt_path[:-4] + ".png"
            if not os.path.exists(gt_path) and gt_path.endswith(".png"):
                gt_path = gt_path[:-4] + ".jpg"
            gt_data = np.array(cv2.imread(gt_path, 0), dtype=np.int32)

            tmp_img_path = os.path.join(aug_imgs_dir, aug_list_dir + "_sharpen")
            create_dir(tmp_img_path)
            tmp_edge_path = os.path.join(aug_edges_dir, aug_list_dir + "_sharpen")
            create_dir(tmp_edge_path)
            # --------------------------here is the main augmentation part-------------------------
            aug_drop = iaa.Sequential([iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))])
            img_aug, edge_aug = aug_single_img(img_data, gt_data, aug_drop)
            # -------------------------------------------------------------------------------------
            img_name = img_name[:-4] + ".png"
            cv2.imwrite(os.path.join(tmp_img_path, img_name), img_aug)
            cv2.imwrite(os.path.join(tmp_edge_path, img_name), edge_aug)


def crop_aug_dir(aug_imgs_dir, aug_edges_dir):
    wait_aug_lists = os.listdir(aug_imgs_dir)
    print("waiting drop list:", wait_aug_lists)
    for aug_list_dir in wait_aug_lists:
        if "crop" in aug_list_dir:
            continue
        print("processing " + aug_list_dir + " ... ...")
        imgs = get_imgs_list(os.path.join(aug_imgs_dir, aug_list_dir))

        for index in tqdm(range(len(imgs))):
            img_data = cv2.imread(imgs[index])
            img_name = os.path.basename(imgs[index])
            gt_path = os.path.join(aug_edges_dir, aug_list_dir, img_name)
            if not os.path.exists(gt_path) and gt_path.endswith(".jpg"):
                gt_path = gt_path[:-4] + ".png"
            if not os.path.exists(gt_path) and gt_path.endswith(".png"):
                gt_path = gt_path[:-4] + ".jpg"
            gt_data = np.array(cv2.imread(gt_path, 0), dtype=np.int32)

            # ---------------------------begin the cropping augmentation part-------------------------------
            height, width = img_data.shape[:2]
            # aug_crop_mid = iaa.Sequential([iaa.Crop(percent=0.2, keep_size=False)])
            # img_aug_mid, edge_aug_mid = aug_single_img(img_data, gt_data, aug_crop_mid)
            cut_ratio_max = 0.7
            if height <= width:
                cut_ratio = min(1 - height / width, cut_ratio_max)
                # 如果percent=是一个4个元素的tuple,那么4个元素分别代表(top, right, bottom, left)
                aug_crop_p1 = iaa.Sequential([iaa.Crop(percent=(0, cut_ratio, 0, 0), keep_size=False)])
                img_aug_p1, edge_aug_p1 = aug_single_img(img_data, gt_data, aug_crop_p1)
                aug_crop_p2 = iaa.Sequential([iaa.Crop(percent=(0, 0, 0, cut_ratio), keep_size=False)])
                img_aug_p2, edge_aug_p2 = aug_single_img(img_data, gt_data, aug_crop_p2)
            else:
                cut_ratio = min(1 - width / height, cut_ratio_max)
                aug_crop_p1 = iaa.Sequential([iaa.Crop(percent=(cut_ratio, 0, 0, 0), keep_size=False)])
                img_aug_p1, edge_aug_p1 = aug_single_img(img_data, gt_data, aug_crop_p1)
                aug_crop_p2 = iaa.Sequential([iaa.Crop(percent=(0, 0, cut_ratio, 0), keep_size=False)])
                img_aug_p2, edge_aug_p2 = aug_single_img(img_data, gt_data, aug_crop_p2)
            # ----------------------writing the crop1(left or top)----------------------
            tmp_img_path = os.path.join(aug_imgs_dir, aug_list_dir + "_crop1")
            create_dir(tmp_img_path)
            tmp_edge_path = os.path.join(aug_edges_dir, aug_list_dir + "_crop1")
            create_dir(tmp_edge_path)
            img_name = img_name[:-4] + ".png"
            cv2.imwrite(os.path.join(tmp_img_path, img_name), img_aug_p1)
            cv2.imwrite(os.path.join(tmp_edge_path, img_name), edge_aug_p1)
            # # ----------------------writing the crop2(right or bot)----------------------
            # tmp_img_path = os.path.join(aug_imgs_dir, aug_list_dir + "_crop2")
            # create_dir(tmp_img_path)
            # tmp_edge_path = os.path.join(aug_edges_dir, aug_list_dir + "_crop2")
            # create_dir(tmp_edge_path)
            # img_name = img_name[:-4] + ".png"
            # cv2.imwrite(os.path.join(tmp_img_path, img_name), img_aug_p2)
            # cv2.imwrite(os.path.join(tmp_edge_path, img_name), edge_aug_p2)
            # # ----------------------writing the crop3(mid)----------------------
            # tmp_img_path = os.path.join(aug_imgs_dir, aug_list_dir + "_crop3")
            # create_dir(tmp_img_path)
            # tmp_edge_path = os.path.join(aug_edges_dir, aug_list_dir + "_crop3")
            # create_dir(tmp_edge_path)
            # img_name = img_name[:-4] + ".png"
            # cv2.imwrite(os.path.join(tmp_img_path, img_name), img_aug_mid)
            # cv2.imwrite(os.path.join(tmp_edge_path, img_name), edge_aug_mid)

def Contrast_aug_dir(aug_imgs_dir, aug_edges_dir):
    wait_aug_lists = os.listdir(aug_imgs_dir)
    print("waiting flip list:", wait_aug_lists)
    for aug_list_dir in wait_aug_lists:
        if "Contrast" in aug_list_dir:
            continue
        print("processing " + aug_list_dir + " ... ...")
        imgs = get_imgs_list(os.path.join(aug_imgs_dir, aug_list_dir))

        for each_img in imgs:
            img_data = cv2.imread(each_img)
            img_name = os.path.basename(each_img)
            gt_path = os.path.join(aug_edges_dir, aug_list_dir, img_name)
            if not os.path.exists(gt_path) and gt_path.endswith(".jpg"):
                gt_path = gt_path[:-4] + ".png"
            if not os.path.exists(gt_path) and gt_path.endswith(".png"):
                gt_path = gt_path[:-4] + ".jpg"
            gt_data = np.array(cv2.imread(gt_path, 0), dtype=np.int32)

            tmp_img_path = os.path.join(aug_imgs_dir, aug_list_dir + "_Contrast")
            create_dir(tmp_img_path)
            tmp_edge_path = os.path.join(aug_edges_dir, aug_list_dir + "_Contrast")
            create_dir(tmp_edge_path)

            # --------------------------here is the main augmentation part-------------------------
            img_aug = float(1.3) * img_data
            img_aug[img_aug>255] =255
            img_aug = np.round(img_aug)
            img_aug = img_aug.astype(np.uint8)
            edge_aug = gt_data
            # -------------------------------------------------------------------------------------
            img_name = img_name[:-4] + ".png"
            cv2.imwrite(os.path.join(tmp_img_path, img_name), img_aug)
            cv2.imwrite(os.path.join(tmp_edge_path, img_name), edge_aug)

def main_aug(base_dir):
    imgs_dir = os.path.join(base_dir, "image")
    edges_dir = os.path.join(base_dir, "edge")
    print(imgs_dir)
    aug_imgs_dir = os.path.join(imgs_dir, "aug")
    aug_edges_dir = os.path.join(edges_dir, "aug")
    create_dir(aug_imgs_dir)
    create_dir(aug_edges_dir)

    print("==> Copy raw images and edges... ...")
    tmp_target = os.path.join(aug_imgs_dir, "real")
    if not os.path.exists(tmp_target):
        shutil.copytree(os.path.join(imgs_dir, "real"), tmp_target)
    tmp_target = os.path.join(aug_edges_dir, "real")
    if not os.path.exists(tmp_target):
        shutil.copytree(os.path.join(edges_dir, "real"), tmp_target)
    print("Done!")

    print("==> Start Crop Augmentaion")
    crop_aug_dir(aug_imgs_dir, aug_edges_dir)
    print("Crop Done!")

    print("==> Start Rotate Augmentaion")
    rotate_aug_dir(aug_imgs_dir, aug_edges_dir, rot_interval=120)
    print("Rotation Done!")

    # print("==> Start Flip Augmentaion")
    # flip_aug_dir(aug_imgs_dir, aug_edges_dir)
    # print("Flip Done!")

    print("==> Start CoarseDropout Augmentaion")
    CoarseDropout_aug_dir(aug_imgs_dir, aug_edges_dir)
    print("CoarseDropout Done!")

    print("==> Start GaussianBlur Augmentaion")
    GaussianBlur_aug_dir(aug_imgs_dir, aug_edges_dir)
    print("GaussianBlur Done!")

    print("==> Start sharpen Augmentaion")
    Sharpen_aug_dir(aug_imgs_dir, aug_edges_dir)
    print("sharpen Augmentaion!")

    print("==> Start Contrast_aug_dir Augmentaion")
    Contrast_aug_dir(aug_imgs_dir, aug_edges_dir)
    print("Contrast_aug_dir Augmentaion!")





base_dir = "E:/pytorch_model/Super_Edge_clean/datasets/linemod_data"

main_aug(base_dir)
