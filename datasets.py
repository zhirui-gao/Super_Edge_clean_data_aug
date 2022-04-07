import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import measure


def dataset_info(dataset_name):
    config = {
        'BIPED_pseudo': {
            'img_height': 512,  # 720 # 1088
            'img_width': 512,  # 1280 5 1920
            'test_list': None,
            'train_list': None,
            'data_dir': '/home/speed_kai/yeyunfan_phd/edge_detection/datasets/BIPED/BIPED_pseudo',
            'yita': 0.5
        },
        'RENDER': {
        'img_height': 512,  # 720 # 1088
        'img_width': 512,  # 1280 5 1920
        'test_list': None,
        'train_list': None,
        'data_dir': '/home/speed_kai/yeyunfan_phd/edge_detection/datasets/RENDER',
        'yita': 0.5
        },
        'BIPED_15%': {
            'img_height': 512,
            'img_width': 512,
            'test_list': None,
            'train_list': 'train_pair.lst',
            'data_dir': '/home/speed_kai/yeyunfan_phd/edge_detection/datasets/BIPED/BIPED_few_shot/BIPED_15%',
            # mean_rgb
            'yita': 0.5
        },
        'BIPED_2.5%': {
            'img_height': 512,
            'img_width': 512,
            'test_list': None,
            'train_list': 'train_pair.lst',
            'data_dir': '/home/speed_kai/yeyunfan_phd/edge_detection/datasets/BIPED/BIPED_few_shot/BIPED_2.5%',
            # mean_rgb
            'yita': 0.5
        },
        'BIPED_10%': {
            'img_height': 512,
            'img_width': 512,
            'test_list': None,
            'train_list': 'train_pair.lst',
            'data_dir': '/home/speed_kai/yeyunfan_phd/edge_detection/datasets/BIPED/BIPED_few_shot/BIPED_10%',
            # mean_rgb
            'yita': 0.5
        },
        'BIPED_5%': {
            'img_height': 512,
            'img_width': 512,
            'test_list': None,
            'train_list': 'train_pair.lst',
            'data_dir': '/home/speed_kai/yeyunfan_phd/edge_detection/datasets/BIPED/BIPED_few_shot/BIPED_5%',
            # mean_rgb
            'yita': 0.5
        },
        'COCO': {
            'img_height': 512,
            'img_width': 512,
            'test_list': None,
            'train_list': 'train_pair.lst',
            'data_dir': '/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge/data_aug/train_coco',  # mean_rgb
            'yita': 0.5
        },
        'CLASSIC': {
            'img_height': 512,
            'img_width': 512,
            'test_list': None,
            'data_dir': 'data',  # mean_rgb
            # 'data_dir': '/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge/scripts/split_image_test/raw_and_crop',
            # mean_rgb
            'yita': 0.5
        },

        'BIPED': {
            'img_height': 720,  # 720 # 1088
            'img_width': 1280,  # 1280 5 1920
            'test_list': 'test_rgb.lst',
            'train_list': 'train_rgb.lst',
            'data_dir': '/opt/dataset/BIPED/edges',  # mean_rgb
            'yita': 0.5
        },
        'BSDS': {
            'img_height': 512, #321
            'img_width': 512, #481
            'test_list': 'test_pair.lst',
            'data_dir': '/opt/dataset/BSDS',  # mean_rgb
            'yita': 0.5
        },
        'BSDS300': {
            'img_height': 512, #321
            'img_width': 512, #481
            'test_list': 'test_pair.lst',
            'data_dir': '/opt/dataset/BSDS300',  # NIR
            'yita': 0.5
        },
        'PASCAL': {
            'img_height': 375, # 375
            'img_width': 500, #500
            'test_list': 'test_pair.lst',
            'data_dir': '/opt/dataset/PASCAL',  # mean_rgb
            'yita': 0.3
        },
        'CID': {
            'img_height': 512,
            'img_width': 512,
            'test_list': 'test_pair.lst',
            'data_dir': '/opt/dataset/CID',  # mean_rgb
            'yita': 0.3
        },
        'NYUD': {
            'img_height': 425,
            'img_width': 560,
            'test_list': 'test_pair.lst',
            'data_dir': '/opt/dataset/NYUD',  # mean_rgb
            'yita': 0.5
        },
        'MULTICUE': {
            'img_height': 720,
            'img_width': 1280,
            'test_list': 'test_pair.lst',
            'data_dir': '/opt/dataset/MULTICUE',  # mean_rgb
            'yita': 0.3
        },
        'DCD': {
            'img_height': 240,
            'img_width': 360,
            'test_list': 'test_pair.lst',
            'data_dir': '/opt/dataset/DCD',  # mean_rgb
            'yita': 0.2
        }
    }
    return config[dataset_name]


def fit_img_postfix(img_path):
    if not os.path.exists(img_path) and img_path.endswith(".jpg"):
        img_path = img_path[:-4] + ".png"
    if not os.path.exists(img_path) and img_path.endswith(".png"):
        img_path = img_path[:-4] + ".jpg"
    return img_path


class CocoDataset(Dataset):
    def __init__(self,
                 data_root,
                 img_height,
                 img_width,
                 mean_bgr,
                 #  is_scaling=None,
                 # Whether to crop image or otherwise resize image to match image height and width.
                 crop_img=False
                 #  arg=None
                 ):
        self.data_root = data_root
        self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img

        self.data_index = self._build_index()

    def _build_index(self):
        data_root = os.path.abspath(self.data_root)
        images_path = os.path.join(data_root, 'image', "aug")
        labels_path = os.path.join(data_root, 'edge', "aug")

        sample_indices = []
        for directory_name in os.listdir(images_path):
            image_directories = os.path.join(images_path, directory_name)
            for file_name_ext in os.listdir(image_directories):
                file_name = os.path.splitext(file_name_ext)[0]
                sample_indices.append(
                    (os.path.join(images_path, directory_name, file_name + '.png'),
                     os.path.join(labels_path, directory_name, file_name + '.png'),)
                )
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]
        image_path = fit_img_postfix(image_path)
        label_path = fit_img_postfix(label_path)
        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image, label = self.transform(img=image, gt=label)

        return dict(images=image, labels=label)

    def transform(self, img, gt):
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        gt /= 255. # for DexiNed input and BDCN

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr

        crop_size = self.img_height if self.img_height == self.img_width else 400
        # print(crop_size)
        img = cv2.resize(img, dsize=(crop_size, crop_size))
        gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        gt[gt > 0.2] += 0.5
        gt = np.clip(gt, 0., 1.)
        # # For RCF input
        # # -----------------------------------
        # gt[gt==0]=0.
        # gt[np.logical_and(gt>0.,gt<0.5)] = 2.
        # gt[gt>=0.5]=1.
        #
        # gt = gt.astype('float32')
        # ----------------------------------

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()
        return img, gt


class CocoDataset_no_aug(Dataset):
    def __init__(self,
                 images_path,
                 labels_path,
                 img_height,
                 img_width,
                 mean_bgr,
                 crop_img=False
                 #  arg=None
                 ):
        self.images_path = images_path
        self.labels_path = labels_path
        # self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []
        for file_name_ext in os.listdir(self.images_path):
            file_name = os.path.splitext(file_name_ext)[0]
            sample_indices.append(
                (os.path.join(self.images_path, file_name + '.png'),
                 os.path.join(self.labels_path, file_name + '.png'),)
            )
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]
        image_path = fit_img_postfix(image_path)
        label_path = fit_img_postfix(label_path)
        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image, label = self.transform(img=image, gt=label)

        return dict(images=image, labels=label)

    def transform(self, img, gt):
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        gt /= 255. # for DexiNed input and BDCN

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr

        crop_size = self.img_height if self.img_height == self.img_width else 400
        # print(crop_size)
        img = cv2.resize(img, dsize=(crop_size, crop_size))
        gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        gt[gt > 0.2] += 0.5
        gt = np.clip(gt, 0., 1.)
        # # For RCF input
        # # -----------------------------------
        # gt[gt==0]=0.
        # gt[np.logical_and(gt>0.,gt<0.5)] = 2.
        # gt[gt>=0.5]=1.
        #
        # gt = gt.astype('float32')
        # ----------------------------------

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()
        return img, gt


class Dataset_multiscale(Dataset):
    def __init__(self,
                 images_path,
                 labels_path,
                 img_height,
                 img_width,
                 mean_bgr,
                 crop_img=False
                 #  arg=None
                 ):
        self.images_path = images_path
        self.labels_path = labels_path
        # self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []
        for file_name_ext in os.listdir(self.images_path):
            file_name = os.path.splitext(file_name_ext)[0]
            sample_indices.append(
                (os.path.join(self.images_path, file_name + '.png'),
                 os.path.join(self.labels_path, file_name + '.png'),)
            )
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]
        image_path = fit_img_postfix(image_path)
        label_path = fit_img_postfix(label_path)
        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image, label, image_s1, image_s2 = self.transform(img=image, gt=label)
        return dict(images=image, labels=label, images_s1=image_s1, images_s2=image_s2)

    def transform(self, img, gt):
        crop_size = self.img_height if self.img_height == self.img_width else 400
        # -----------------------------------------------------------
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        gt /= 255. # for DexiNed input and BDCN
        gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        gt[gt > 0.2] += 0.5
        gt = np.clip(gt, 0., 1.)
        gt = torch.from_numpy(np.array([gt])).float()
        # -----------------------------------------------------------
        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        # print(crop_size)
        img = cv2.resize(img, dsize=(crop_size, crop_size))
        img_s1 = cv2.resize(img, None, fx=0.5, fy=0.5)
        img_s2 = cv2.resize(img, None, fx=1.5, fy=1.5)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        img_s1 = img_s1.transpose((2, 0, 1))
        img_s1 = torch.from_numpy(img_s1.copy()).float()

        img_s2 = img_s2.transpose((2, 0, 1))
        img_s2 = torch.from_numpy(img_s2.copy()).float()

        return img, gt, img_s1, img_s2


class Dataset_two_scale(Dataset):
    def __init__(self,
                 images_path,
                 labels_path,
                 img_height,
                 img_width,
                 mean_bgr,
                 crop_img=False
                 #  arg=None
                 ):
        self.images_path = images_path
        self.labels_path = labels_path
        # self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []
        for file_name_ext in os.listdir(self.images_path):
            file_name = os.path.splitext(file_name_ext)[0]
            sample_indices.append(
                (os.path.join(self.images_path, file_name + '.png'),
                 os.path.join(self.labels_path, file_name + '.png'),)
            )
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]
        image_path = fit_img_postfix(image_path)
        label_path = fit_img_postfix(label_path)
        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image, label, image_s1 = self.transform(img=image, gt=label)
        return dict(images=image, labels=label, images_s1=image_s1)

    def transform(self, img, gt):
        crop_size = self.img_height if self.img_height == self.img_width else 400
        # -----------------------------------------------------------
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        gt /= 255. # for DexiNed input and BDCN
        gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        gt[gt > 0.2] += 0.5
        gt = np.clip(gt, 0., 1.)
        gt = torch.from_numpy(np.array([gt])).float()
        # -----------------------------------------------------------
        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        # print(crop_size)
        img = cv2.resize(img, dsize=(crop_size, crop_size))
        img_s1 = cv2.resize(img, None, fx=0.5, fy=0.5)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        img_s1 = img_s1.transpose((2, 0, 1))
        img_s1 = torch.from_numpy(img_s1.copy()).float()
        return img, gt, img_s1


class Dataset_with_blur(Dataset):
    def __init__(self,
                 images_path,
                 labels_path,
                 img_height,
                 img_width,
                 mean_bgr,
                 crop_img=False
                 #  arg=None
                 ):
        self.images_path = images_path
        self.labels_path = labels_path
        # self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []
        for file_name_ext in os.listdir(self.images_path):
            file_name = os.path.splitext(file_name_ext)[0]
            sample_indices.append(
                (os.path.join(self.images_path, file_name + '.png'),
                 os.path.join(self.labels_path, file_name + '.png'),)
            )
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]
        image_path = fit_img_postfix(image_path)
        label_path = fit_img_postfix(label_path)
        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image, label, image_blur = self.transform(img=image, gt=label)
        return dict(images=image, labels=label, images_blur=image_blur)

    def transform(self, img, gt):
        crop_size = self.img_height if self.img_height == self.img_width else 400
        # -----------------------------------------------------------
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        gt /= 255. # for DexiNed input and BDCN
        gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        gt[gt > 0.2] += 0.5
        gt = np.clip(gt, 0., 1.)
        gt = torch.from_numpy(np.array([gt])).float()

        # ------------------------blur-----------------------------------

        # # ----------------------blur strategy 1----------------------------
        # img_blur = cv2.GaussianBlur(img, (15, 15), 0)
        # img_blur = cv2.ximgproc.l0Smooth(img, kappa=2)

        # # ----------------------blur strategy 2(new)----------------------------
        # img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        # img_blur = cv2.bilateralFilter(img_blur, 15, 50, 50)

        # # ----------------------blur strategy 3(new)----------------------------
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        img_blur = cv2.bilateralFilter(img_blur, 15, 50, 50)
        img_blur = cv2.bilateralFilter(img_blur, 15, 50, 50)

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        # print(crop_size)
        img = cv2.resize(img, dsize=(crop_size, crop_size))

        img_blur = np.array(img_blur, dtype=np.float32)
        img_blur -= self.mean_bgr
        # print(crop_size)
        img_blur = cv2.resize(img_blur, dsize=(crop_size, crop_size))

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        img_blur = img_blur.transpose((2, 0, 1))
        img_blur = torch.from_numpy(img_blur.copy()).float()
        return img, gt, img_blur


class Dataset_with_blur_connectivity(Dataset):
    def __init__(self,
                 images_path,
                 labels_path,
                 img_height,
                 img_width,
                 mean_bgr,
                 crop_img=False
                 #  arg=None
                 ):
        self.images_path = images_path
        self.labels_path = labels_path
        # self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []
        for file_name_ext in os.listdir(self.images_path):
            file_name = os.path.splitext(file_name_ext)[0]
            sample_indices.append(
                (os.path.join(self.images_path, file_name + '.png'),
                 os.path.join(self.labels_path, file_name + '.png'),)
            )
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]
        image_path = fit_img_postfix(image_path)
        label_path = fit_img_postfix(label_path)
        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        area_num = self.get_connectivity_num(label)

        image, label, image_blur = self.transform(img=image, gt=label)
        return dict(images=image, labels=label, images_blur=image_blur, area_num=area_num)


    def get_connectivity_num(self, edge_map):
        temp_labels = measure.label(edge_map, connectivity=2)  # connectivity=2是8连通区域标记，等于1则是4连通
        props = measure.regionprops(temp_labels)
        return len(props)


    def transform(self, img, gt):
        crop_size = self.img_height if self.img_height == self.img_width else 400
        # -----------------------------------------------------------
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        gt /= 255. # for DexiNed input and BDCN
        gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        gt[gt > 0.2] += 0.5
        gt = np.clip(gt, 0., 1.)
        gt = torch.from_numpy(np.array([gt])).float()

        # ------------------------blur-----------------------------------

        # # ----------------------blur strategy 1----------------------------
        # img_blur = cv2.GaussianBlur(img, (15, 15), 0)
        # img_blur = cv2.ximgproc.l0Smooth(img, kappa=2)

        # # ----------------------blur strategy 2(new)----------------------------
        # img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        # img_blur = cv2.bilateralFilter(img_blur, 15, 50, 50)

        # # ----------------------blur strategy 3(new)----------------------------
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        img_blur = cv2.bilateralFilter(img_blur, 15, 50, 50)
        img_blur = cv2.bilateralFilter(img_blur, 15, 50, 50)

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        # print(crop_size)
        img = cv2.resize(img, dsize=(crop_size, crop_size))

        img_blur = np.array(img_blur, dtype=np.float32)
        img_blur -= self.mean_bgr
        # print(crop_size)
        img_blur = cv2.resize(img_blur, dsize=(crop_size, crop_size))

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        img_blur = img_blur.transpose((2, 0, 1))
        img_blur = torch.from_numpy(img_blur.copy()).float()
        return img, gt, img_blur


class Dataset_with_l0smooth(Dataset):
    def __init__(self,
                 images_path,
                 labels_path,
                 img_height,
                 img_width,
                 mean_bgr,
                 crop_img=False
                 #  arg=None
                 ):
        self.images_path = images_path
        self.labels_path = labels_path
        # self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []
        for file_name_ext in os.listdir(self.images_path):
            if not(file_name_ext.endswith(".jpg") or file_name_ext.endswith(".png")):
                continue
            file_name = os.path.splitext(file_name_ext)[0]
            sample_indices.append(
                (os.path.join(self.images_path, file_name + '.png'),
                 os.path.join(self.labels_path, file_name + '.png'),)
            )
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]
        image_path = fit_img_postfix(image_path)
        label_path = fit_img_postfix(label_path)

        img_name = os.path.basename(image_path)
        # imgs_blur_path = fit_img_postfix(os.path.join(self.images_path, "l0smooth", img_name))
        imgs_blur_path = fit_img_postfix(os.path.join(self.images_path + "_gaussian_l0smooth", img_name))
        # print(imgs_blur_path)
        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image_blur = cv2.imread(imgs_blur_path, cv2.IMREAD_COLOR)
        # print(image.shape, label.shape, image_blur.shape)

        image, label, image_blur = self.transform(img=image, gt=label, img_blur=image_blur)
        return dict(images=image, labels=label, images_blur=image_blur)

    def transform(self, img, gt, img_blur):
        crop_size = self.img_height if self.img_height == self.img_width else 400
        # -----------------------------------------------------------
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        gt /= 255.  # for DexiNed input and BDCN
        gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        gt[gt > 0.2] += 0.5
        gt = np.clip(gt, 0., 1.)
        gt = torch.from_numpy(np.array([gt])).float()
        # ------------------------blur-----------------------------------
        # img_blur = cv2.bilateralFilter(img, 15, 50, 50)
        # img_blur = cv2.bilateralFilter(img_blur, 15, 50, 50)
        # img_blur = cv2.GaussianBlur(img_blur, (5, 5), 0)
        # img_blur = cv2.ximgproc.l0Smooth(img, kappa=2)

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        # print(crop_size)
        img = cv2.resize(img, dsize=(crop_size, crop_size))

        img_blur = np.array(img_blur, dtype=np.float32)
        img_blur -= self.mean_bgr
        # print(crop_size)
        img_blur = cv2.resize(img_blur, dsize=(crop_size, crop_size))

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        img_blur = img_blur.transpose((2, 0, 1))
        img_blur = torch.from_numpy(img_blur.copy()).float()
        return img, gt, img_blur