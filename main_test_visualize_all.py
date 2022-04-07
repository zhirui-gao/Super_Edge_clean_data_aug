from __future__ import print_function
import argparse
import os
import time
from math import ceil
import cv2
import torch
from tqdm import tqdm
from losses import *
from model import DexiNed
from utils import (image_normalization, merge_canny4pred, adapt_img_name, get_imgs_list, concatenate_images)


def get_block_from_preds(tensor, img_shape=None, block_num=7):
    # 第7个element是前6个edge map的fusion
    fuse_map = torch.sigmoid(tensor[block_num - 1]).cpu().detach().numpy()  # like (1, 1, 512, 512)
    fuse_map = np.squeeze(fuse_map)     # like (512, 512)
    fuse_map = np.uint8(image_normalization(fuse_map))
    fuse_map = cv2.bitwise_not(fuse_map)
    # Resize prediction to match input image size
    # img_shape = [img_shape[1], img_shape[0]]  # (H, W) -> (W, H)
    # if not fuse_map.shape[1] == img_shape[0] or not fuse_map.shape[0] == img_shape[1]:
    #     fuse_map = cv2.resize(fuse_map, (img_shape[0], img_shape[1]))
    fuse_map = cv2.resize(fuse_map, (img_shape[1], img_shape[0]))
    return fuse_map.astype(np.uint8)


def get_avg_from_preds(tensor, img_shape=None):
    all_block_preds = []
    for i in range(len(tensor)):
        edge_map = torch.sigmoid(tensor[i]).cpu().detach().numpy()
        edge_map = np.squeeze(edge_map)  # like (512, 512)
        edge_map = np.uint8(image_normalization(edge_map))
        # edge_map = cv2.bitwise_not(edge_map)
        # Resize prediction to match input image size
        edge_map = cv2.resize(edge_map, (img_shape[1], img_shape[0]))
        all_block_preds.append(edge_map)
    average = np.array(all_block_preds, dtype=np.float32)
    average = np.uint8(np.mean(average, axis=0))
    average = cv2.bitwise_not(average)
    return average


def get_all_from_preds(tensor, raw_img, gt, img_shape=None, row=2):
    results = [raw_img]
    if gt is not None:
        gt = cv2.bitwise_not(gt)
        # results.append(cv2.applyColorMap(gt, cv2.COLORMAP_JET))
        results.append(gt)
    else:
        results.append(raw_img)

    all_block_preds = []
    for i in range(len(tensor)):
        edge_map = torch.sigmoid(tensor[i]).cpu().detach().numpy()
        edge_map = np.squeeze(edge_map)  # like (512, 512)
        edge_map = np.uint8(image_normalization(edge_map))
        edge_map = cv2.bitwise_not(edge_map)
        # Resize prediction to match input image size
        edge_map = cv2.resize(edge_map, (img_shape[1], img_shape[0]))
        all_block_preds.append(edge_map)

        if len(edge_map.shape) < 3:
            edge_map = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)

        edge_map = cv2.putText(edge_map, "block_"+str(i+1), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
        results.append(edge_map)

        # results.append(cv2.applyColorMap(edge_map, cv2.COLORMAP_JET))

    average = np.array(all_block_preds, dtype=np.float32)
    average = np.uint8(np.mean(average, axis=0))
    average = cv2.cvtColor(average, cv2.COLOR_GRAY2BGR)
    average = cv2.putText(average, "average", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA)

    results.append(average)
    # results.append(cv2.applyColorMap(average, cv2.COLORMAP_JET))

    row = row
    column = ceil(len(results) / row)
    all_preds = concatenate_images(results, row=row, column=column, interval=10, strategy="left2right")
    return all_preds


def get_images_data(val_dataset_path):
        img_width = 512
        img_height = 512
        # print(f"resize target size: {(img_height, img_width,)}")
        imgs = get_imgs_list(val_dataset_path)
        images_data = []
        for j, image_path in enumerate(imgs):
            file_name = os.path.basename(image_path)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            im_shape = [img.shape[0], img.shape[1]]
            img = cv2.resize(img, (img_width, img_height))
            img = np.array(img, dtype=np.float32)
            img -= [103.939, 116.779, 123.68]
            img = img.transpose((2, 0, 1))  # (512, 512, 3)  to (3, 512, 512)
            img = torch.from_numpy(img.copy()).float()  # numpy格式为（H,W,C), tensor格式是torch(C,H,W)
            img = img.unsqueeze(dim=0)  # torch.Size([1, 3, 512, 512])
            images_data.append(dict(image=img, file_name=file_name, image_shape=im_shape))
        return images_data


def pre_process(img_data, device, width, height):
    image = cv2.resize(img_data, (width, height))
    image = np.array(image, dtype=np.float32)
    image -= [103.939, 116.779, 123.68]
    image = image.transpose((2, 0, 1))  # (512, 512, 3)  to (3, 512, 512)
    image = torch.from_numpy(image.copy()).float()  # numpy格式为（H,W,C), tensor格式是torch(C,H,W)
    image = image.unsqueeze(dim=0)  # torch.Size([1, 3, 512, 512])
    image = image.to(device)
    return image


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DexiNed trainer.')
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""
    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")
    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')
    checkpoint_path = "./checkpoints/minshi/20/20_model.pth"
    print("checkpoint_path:", checkpoint_path)
    # Get computing device
    model = DexiNed().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))


    scale = 512
    test_imgs_path = "./datasets/test_data"


    output_dir = os.path.join(os.path.dirname(checkpoint_path), "results_scale" + str(scale))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print(f"output_dir: {output_dir}")

    img_width = scale
    img_height = scale
    print(f"resize target size: {(img_height, img_width,)}")
    imgs = get_imgs_list(test_imgs_path)
    # images_data = get_images_data(test_imgs_path)
    # Testing
    with torch.no_grad():
        for i in tqdm(range(len(imgs))):
            if i == 5000:
                break
            file_name = os.path.basename(imgs[i])
            img_data = cv2.imread(imgs[i], cv2.IMREAD_COLOR)
            # # blur strategy
            # img_data = cv2.GaussianBlur(img_data, (5, 5), 0)
            # for _ in range(2):
            #     img_data = cv2.bilateralFilter(img_data, 15, 50, 50)

            image_shape = [img_data.shape[0], img_data.shape[1]]
            # gt_data = cv2.imread(adapt_img_name(os.path.join(gt_path, file_name)), cv2.IMREAD_COLOR)

            image = pre_process(img_data, device, img_width, img_height)
            preds = model(image)

            fused = get_block_from_preds(preds, img_shape=image_shape, block_num=7)   # block 7 is the fused edge map
            output_dir_f = os.path.join(output_dir, "fused")
            if not os.path.exists(output_dir_f):
                os.mkdir(output_dir_f)
            output_file_name_f = os.path.join(output_dir_f, file_name)[:-4] + ".png"
            cv2.imwrite(output_file_name_f, fused)

            # block_num = 1
            # block = get_block_from_preds(preds, img_shape=image_shape, block_num=block_num)  # block 7 is the fused edge map
            # output_dir_b = os.path.join(output_dir, "block" + str(block_num))
            # if not os.path.exists(output_dir_b):
            #     os.mkdir(output_dir_b)
            # output_file_name_b = os.path.join(output_dir_b, file_name)[:-4] + ".png"
            # cv2.imwrite(output_file_name_b, block)

            avg = get_avg_from_preds(preds, img_shape=image_shape)
            output_dir_avg = os.path.join(output_dir, "avg")
            if not os.path.exists(output_dir_avg):
                os.mkdir(output_dir_avg)
            output_file_name_avg = os.path.join(output_dir_avg, file_name)[:-4] + ".png"
            cv2.imwrite(output_file_name_avg, avg)

            # all = get_all_from_preds(preds, raw_img=img_data, gt=gt_data, img_shape=image_shape)
            # output_dir_all = os.path.join(output_dir, "all")
            # if not os.path.exists(output_dir_all):
            #     os.mkdir(output_dir_all)
            # output_file_name_all = os.path.join(output_dir_all, file_name)[:-4] + ".png"
            # cv2.imwrite(output_file_name_all, all)
    return


if __name__ == '__main__':
    args = parse_args()
    main(args)
