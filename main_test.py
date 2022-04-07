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
from utils import (image_normalization, save_image_batch_to_disk,
                   visualize_result)


def search_best_range(pred_data, threshold, img_blur, range_min, range_max, step, beta):
    # threshold = 10
    (h, w) = pred_data.shape
    binary_map = np.zeros((h, w), dtype=np.uint8)
    binary_map[pred_data <= threshold] = 1
    binary_map[pred_data > threshold] = 255

    binary_map_num = np.sum(binary_map == 255)
    # pred_data[pred_data <= threshold] = 1
    # pred_data[pred_data > threshold] = 255
    # pred_data_num = np.sum(pred_data == 255)

    F_list = []
    T_list = []
    for t_min in range(range_min[0], range_min[1], step):
        for t_max in range(range_max[0], range_max[1], step):
            if t_max <= t_min:
                continue
            # edge_canny = cv2.Canny(img_blur, 30, 90, apertureSize=3, L2gradient=False)
            edge_canny = cv2.Canny(img_blur, t_min, t_max, apertureSize=3, L2gradient=False)
            canny_edge_num = np.sum(edge_canny == 255)
            # print(pred_data.shape, edge_canny.shape)

            overlap_num = len(np.where(binary_map == edge_canny)[0])

            # recall and precision for canny, here we need a high recall
            recall = overlap_num / binary_map_num
            precision = overlap_num / canny_edge_num
            # beta < 1则F更看中precision，beta > 1则更注重recall
            bias = 0.01   # 防止分母为0
            F = ((beta * beta + 1) * (2 * precision * recall) + bias) / (beta * beta * precision + recall + bias)

            F_list.append(F)
            T_list.append((t_min, t_max))
            # print([t_min, t_max])
            # print("overlap_num:", overlap_num, ", pred_data_num:", pred_data_num, ", canny_edge_num:", canny_edge_num)
            # print("recall:", recall, "precision:", precision, "F-score:", F)
            # print("-----------------------------------------------------------------")

    zipped = zip(F_list, T_list)  # 进行封装，把要排序的分值放在前面
    zipped = sorted(zipped, reverse=True)  # 进行逆序排列
    sorted_f, sorted_t = zip(*zipped)  # 进行解压，已经按照分值排好
    return sorted_f[0], sorted_t[0]


def get_imgs_list(imgs_dir):
    return [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.pgm')]


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


def get_fuse_from_preds(tensor, img_shape=None):
    # 第7个element是前6个edge map的fusion
    fuse_map = torch.sigmoid(tensor[6]).cpu().detach().numpy()  # like (1, 1, 512, 512)
    fuse_map = np.squeeze(fuse_map)     # like (512, 512)
    fuse_map = np.uint8(image_normalization(fuse_map))
    fuse_map = cv2.bitwise_not(fuse_map)
    # Resize prediction to match input image size
    # img_shape = [img_shape[1], img_shape[0]]  # (H, W) -> (W, H)
    # if not fuse_map.shape[1] == img_shape[0] or not fuse_map.shape[0] == img_shape[1]:
    #     fuse_map = cv2.resize(fuse_map, (img_shape[0], img_shape[1]))
    fuse_map = cv2.resize(fuse_map, (img_shape[1], img_shape[0]))
    return fuse_map.astype(np.uint8)


def get_all_from_preds(tensor, raw_img, img_shape=None, row=4):
    canny_nms = False
    results = [raw_img]

    for i in range(len(tensor)):
        edge_map = torch.sigmoid(tensor[i]).cpu().detach().numpy()
        edge_map = np.squeeze(edge_map)  # like (512, 512)
        edge_map = np.uint8(image_normalization(edge_map))
        # edge_map = cv2.bitwise_not(edge_map)
        # Resize prediction to match input image size
        edge_map = cv2.resize(edge_map, (img_shape[1], img_shape[0]))
        if len(edge_map.shape) < 3:
            edge_map = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
        results.append(edge_map)

        if canny_nms:
            edge_map = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)

            _, canny_range = search_best_range(edge_map, 10, raw_img, range_min=(10, 30), range_max=(40, 80), step=10, beta=2)
            # print(canny_range)
            canny = cv2.Canny(raw_img, canny_range[0], canny_range[1], apertureSize=3, L2gradient=False)
            # canny = cv2.Canny(raw_img, 10, 20)
            edge_canny = edge_map * canny
            edge_canny = cv2.cvtColor(edge_canny, cv2.COLOR_GRAY2BGR)
            results.append(edge_canny)

    row = row
    column = ceil(len(results) / row)
    all_preds = concatenate_images(results, row=row, column=column, interval=10, strategy="left2right")
    return all_preds


def get_images_data(val_dataset_path):
        img_width = 512
        img_height = 512
        print(f"resize target size: {(img_height, img_width,)}")
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

    checkpoint_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/checkpoints/RENDER/10/10_model.pth"
    print("checkpoint_path:", checkpoint_path)
    # Get computing device
    model = DexiNed().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # base_dir = "/home/speed_kai/yeyunfan_phd/COCO_datasets/COCO_COCO_2017_Val_images/val2017"
    # test_imgs_path = os.path.join(base_dir, "image")
    # output_dir = os.path.join(base_dir, "results_render_epoch10")

    test_imgs_path = "/home/speed_kai/yeyunfan_phd/edge_detection/datasets/BIPED/edges/imgs/test/rgbr"
    output_dir = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/results/BIPED/render_10"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print(f"output_dir: {output_dir}")

    img_width = 512
    img_height = 512
    print(f"resize target size: {(img_height, img_width,)}")
    imgs = get_imgs_list(test_imgs_path)
    # images_data = get_images_data(test_imgs_path)
    # Testing
    with torch.no_grad():
        for i in tqdm(range(len(imgs))):
            if i == 200:
                break
            file_name = os.path.basename(imgs[i])
            img_data = cv2.imread(imgs[i], cv2.IMREAD_COLOR)
            image_shape = [img_data.shape[0], img_data.shape[1]]

            image = cv2.resize(img_data, (img_width, img_height))
            image = np.array(image, dtype=np.float32)
            image -= [103.939, 116.779, 123.68]
            image = image.transpose((2, 0, 1))  # (512, 512, 3)  to (3, 512, 512)
            image = torch.from_numpy(image.copy()).float()  # numpy格式为（H,W,C), tensor格式是torch(C,H,W)
            image = image.unsqueeze(dim=0)  # torch.Size([1, 3, 512, 512])
            image = image.to(device)
            preds = model(image)


            # fused = get_fuse_from_preds(preds, img_shape=image_shape)
            # output_dir_f = os.path.join(output_dir, "fused")
            # if not os.path.exists(output_dir_f):
            #     os.mkdir(output_dir_f)
            # output_file_name_f = os.path.join(output_dir_f, file_name)[:-4] + ".png"
            # cv2.imwrite(output_file_name_f, fused)

            all = get_all_from_preds(preds, raw_img=img_data, img_shape=image_shape, row=4)
            output_dir_all = os.path.join(output_dir, "all")
            if not os.path.exists(output_dir_all):
                os.mkdir(output_dir_all)
            output_file_name_all = os.path.join(output_dir_all, file_name)[:-4] + ".png"
            cv2.imwrite(output_file_name_all, all)
    return


if __name__ == '__main__':
    args = parse_args()
    main(args)
