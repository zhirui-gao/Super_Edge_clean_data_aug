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
from utils import (image_normalization, merge_canny4pred, adapt_img_name, get_imgs_list, concatenate_images, get_txt_list, get_json_list)
import json

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

def get_edge(raw_img, roi_data, model, scale,device,output_dir,cam_id):
    img_width = scale
    img_height = scale
    output_dir_f = os.path.join(output_dir, "edge_result"+str(cam_id))
    if not os.path.exists(output_dir_f):
        os.mkdir(output_dir_f)
    with torch.no_grad():
        for each_roi in roi_data['roi']:
            part_name = each_roi['name']
            lx = int(max(0, each_roi['lx']))
            ly = int(max(0, each_roi['ly']))
            rx = int(min(raw_img.shape[1], each_roi['rx']))
            ry = int(min(raw_img.shape[0], each_roi['ry']))
            if rx<=lx or ry<=ly:
                continue


            roi_img = raw_img[ly:ry, lx:rx]
            image_shape = [roi_img.shape[0]*4, roi_img.shape[1]*4]
            image = pre_process(roi_img, device, img_width, img_height)
            preds = model(image)
            fused = get_block_from_preds(preds, img_shape=image_shape, block_num=7)  # block 7 is the fused edge map
            img_file_name_f = os.path.join(output_dir_f, part_name+".png")
            roi_img = cv2.resize(roi_img, (0, 0), fx=4, fy=4)
            n = roi_img.shape[0] / 16
            m = roi_img.shape[1] / 16
            n = int(n)
            m = int(m)
            roi_img = roi_img[0:16*n, 0:16*m]
            fused = fused[0:16*n, 0:16*m]
            assert (roi_img.shape[0]==fused.shape[0] and roi_img.shape[1]==fused.shape[1])
            cv.imwrite(img_file_name_f, roi_img)
            fused = 255-fused
            output_file_name_f = os.path.join(output_dir_f, part_name + "_edge.png")
            cv2.imwrite(output_file_name_f, fused)
            fused[fused <= 120] = 1
            fused[fused>120] = 0

            output_file_name_ans = os.path.join(output_dir_f, part_name + "_ans.png")
            cv2.imwrite(output_file_name_ans, fused)


def get_roi_edge(roi_img, part_name,  model, scale,device,output_dir,cam_id):
    img_width = scale
    img_height = scale
    output_dir_f = os.path.join(output_dir, "edge_result"+str(cam_id))
    if not os.path.exists(output_dir_f):
        os.mkdir(output_dir_f)
    kernel = np.ones((3, 3), np.uint8)
    with torch.no_grad():
        image_shape = [roi_img.shape[0]*4, roi_img.shape[1]*4]
        image = pre_process(roi_img, device, img_width, img_height)
        preds = model(image)
        fused = get_block_from_preds(preds, img_shape=image_shape, block_num=7)  # block 7 is the fused edge map
        img_file_name_f = os.path.join(output_dir_f, part_name+".png")
        roi_img = cv2.resize(roi_img, (0, 0), fx=4, fy=4)
        n = roi_img.shape[0] / 16
        m = roi_img.shape[1] / 16
        n = int(n)
        m = int(m)
        roi_img = roi_img[0:16*n, 0:16*m]
        fused = fused[0:16*n, 0:16*m]
        assert (roi_img.shape[0]==fused.shape[0] and roi_img.shape[1]==fused.shape[1])
        cv.imwrite(img_file_name_f, roi_img)
        fused = 255-fused
        fused = cv2.dilate(fused, kernel, iterations=1)
        output_file_name_f = os.path.join(output_dir_f, part_name + "_edge.png")
        cv2.imwrite(output_file_name_f, fused)
        fused[fused <= 120] = 0
        fused[fused>120] = 1
        output_file_name_ans = os.path.join(output_dir_f, part_name + "_ans.png")
        cv2.imwrite(output_file_name_ans, fused)

def get_fenshi_name(fenshi_path):
    fenshi_1_files = get_txt_list(fenshi_path)
    list_name =['none']*67
    for i in range(len(fenshi_1_files)):
        txt_file = fenshi_1_files[i]
        id = int(os.path.basename(txt_file[:-4]))
        with open(txt_file,'r') as f:
            data = f.read()
            names = data.split('\n')
        list_name[id] = names
    return list_name

def main(args):
    """Main function."""
    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")
    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')
    checkpoint_path = "E:/pytorch_model/Super_Edge_clean/checkpoints/minshi/20/20_model.pth"
    print("checkpoint_path:", checkpoint_path)
    # Get computing device
    model = DexiNed().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    scale = 512
    camera_imgs_dir = "D:/gzr/CadAlign_LXF_linemod_fusion-release/CadAlign/data/undistortImg"
    camera_rois_dir = "D:/gzr/CadAlign_LXF_linemod_fusion-release/CadAlign/data/undistortImg_json"
    camera_imgs = get_imgs_list(camera_imgs_dir)

    output_dir = os.path.join(os.path.dirname(checkpoint_path), "result" + str(scale))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fenshi_1 = "D:/gzr/CadAlign_LXF_linemod_fusion-release/CadAlign/config/change_roi"
    list_name_1 = get_fenshi_name(fenshi_1)
    fenshi_2 = "D:/gzr/CadAlign_LXF_linemod_fusion-release/CadAlign/config/change_roi_2"
    list_name_2 = get_fenshi_name(fenshi_2)
    fenshi_3 = "D:/gzr/CadAlign_LXF_linemod_fusion-release/CadAlign/config/change_roi_3"
    list_name_3 = get_fenshi_name(fenshi_3)

    camera_rois = get_json_list(camera_rois_dir)
    for i in tqdm(range(len(camera_rois))):
        # print(camera_rois[i])
        camera_roi = os.path.basename(camera_rois[i][:-5])
        roi_id = int(camera_roi[10:])
        # print(camera_roi)

        if roi_id < 40 or roi_id > 62:
            continue
        with open(camera_rois[i], 'r') as f:
            roi_data = json.load(f)
            img0_path = os.path.join(camera_imgs_dir, 'CCD'+str(roi_id)+'.bmp')
            img1_path = os.path.join(camera_imgs_dir, 'CCD'+str(roi_id+100)+'.bmp')
            img2_path = os.path.join(camera_imgs_dir, 'CCD'+str(roi_id+200)+'.bmp')
            img3_path = os.path.join(camera_imgs_dir, 'CCD'+str(roi_id+300)+'.bmp')

            img0 = cv2.imread(img0_path)
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            img3 = cv2.imread(img3_path)

            for each_roi in roi_data['roi']:
                part_name = each_roi['name']
                lx = int(max(0, each_roi['lx']))
                ly = int(max(0, each_roi['ly']))
                rx = int(min(img0.shape[1], each_roi['rx']))
                ry = int(min(img0.shape[0], each_roi['ry']))
                if rx <= lx or ry <= ly:
                    continue
                if part_name in list_name_3[roi_id]:
                    roi_img = img3[ly:ry, lx:rx]
                elif part_name in list_name_2[roi_id]:
                    roi_img = img2[ly:ry, lx:rx]
                elif part_name in list_name_1[roi_id]:
                    roi_img = img1[ly:ry, lx:rx]
                else:
                    roi_img = img0[ly:ry, lx:rx]
                get_roi_edge(roi_img, part_name, model, scale, device, output_dir, roi_id)
    # for i in tqdm(range(len(camera_imgs))):
    #     camera_name = os.path.basename(camera_imgs[i][:-4])
    #     print(camera_name)
    #     camera_id = int(camera_name[3:])
    #     if camera_id > 66:
    #         continue
    #     if camera_id < 40:
    #         continue
    #     #img = cv2.imread(camera_imgs[i])
    #     roi_path = os.path.join(camera_rois_dir, "camera-" + camera_name + '.json')
    #     with open(roi_path, 'r') as f:
    #         data = json.load(f)
    #         get_edge(camera_id, data, model, scale, device, output_dir, camera_id)

if __name__ == '__main__':
    args = parse_args()
    main(args)
