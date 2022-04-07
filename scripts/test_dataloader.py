from __future__ import print_function
import numpy as np
import os
import cv2
import torch
import time
from torch.utils.data import DataLoader
import sys
sys.path.append("../")
from datasets import CocoDataset, CocoDataset_no_aug, Dataset_multiscale, Dataset_two_scale, Dataset_with_blur, Dataset_with_l0smooth
from losses import *
from model import DexiNed
from utils import (image_normalization, visualize_result, get_imgs_list, create_pseudo_label_with_canny,
                   get_uncertainty, concatenate_images)


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


def get_uncertainty(preds):
    new_preds = []
    for each_pred in preds:
        if len(each_pred.shape) == 2:
            each_pred = np.expand_dims(each_pred, axis=0)
        new_preds.append(each_pred)
    cat = np.concatenate(new_preds, axis=0)     # cat.shape: (N, 720, 1280)
    std = cat.std(axis=0)                       # std.shape: (720, 1280)
    return std


def get_uncertainty_from_all_block(preds_list):
    tensor = torch.stack(preds_list, dim=0) # after stack: torch.Size([7, B, 1, 512, 512])
    tensor = tensor.transpose(0, 1)     # after transpose: torch.Size([B, 7, 1, 512, 512])
    confs = []
    for each_pred in tensor:
        # print(each_pred.shape)  # torch.Size([7, 1, 512, 512])
        all_block_preds = []
        for i in range(len(each_pred)):
            edge_map = torch.sigmoid(each_pred[i]).cpu().detach().numpy()
            edge_map = np.squeeze(edge_map)  # like (512, 512)
            edge_map = np.uint8(image_normalization(edge_map))
            edge_map = cv2.bitwise_not(edge_map)
            all_block_preds.append(edge_map)
            cv2.imwrite("./block_"+str(i+1)+".png", edge_map)

        std = get_uncertainty(all_block_preds)
        # hm = cv2.applyColorMap(std.astype(np.uint8), cv2.COLORMAP_JET)
        # cv2.imwrite("./all_block_uncertainty.png", hm)
        # print(std.shape, np.max(std), np.mean(std))
        confidence = np.exp(-std)
        # print(confidence.shape, np.max(confidence), np.mean(confidence))
        confs.append(torch.from_numpy(confidence))

    conf_tensors = torch.stack(confs, dim=0).to(device)
    conf_tensors = torch.unsqueeze(conf_tensors, dim=1)
    # print(conf_tensors.shape)       # torch.Size([4, 1, 512, 512])
    return conf_tensors


def get_confidence_for_all_blocks(preds_list, preds_list_s1):
    tensor = torch.stack(preds_list, dim=0)  # after stack: torch.Size([7, B, 1, H, W])
    tensor = tensor.transpose(0, 1)  # after transpose: torch.Size([B, 7, 1, H, W])

    tensor_s1 = torch.stack(preds_list_s1, dim=0)
    tensor_s1 = tensor_s1.transpose(0, 1)

    confs_tmp_imgs = []
    batch_index = 0
    for preds, preds_s1 in zip(tensor, tensor_s1):
        batch_index += 1
        # preds.shape: torch.Size([7, 1, 512, 512])
        # vis_results = []
        confs_tmp = []
        for i in range(len(preds)):
            edge_map = torch.sigmoid(preds[i]).cpu().detach().numpy()
            edge_map = np.squeeze(edge_map)  # like (512, 512)
            edge_map = np.uint8(image_normalization(edge_map))
            # edge_map = cv2.bitwise_not(edge_map)

            edge_map_s1 = torch.sigmoid(preds_s1[i]).cpu().detach().numpy()
            edge_map_s1 = np.squeeze(edge_map_s1)
            edge_map_s1 = np.uint8(image_normalization(edge_map_s1))
            edge_map_s1 = cv2.resize(edge_map_s1, (512, 512))

            _, s1_mask = cv2.threshold(edge_map_s1, 0, 255, cv2.THRESH_OTSU)
            s1_mask[s1_mask == 255] = 1
            intersect = s1_mask * edge_map

            # -----------------------------------key code-----------------------------------
            _, intersect_bi = cv2.threshold(intersect, 0, 255, cv2.THRESH_OTSU)
            alpha = 1  # for normal pixels, set loss weight as alpha, default=1;
            beta = 2  # for pixels that we want to emphasize, set weight as beta, default=2
            confidence = np.clip(intersect_bi, alpha, beta)  # confidence map for each block
            # confidence = intersect / 255      # which is better?
            # confidence.shape: (512, 512)
            # -----------------------------------key code-----------------------------------
            confs_tmp.append(torch.from_numpy(confidence))

            # print(np.max(confidence), np.min(confidence), np.mean(confidence))
            # vis_results.append(edge_map_s1)
            # vis_results.append(edge_map)
            # vis_results.append(intersect)
            # vis_results.append(intersect_bi)

        confs_for_img = torch.stack(confs_tmp, dim=0)  # confs_for_img.shape: torch.Size([7, H, W])

        confs_tmp_imgs.append(confs_for_img)
        # vis_img = concatenate_images(vis_results, 4, 7, 15, strategy="top2down")
        # cv2.imwrite("vis_all_" + str(batch_index) + ".png", vis_img)
    confs_for_batch_imgs = torch.stack(confs_tmp_imgs, dim=0).to(device)  # confs_for_batch_imgs.shape: torch.Size([B, 7, H, W])
    confs_for_batch_imgs = confs_for_batch_imgs.transpose(0, 1)  # after transpose: torch.Size([7, B, H, W])
    confs_for_batch_imgs = confs_for_batch_imgs.unsqueeze(dim=2)  # after transpose: torch.Size([7, B, 1, H, W])
    l_weight_confs = []
    for each_level in confs_for_batch_imgs:
        l_weight_confs.append(each_level)
    return l_weight_confs



def get_confidence_for_all_blocks_with_blur(preds_list, preds_list_blur):
    tensor = torch.stack(preds_list, dim=0)  # after stack: torch.Size([7, B, 1, H, W])
    tensor = tensor.transpose(0, 1)  # after transpose: torch.Size([B, 7, 1, H, W])
    tensor_blur = torch.stack(preds_list_blur, dim=0)
    tensor_blur = tensor_blur.transpose(0, 1)

    confs_tmp_imgs = []
    batch_index = 0
    for preds, preds_blur in zip(tensor, tensor_blur):
        batch_index += 1
        # preds.shape: torch.Size([7, 1, 512, 512])
        # vis_results = []
        confs_tmp = []
        for i in range(len(preds)):
            edge_map = torch.sigmoid(preds[i]).cpu().detach().numpy()
            edge_map = np.squeeze(edge_map)  # like (H, W)
            edge_map = np.uint8(image_normalization(edge_map))
            # edge_map = cv2.bitwise_not(edge_map)

            edge_map_blur = torch.sigmoid(preds_blur[i]).cpu().detach().numpy()
            edge_map_blur = np.squeeze(edge_map_blur)
            edge_map_blur = np.uint8(image_normalization(edge_map_blur))

            edge_map = np.expand_dims(edge_map, axis=0)  # (1, H, W)
            edge_map_blur = np.expand_dims(edge_map_blur, axis=0)

            raw_and_blur = [edge_map, edge_map_blur]
            cat = np.concatenate(raw_and_blur, axis=0)  # cat.shape: (N, H, W)
            std = cat.std(axis=0)  # (1, H, W)
            confidence = np.exp(-std)  # (1, H, W)
            confs_tmp.append(torch.from_numpy(confidence))

            # print(np.max(std), np.min(std), np.mean(std))
            # print(np.max(confidence), np.min(confidence), np.mean(confidence))
            # vis_results.append(np.squeeze(edge_map).astype(np.uint8))
            # vis_results.append(np.squeeze(edge_map_blur).astype(np.uint8))
            # vis_results.append(cv2.applyColorMap(np.squeeze(std).astype(np.uint8), cv2.COLORMAP_JET))
        confs_for_img = torch.stack(confs_tmp, dim=0)  # confs_for_img.shape: torch.Size([7, H, W])
        confs_tmp_imgs.append(confs_for_img)
        # vis_img = concatenate_images(vis_results, 3, 7, 15, strategy="top2down")
        # cv2.imwrite("vis_all_" + str(batch_index) + ".png", vis_img)

    confs_for_batch_imgs = torch.stack(confs_tmp_imgs, dim=0).to(
        device)  # confs_for_batch_imgs.shape: torch.Size([B, 7, H, W])
    confs_for_batch_imgs = confs_for_batch_imgs.transpose(0, 1)  # after transpose: torch.Size([7, B, H, W])
    confs_for_batch_imgs = confs_for_batch_imgs.unsqueeze(dim=2)  # after transpose: torch.Size([7, B, 1, H, W])
    l_weight_confs = []
    for each_level in confs_for_batch_imgs:
        l_weight_confs.append(each_level)
    return l_weight_confs


def get_confidence_from_avg_with_blur(preds_list, preds_list_blur):
    tensor = torch.stack(preds_list, dim=0)  # after stack: torch.Size([7, B, 1, H, W])
    tensor = tensor.transpose(0, 1)  # after transpose: torch.Size([B, 7, 1, H, W])
    tensor_blur = torch.stack(preds_list_blur, dim=0)
    tensor_blur = tensor_blur.transpose(0, 1)

    # ------forget to add sigmoid before calculate the uncertainty------
    tensor = torch.sigmoid(tensor)
    tensor_blur = torch.sigmoid(tensor_blur)
    # ------------------------------------------------------------------

    confs_imgs = []
    batch_index = 0
    for preds, preds_blur in zip(tensor, tensor_blur):
        batch_index += 1
        # preds.shape: torch.Size([7, 1, 512, 512])
        # vis_results = []
        all_block_preds = []
        all_block_preds_blur = []
        for i in range(len(preds)):
            edge_map = torch.sigmoid(preds[i]).cpu().detach().numpy()
            edge_map = np.squeeze(edge_map)  # like (H, W)
            edge_map = np.uint8(image_normalization(edge_map))
            all_block_preds.append(edge_map)

            edge_map_blur = torch.sigmoid(preds_blur[i]).cpu().detach().numpy()
            edge_map_blur = np.squeeze(edge_map_blur)
            edge_map_blur = np.uint8(image_normalization(edge_map_blur))
            all_block_preds_blur.append(edge_map_blur)

        average = np.array(all_block_preds, dtype=np.float32)
        average = np.uint8(np.mean(average, axis=0))
        average = np.expand_dims(average, axis=0)  # (1, H, W)

        average_blur = np.array(all_block_preds_blur, dtype=np.float32)
        average_blur = np.uint8(np.mean(average_blur, axis=0))
        average_blur = np.expand_dims(average_blur, axis=0)  # (1, H, W)

        concat = np.concatenate([average, average_blur], axis=0)  # cat.shape: (B, H, W)
        std = concat.std(axis=0)  # (H, W)
        confidence = np.exp(-std)  # (H, W)
        confs_imgs.append(torch.from_numpy(confidence))
        # print(np.max(std), np.min(std), np.mean(std))
        # print(np.max(confidence), np.min(confidence), np.mean(confidence))
        # vis_results.append(np.squeeze(average).astype(np.uint8))
        # vis_results.append(np.squeeze(average_blur).astype(np.uint8))
        # vis_results.append(cv2.applyColorMap(np.squeeze(std).astype(np.uint8), cv2.COLORMAP_JET))
        # vis_img = concatenate_images(vis_results, 1, 3, 15, strategy="left2right")
        # cv2.imwrite("vis_all_" + str(batch_index) + ".png", vis_img)

    confs_for_batch_imgs = torch.stack(confs_imgs, dim=0).to(device)  # torch.Size([B, 512, 512])
    confs_for_batch_imgs = confs_for_batch_imgs.unsqueeze(dim=1)  # torch.Size([B, 1, 512, 512])
    confs_for_batch_imgs = confs_for_batch_imgs.to(device)
    return confs_for_batch_imgs


def get_confidence_kl(preds, preds_blur):
    preds = torch.sigmoid(preds)
    preds_blur = torch.sigmoid(preds_blur)
    # variance = kl_distance(torch.log(preds), preds_blur)
    # variance = torch.exp(-variance)
    pred1 = torch.stack([preds, 1 - preds], dim=-1)
    pred2 = torch.stack([preds_blur, 1 - preds_blur], dim=-1)
    # print(pred1.shape, pred2.shape) # torch.Size([B, 1, H, W, 2])
    kl_distance = torch.nn.KLDivLoss(reduction="none")
    variance = torch.sum(kl_distance(torch.log(pred1), pred2), dim=-1)
    confidence = torch.exp(-variance)   # torch.Size([B, 1, H, W])
    return confidence


# ------------try to get the uncertainty map using the variance of multi layer(all blocks)-------------
device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')
model = DexiNed().to(device)
checkpoint_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/" \
                  "checkpoints/RENDER/10/10_model.pth"

# checkpoint_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/" \
#                   "checkpoints/RENDER/10/10_model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
imgs_path = "/home/speed_kai/yeyunfan_phd/COCO_datasets/COCO_COCO_2017_Val_images/val2017/image"
current_label_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/" \
                     "datasets/COCO/val2017_iterative_with_blur_consistency_mse0.001_batch4_size400/start/edges"
# dataset_train = Dataset_with_l0smooth(images_path=imgs_path, labels_path=current_label_path, img_width=512, img_height=512, mean_bgr=[103.939, 116.779, 123.68])

dataset_train = Dataset_with_blur(images_path=imgs_path, labels_path=current_label_path, img_width=400, img_height=400, mean_bgr=[103.939, 116.779, 123.68])
# Dataset_with_l0smooth, Dataset_with_blur

batch_size = 2
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 1.3]
for batch_id, sample_batched in enumerate(dataloader_train):

    images = sample_batched['images'].to(device)  # BxCxHxW
    labels = sample_batched['labels'].to(device)  # Bx1xHxW
    preds_list = model(images)          # len(preds_list): 7, torch.Size([B, 1, H, W])


    images_blur = sample_batched['images_blur'].to(device)
    preds_list_blur = model(images_blur)    # len(preds_list_s1): 7, torch.Size([B, 1, H, W])
    print(len(preds_list_blur))
    print(images.shape, labels.shape)

    # loss = 0
    sum_mse = 0
    sum_bce_conf = 0
    # sum_bce = 0
    # sum_bce_consistency = 0
    for preds, preds_blur, l_w in zip(preds_list, preds_list_blur, l_weight):
        # print(preds.shape) # torch.Size([B, 1, H, W])
        # bce_loss = CE_loss(preds, labels, l_w)
        # sum_bce += bce_loss
        confidence = get_confidence_kl(preds, preds_blur)
        # print(confidence)
        # print(confidence.shape)
        # print(torch.min(confidence), torch.max(confidence), torch.mean(confidence))
        bce_conf_loss = CE_loss_with_confidence(preds, labels, confidence, l_w)
        sum_bce_conf += bce_conf_loss

        mse_loss = MSE_loss(preds, preds_blur, l_w)
        sum_mse += mse_loss
        print("bce_conf_loss:", bce_conf_loss, "mse_loss:", mse_loss)
    # dice_loss = Dice_Loss_consistency(preds_list[6], preds_list_blur[6])
    # print("DICE loss for fuse map:", dice_loss)
    sum_mse = 0.001 * sum_mse
    # print("BCE loss:", sum_bce)
    print("BCE loss with confidence:", sum_bce_conf)
    print("MSE loss:", sum_mse)
    loss = sum_bce_conf + sum_mse
    print("total loss:", loss)


    # confidence = get_confidence_from_avg_with_blur(preds_list, preds_list_blur)
    # loss = 0
    # for preds, l_w in zip(preds_list, l_weight):
    #     loss += CE_loss_with_confidence(preds, labels, confidence, l_w)
    # print("BCE loss with blur:", loss)

    break


'''
# ------------try to get the uncertainty map using the variance of multi layer(all blocks)-------------
device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')
model = DexiNed().to(device)
checkpoint_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/checkpoints/RENDER/10/10_model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
imgs_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/datasets/BIPED/image"
current_label_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/datasets/BIPED/pseudo_label/start/edges"
dataset_train = CocoDataset_no_aug(images_path=imgs_path, labels_path=current_label_path,
                                   img_width=512, img_height=512, mean_bgr=[103.939, 116.779, 123.68])

dataloader_train = DataLoader(dataset_train, batch_size=3, shuffle=True, num_workers=8)
l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 1.3]
for batch_id, sample_batched in enumerate(dataloader_train):
    images = sample_batched['images'].to(device)  # BxCxHxW
    labels = sample_batched['labels'].to(device)  # Bx1xHxW
    preds_list = model(images)

    conf_tensors = get_uncertainty_from_all_block(preds_list)

    # loss = sum([CE_loss(preds, labels, l_w) for preds, l_w in zip(preds_list, l_weight)])  # bdcn_loss\
    loss = 0
    for preds, l_w in zip(preds_list, l_weight):
        # loss += CE_loss(preds, labels, l_w)
        loss += CE_loss_with_confidence(preds, labels, conf_tensors, l_w)
    print(loss)
'''



'''
# ------------try to input multi-scale images to get the uncertainty map-------------
device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')
model = DexiNed().to(device)
checkpoint_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/checkpoints/RENDER/10/10_model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

imgs_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/datasets/BIPED/image"
current_label_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/datasets/BIPED/pseudo_label/start/edges"
dataset_train = Dataset_multiscale(images_path=imgs_path, labels_path=current_label_path,
                                   img_width=512, img_height=512, mean_bgr=[103.939, 116.779, 123.68])
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=8)
for batch_id, sample_batched in enumerate(dataloader_train):
    images = sample_batched['images'].to(device)  # BxCxHxW
    labels = sample_batched['labels'].to(device)  # Bx1xHxW

    images_s1 = sample_batched['images_s1'].to(device)  # scale 0.5
    images_s2 = sample_batched['images_s2'].to(device)  # scale 1.5
    print(images.shape, images_s1.shape, images_s2.shape)

    preds_list = model(images)
    preds_list_s1 = model(images_s1)
    preds_list_s2 = model(images_s2)
    print(preds_list[6].shape, preds_list_s1[6].shape, preds_list_s2[6].shape)

    avg = get_avg_from_preds(preds_list, img_shape=(512, 512))
    avg_s1 = get_avg_from_preds(preds_list_s1, img_shape=(512, 512))
    avg_s2 = get_avg_from_preds(preds_list_s2, img_shape=(512, 512))

    print(avg.shape, avg_s1.shape, avg_s2.shape)
    cv2.imwrite("./avg.png", avg)
    cv2.imwrite("./avg_s1.png", avg_s1)
    cv2.imwrite("./avg_s2.png", avg_s2)


    multiscale_results = [avg, avg_s1, avg_s2]
    std = get_uncertainty(multiscale_results)
    confidence = np.exp(-std)  # 0 < confidence <= 1

    confidence = torch.from_numpy(confidence).unsqueeze(dim=0).unsqueeze(dim=0) # from (512, 512) to torch.Size([1, 1, 512, 512])
    print(confidence.shape)
    confidence = confidence.to(device)
    # hm = cv2.applyColorMap(std.astype(np.uint8), cv2.COLORMAP_JET)
    # cv2.imwrite(os.path.join("./avg_uncertainty.png"), hm)
    # print(np.max(confidence), np.mean(confidence), np.min(confidence))
    l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 1.3]
    # loss = sum([CE_loss(preds, labels, l_w) for preds, l_w in zip(preds_list, l_weight)])  # bdcn_loss\
    loss = 0
    for preds, l_w in zip(preds_list, l_weight):
        # loss += CE_loss(preds, labels, l_w)
        loss += CE_loss_with_confidence(preds, labels, confidence, l_w)

    print(loss)
    break
'''
