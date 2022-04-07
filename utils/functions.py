import numpy as np
import cv2
import os
import torch
from .image import image_normalization


device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')


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
    # batch_index = 0
    for preds, preds_blur in zip(tensor, tensor_blur):
        # batch_index += 1
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
        # average = np.uint8(np.mean(average, axis=0))
        average = np.mean(average, axis=0)
        average = np.expand_dims(average, axis=0)  # (1, H, W)

        average_blur = np.array(all_block_preds_blur, dtype=np.float32)
        # average_blur = np.uint8(np.mean(average_blur, axis=0))
        average_blur = np.mean(average_blur, axis=0)
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


def get_confidence_for_all_blocks(preds_list, preds_list_s1, alpha, beta):
    tensor = torch.stack(preds_list, dim=0)  # after stack: torch.Size([7, B, 1, H, W])
    tensor = tensor.transpose(0, 1)  # after transpose: torch.Size([B, 7, 1, H, W])

    tensor_s1 = torch.stack(preds_list_s1, dim=0)
    tensor_s1 = tensor_s1.transpose(0, 1)

    confs_tmp_imgs = []
    batch_index = 0
    for preds, preds_s1 in zip(tensor, tensor_s1):
        batch_index += 1
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
            # alpha = 1  # normal pixels weight
            # beta = 2  # confident pixels weight
            confidence = np.clip(intersect_bi, alpha, beta)  # confidence map for each block
            # confidence = intersect / 255      # which is better?
            # -----------------------------------key code-----------------------------------
            confs_tmp.append(torch.from_numpy(confidence))
            # print(np.max(confidence), np.min(confidence), np.mean(confidence))

        confs_for_img = torch.stack(confs_tmp, dim=0)  # confs_for_img.shape: torch.Size([7, H, W])
        confs_tmp_imgs.append(confs_for_img)
    confs_for_batch_imgs = torch.stack(confs_tmp_imgs, dim=0).to(device)  # confs_for_batch_imgs.shape: torch.Size([B, 7, H, W])
    confs_for_batch_imgs = confs_for_batch_imgs.transpose(0, 1)  # after transpose: torch.Size([7, B, H, W])
    confs_for_batch_imgs = confs_for_batch_imgs.unsqueeze(dim=2)  # after transpose: torch.Size([7, B, 1, H, W])
    l_weight_confs = []
    for each_level in confs_for_batch_imgs:
        l_weight_confs.append(each_level)
    return l_weight_confs


def get_uncertainty(preds):
    new_preds = []
    for each_pred in preds:
        if len(each_pred.shape) == 2:
            each_pred = np.expand_dims(each_pred, axis=0)
        new_preds.append(each_pred)
    cat = np.concatenate(new_preds, axis=0)     # cat.shape: (3, 720, 1280)
    std = cat.std(axis=0)                       # std.shape: (720, 1280)
    return std


def get_uncertainty_from_all_block(preds_list):
    tensor = torch.stack(preds_list, dim=0) # after stack: torch.Size([7, B, 1, 512, 512])
    tensor = tensor.transpose(0, 1)     # after transpose: torch.Size([B, 7, 1, 512, 512])
    confs = []
    for each_pred in tensor:
        # print(each_pred.shape)  # torch.Size([7, 1, 512, 512])
        blocks_preds = []
        for i in range(len(each_pred)):
            edge_map = torch.sigmoid(each_pred[i]).cpu().detach().numpy()
            edge_map = np.squeeze(edge_map)  # like (512, 512)
            edge_map = np.uint8(image_normalization(edge_map))
            edge_map = cv2.bitwise_not(edge_map)
            blocks_preds.append(edge_map)

        std = get_uncertainty(blocks_preds)
        # hm = cv2.applyColorMap(std.astype(np.uint8), cv2.COLORMAP_JET)
        # cv2.imwrite("./all_block_uncertainty.png", hm)
        # print(std.shape, np.max(std), np.mean(std))
        confidence = np.exp(-std)
        # print(confidence.shape, np.max(confidence), np.mean(confidence))
        confs.append(torch.from_numpy(confidence))

    conf_tensors = torch.stack(confs, dim=0).to(device)
    conf_tensors = torch.unsqueeze(conf_tensors, dim=1)
    # print(conf_tensors.shape)       # torch.Size([B, 1, 512, 512])
    return conf_tensors


def adapt_img_name(img_path):
    if (not os.path.exists(img_path)) and img_path.endswith(".jpg"):
        img_path = img_path[:-4] + ".png"
    if (not os.path.exists(img_path)) and img_path.endswith(".png"):
        img_path = img_path[:-4] + ".jpg"
    return img_path


def get_imgs_list(imgs_dir):
    return [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.pgm') or f.endswith('.bmp')]

def get_json_list(json_dir):
    return [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]


def get_txt_list(txt_dir):
    return [os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if f.endswith('.txt')]


def concatenate_images(final_result, row, column, interval, strategy="left2right"):
    (h, w) = final_result[0].shape[:2]  # 默认所有的图都是一个尺寸啊

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


def fit_img_postfix(img_path):
    if not os.path.exists(img_path) and img_path.endswith(".jpg"):
        img_path = img_path[:-4] + ".png"
    if not os.path.exists(img_path) and img_path.endswith(".png"):
        img_path = img_path[:-4] + ".jpg"
    return img_path
