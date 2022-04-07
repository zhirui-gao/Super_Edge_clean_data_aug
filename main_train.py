from __future__ import print_function
import argparse
import os
import time, platform
from tqdm import tqdm
import cv2
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import CocoDataset, CocoDataset_no_aug
from losses import CE_loss
from model import DexiNed
from utils import (image_normalization, visualize_result, get_imgs_list, create_pseudo_label_with_canny, get_uncertainty)
from torch.utils.tensorboard import SummaryWriter # for torch 1.4 or greather

def train_one_epoch(epoch, dataloader, model, optimizer, device, log_interval_vis, tb_writer, args=None):
    imgs_res_folder = os.path.join(args.output_dir, args.train_data, 'current_res')
    os.makedirs(imgs_res_folder, exist_ok=True)

    # Put model in training mode
    model.train()
    # choice:[0.6,0.6,1.1,1.1,0.4,0.4,1.3] [0.4,0.4,1.1,1.1,0.6,0.6,1.3],
    # [0.4,0.4,1.1,1.1,0.8,0.8,1.3], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.1],
    # [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 1.3]
    l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 1.3]  # for bdcn loss theory 3 before the last 1.3 0.6-0..5

    for batch_id, sample_batched in enumerate(dataloader):
        images = sample_batched['images'].to(device)  # BxCxHxW
        labels = sample_batched['labels'].to(device)  # BxHxW

        preds_list = model(images)
        loss = sum([CE_loss(preds, labels, l_w) for preds, l_w in zip(preds_list, l_weight)]) # bdcn_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if tb_writer is not None:
            tb_writer.add_scalar('loss',
                                 loss.detach(),
                                 (len(dataloader) * epoch + batch_id))

        if batch_id % 20 == 0:
            print(time.ctime(), 'Epoch: {0} Sample {1}/{2} Loss: {3}'
                  .format(epoch, batch_id, len(dataloader), loss.item()))
        if batch_id % log_interval_vis == 0:
            res_data = []

            img = images.cpu().numpy()
            res_data.append(img[0])

            ed_gt = labels.cpu().numpy()
            res_data.append(ed_gt[0])
            for i in range(len(preds_list)):
                tmp = preds_list[i]
                tmp = tmp[0]
                # print(tmp.shape)
                tmp = torch.sigmoid(tmp).unsqueeze(dim=0)
                tmp = tmp.cpu().detach().numpy()
                res_data.append(tmp)

            vis_imgs = visualize_result(res_data, arg=args)
            del tmp, res_data

            vis_imgs = cv2.resize(vis_imgs, (int(vis_imgs.shape[1]*0.8), int(vis_imgs.shape[0]*0.8)))
            img_test = 'Epoch: {0} Sample {1}/{2} Loss: {3}'.format(epoch, batch_id, len(dataloader), loss.item())
            vis_imgs = cv2.putText(vis_imgs, img_test, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
            # imgs_res_folder_epoch = os.path.join(imgs_res_folder, 'epoch' + str(epoch))
            # if not os.path.exists(imgs_res_folder_epoch):
            #     os.mkdir(imgs_res_folder_epoch)
            cv2.imwrite(os.path.join(imgs_res_folder, "epoch_" + str(epoch) + '_batch_' + str(batch_id) + '.png'), vis_imgs)


def val_one_dir(val_dataset_path, model, output_dir_epoch, device):
    images_data = get_images_data(val_dataset_path)
    print("==>now testing on validation set... ...")
    img_test_dir = os.path.join(output_dir_epoch, "edges_pred")
    os.makedirs(img_test_dir, exist_ok=True)
    with torch.no_grad():
        total_duration = []
        for img_data in images_data:
            image = img_data['image_data'].to(device)
            file_name = img_data['file_name']
            image_shape = img_data['image_shape']
            start_time = time.time()
            preds = model(image)
            tmp_duration = time.time() - start_time
            total_duration.append(tmp_duration)
            fused = get_fuse_from_preds(preds, img_shape=image_shape)
            # output_dir_f = os.path.join(output_dir, "fused")
            output_dir_f = os.path.join(img_test_dir)
            if not os.path.exists(output_dir_f):
                os.mkdir(output_dir_f)
            output_file_name_f = os.path.join(output_dir_f, file_name)[:-4] + ".png"
            cv2.imwrite(output_file_name_f, fused)


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


def pre_process(img_data, device, width, height):
    image = cv2.resize(img_data, (width, height))
    image = np.array(image, dtype=np.float32)
    image -= [103.939, 116.779, 123.68]
    image = image.transpose((2, 0, 1))  # (512, 512, 3)  to (3, 512, 512)
    image = torch.from_numpy(image.copy()).float()  # numpy格式为（H,W,C), tensor格式是torch(C,H,W)
    image = image.unsqueeze(dim=0)  # torch.Size([1, 3, 512, 512])
    image = image.to(device)
    return image


def get_imgs_list(imgs_dir):
    return [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.pgm')]


def get_fuse_from_preds(tensor, img_shape=None):
    # 第7个element是前6个edge map的fusion
    fuse_map = torch.sigmoid(tensor[6]).cpu().detach().numpy()  # like (1, 1, 512, 512)
    fuse_map = np.squeeze(fuse_map)     # like (512, 512)
    img_shape = [img_shape[1], img_shape[0]]  # (H, W) -> (W, H)
    fuse_map = np.uint8(image_normalization(fuse_map))
    fuse_map = cv2.bitwise_not(fuse_map)
    # Resize prediction to match input image size
    if not fuse_map.shape[1] == img_shape[0] or not fuse_map.shape[0] == img_shape[1]:
        fuse_map = cv2.resize(fuse_map, (img_shape[0], img_shape[1]))
    return fuse_map.astype(np.uint8)


def get_images_data(val_dataset_path):
    img_width = 512
    img_height = 512
    # print(f"validation resize target size: {(img_height, img_width,)}")
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
        images_data.append(dict(image_data=img, file_name=file_name, image_shape=im_shape))
    return images_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DexiNed trainer.')
    # Training settings
    # train_dir = '/home/speed_kai/yeyunfan_phd/edge_detection/datasets/BIPED/BIPED_pseudo'
    # print("TRAIN_DATA:", TRAIN_DATA)
    # print("train_dir:", train_dir)
    # parser.add_argument('--input_dir', type=str, default=train_dir)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--iterative_train', type=bool, default=True)
    parser.add_argument('--log_interval_vis', type=int, default=40)
    # parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD', help='weight decay (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=2, metavar='B')
    parser.add_argument('--workers', default=8, type=int, help='The number of workers for the dataloaders.')
    parser.add_argument('--tensorboard', type=bool, default=True, help='Use Tensorboard for logging.')
    parser.add_argument('--channel_swap',default=[2, 1, 0], type=int)
    parser.add_argument('--img_width', type=int, default=512, help='Image width for training.')
    parser.add_argument('--img_height', type=int, default=512, help='Image height for training.')
    parser.add_argument('--mean_pixel_values', default=[103.939, 116.779, 123.68], type=float)
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""

    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")

    args.train_data = 'COCO_val2017_post_blur'
    # Tensorboard summary writer
    tb_writer = None
    training_dir = os.path.join(args.output_dir, args.train_data)
    os.makedirs(training_dir, exist_ok=True)

    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')
    model = DexiNed().to(device)
    args.iterative_train = True
    if args.iterative_train:
        checkpoint_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/checkpoints/RENDER/10/10_model.pth"
        # checkpoint_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/" \
        #                   "checkpoints/BIPED_10%_no_iterative/24/24_model.pth"
        if args.tensorboard:
            # from tensorboardX import SummaryWriter  # previous torch version

            tb_writer = SummaryWriter(log_dir=training_dir)

        print("=================================================================")
        print("This training process will use pre-trained model!")
        print("Pre-trained model path:", checkpoint_path)
        print("=================================================================")
        # ini_epoch=17
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        # dataset_train = CocoDataset(args.input_dir, img_width=args.img_width, img_height=args.img_height, mean_bgr=args.mean_pixel_values)

        # imgs_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/datasets/BIPED/image"
        # imgs_path = "/home/speed_kai/yeyunfan_phd/edge_detection/datasets/BIPED/BIPED_few_shot/BIPED_10%/no_label"
        imgs_path = "/home/speed_kai/yeyunfan_phd/COCO_datasets/COCO_COCO_2017_Val_images/val2017/image"
        pseudo_label_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/" \
                            "Super_Edge_clean/datasets/COCO/val2017_iterative_post_blur"
        if not os.path.exists(pseudo_label_path):
            os.mkdir(pseudo_label_path)

        # ------------------------------ initializing------------------------------
        print("creating initial pseudo model ... ...")
        output_dir_initial = os.path.join(pseudo_label_path, "start")
        if not os.path.exists(output_dir_initial):
            os.mkdir(output_dir_initial)
        output_dir_edges = os.path.join(output_dir_initial, "edges")
        if not os.path.exists(output_dir_edges):
            os.mkdir(output_dir_edges)
        output_dir_avg = os.path.join(output_dir_initial, "avg")
        if not os.path.exists(output_dir_avg):
            os.mkdir(output_dir_avg)

        imgs = get_imgs_list(imgs_path)
        test_img_width = 512
        test_img_height = 512

        # ---------------------------save time if already have pseudo label---------------------
        need_initial_pseudo_label = False
        if need_initial_pseudo_label:
            with torch.no_grad():
                for i in tqdm(range(len(imgs))):
                    file_name = os.path.basename(imgs[i])
                    img_data = cv2.imread(imgs[i], cv2.IMREAD_COLOR)
                    image_shape = [img_data.shape[0], img_data.shape[1]]
                    image = pre_process(img_data, device, test_img_width, test_img_height)
                    preds = model(image)

                    avg = get_avg_from_preds(preds, img_shape=image_shape)
                    output_file_name_avg = os.path.join(output_dir_avg, file_name)[:-4] + ".png"
                    cv2.imwrite(output_file_name_avg, avg)

                    merge = create_pseudo_label_with_canny(img_data, avg)
                    output_file_name_edge = os.path.join(output_dir_edges, file_name)[:-4] + ".png"
                    cv2.imwrite(output_file_name_edge, merge)
        # ------------------------------ initializing end ------------------------------

        max_round = 5
        epochs_of_one_round = 1
        for round in range(1, max_round + 1):
            # ----------------------stage 1: training model with the last pseudo label--------------------------
            print("start round " + str(round) + '... ...')
            if round == 1:
                current_label_path = output_dir_edges
            else:
                # here load the training data from last pseudo label
                current_label_path = os.path.join(pseudo_label_path, "after_round" + str(round - 1), "edges")

            dataset_train = CocoDataset_no_aug(images_path=imgs_path, labels_path=current_label_path,
                                               img_width=args.img_width, img_height=args.img_height,
                                               mean_bgr=args.mean_pixel_values)
            dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

            is_validation = True
            val_dataset_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/datasets/val_data"

            # ini_epoch = 1
            for epoch in range((round - 1) * epochs_of_one_round + 1, round * epochs_of_one_round + 1):
                print("processing epoch " + str(epoch) + '... ...')

                train_one_epoch(epoch, dataloader_train, model, optimizer, device, args.log_interval_vis, tb_writer, args=args)

            output_dir_epoch = os.path.join(args.output_dir, args.train_data, str(epoch))
            os.makedirs(output_dir_epoch, exist_ok=True)
            model_saved_path = os.path.join(output_dir_epoch, '{0}_model.pth'.format(epoch))
            torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(), model_saved_path)
            if is_validation:
                val_one_dir(val_dataset_path, model, output_dir_epoch, device)

            # ----------------------stage 2: create pseudo label using current model--------------------------
            # "c_o_r" means "current output"
            c_o_path = os.path.join(pseudo_label_path, "after_round" + str(round))
            if not os.path.exists(c_o_path):
                os.mkdir(c_o_path)
            with torch.no_grad():
                for i in tqdm(range(len(imgs))):
                    file_name = os.path.basename(imgs[i])
                    img_data = cv2.imread(imgs[i], cv2.IMREAD_COLOR)
                    image_shape = [img_data.shape[0], img_data.shape[1]]
                    image = pre_process(img_data, device, test_img_width, test_img_height)
                    preds = model(image)

                    avg = get_avg_from_preds(preds, img_shape=image_shape)
                    output_dir_avg = os.path.join(c_o_path, "avg")
                    if not os.path.exists(output_dir_avg):
                        os.mkdir(output_dir_avg)
                    output_file_name_avg = os.path.join(output_dir_avg, file_name)[:-4] + ".png"
                    cv2.imwrite(output_file_name_avg, avg)

                    merge = create_pseudo_label_with_canny(img_data, avg)
                    output_dir_edges = os.path.join(c_o_path, "edges")
                    if not os.path.exists(output_dir_edges):
                        os.mkdir(output_dir_edges)
                    output_file_name_edge = os.path.join(output_dir_edges, file_name)[:-4] + ".png"
                    cv2.imwrite(output_file_name_edge, merge)

if __name__ == '__main__':
    args = parse_args()
    main(args)
