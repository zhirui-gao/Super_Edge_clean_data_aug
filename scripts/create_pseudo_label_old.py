import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import sys
import time
print(os.path.dirname(__file__) + os.sep + '../')
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from model import DexiNed
from skimage import measure, morphology
import json
import shutil

def image_normalization(img, img_min=0, img_max=255):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)
    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255
    :return: a normalized image, if max is 255 the dtype is uint8
    """
    img = np.float32(img)
    epsilon = 1e-12  # whenever an inconsistent image
    img = (img-np.min(img))*(img_max-img_min) / \
        ((np.max(img)-np.min(img))+epsilon)+img_min
    return img


def get_fuse_from_preds(tensor, img_shape=None):
    # 第7个element是前6个edge map的fusion
    fuse_map = torch.sigmoid(tensor[6]).cpu().detach().numpy()  # like (1, 1, 512, 512)
    fuse_map = np.squeeze(fuse_map)  # like (512, 512)
    img_shape = [img_shape[1], img_shape[0]]  # (H, W) -> (W, H)
    fuse_map = np.uint8(image_normalization(fuse_map))
    # fuse_map = cv2.bitwise_not(fuse_map)
    # Resize prediction to match input image size
    if not fuse_map.shape[1] == img_shape[0] or not fuse_map.shape[0] == img_shape[1]:
        fuse_map = cv2.resize(fuse_map, (img_shape[0], img_shape[1]))
    return fuse_map.astype(np.uint8)


def get_imgs_list(imgs_dir):
    return [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if
            f.endswith('.jpg') or f.endswith('.png') or f.endswith('.pgm')]


def preprocess(img_data, target_shape, device):
    mean_bgr = [103.939, 116.779, 123.68]
    img_data = cv2.resize(img_data, target_shape)
    img_data = np.array(img_data, dtype=np.float32)
    img_data -= mean_bgr
    img_data = img_data.transpose((2, 0, 1))  # (512, 512, 3)  to (3, 512, 512)
    img_data = torch.from_numpy(img_data.copy()).float()  # numpy格式为（H,W,C), tensor格式是torch(C,H,W)
    img_data = img_data.unsqueeze(dim=0)  # torch.Size([1, 3, 512, 512])
    img_data = img_data.to(device)
    return img_data


def load_model(checkpoint_path, device):
    model = DexiNed().to(device)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Put model in evaluation mode
    return model


def test_split_concat(img_data, model, split_size, concat=False):
    (h, w, _) = img_data.shape
    fuse_concat_result = np.zeros((h, w), dtype=np.uint8)
    # print("raw image shape:", img_data.shape)
    for piece_size in range(split_size):
        if (not concat) and ((piece_size + 1) != split_size):
            continue
        row = (piece_size + 1)
        column = (piece_size + 1)
        # print("size of each piece:", (h / row, w / column))
        concat_result = np.zeros((h, w), dtype=np.uint8)

        with torch.no_grad():
            total_duration = []
            for r in range(row):
                for c in range(column):
                    # index = r * column + c
                    range_h1 = round(r * h / row)
                    range_h2 = round((r + 1) * h / row)
                    range_w1 = round(c * w / column)
                    range_w2 = round((c + 1) * w / column)
                    # overlap = 10
                    # range_h1 = max(0, range_h1 - overlap)
                    # range_h2 = min(h, range_h2 + overlap)
                    # range_w1 = max(0, range_w1 - overlap)
                    # range_w2 = min(w, range_w2 + overlap)
                    temp_data = img_data[range_h1: range_h2, range_w1: range_w2]

                    image_shape = [temp_data.shape[0], temp_data.shape[1]]
                    image = preprocess(temp_data, (512, 512), device)

                    start_time = time.time()
                    preds = model(image)
                    tmp_duration = time.time() - start_time
                    total_duration.append(tmp_duration)

                    fused = get_fuse_from_preds(preds, img_shape=image_shape)
                    # avg = get_avg_from_preds(preds, img_shape=image_shape)
                    concat_result[range_h1: range_h2, range_w1: range_w2] = fused
                    torch.cuda.empty_cache()

        fuse_concat_result = cv2.add(fuse_concat_result, concat_result)
    return fuse_concat_result


def concatenate_images(final_result, row, column, interval):
    (h, w, _) = final_result[0].shape  # 默认所有的图都是一个尺寸啊

    # 把数量不够的内容用黑图补上
    for i in range(row * column - len(final_result)):
        final_result.append(np.zeros((h, w, 3), dtype=np.uint8))

    # 先创建一个全黑大图，然后再一个个重新覆盖进去
    concat_result = np.zeros((row * h + interval * (row - 1), column * w + interval * (column - 1), 3), dtype=np.uint8)

    for r in range(row):
        for c in range(column):
            index = r * column + c
            # print(index)
            if len(final_result[index].shape) != 3:
                final_result[index] = cv2.cvtColor(final_result[index], cv2.COLOR_GRAY2BGR)
            range_h1 = r * h + r * interval
            range_h2 = (r + 1) * h + r * interval
            range_w1 = c * w + c * interval
            range_w2 = (c + 1) * w + c * interval
            concat_result[range_h1: range_h2, range_w1: range_w2] = final_result[index]
    return concat_result


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


def merge_dl_canny(pred_data, edge_canny, threshold, tolerance_size_min=3, tolerance_size_max=10):
    (h, w) = pred_data.shape
    binary_map = np.zeros((h, w), dtype=np.uint8)
    binary_map[pred_data <= threshold] = 0
    binary_map[pred_data > threshold] = 255

    assert tolerance_size_min % 2 == 1
    padding = int((tolerance_size_min - 1) / 2)
    binary_map = cv2.copyMakeBorder(binary_map, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)

    # padding2 = int((tolerance_size_max - 1) / 2)
    # binary_map2 = cv2.copyMakeBorder(binary_map, padding2, padding2, padding2, padding2, cv2.BORDER_CONSTANT, value=0)
    for x in range(h):
        for y in range(w):
            if edge_canny[x][y] == 255:
                kernel_tmp = binary_map[x:x + tolerance_size_min, y:y + tolerance_size_min]
                edge_pixel_num = np.sum(kernel_tmp == 255)
                if edge_pixel_num < 3:
                    edge_canny[x][y] = 0

                # kernel_tmp2 = binary_map2[x:x + tolerance_size_max, y:y + tolerance_size_max]
                # edge_pixel_num2 = np.sum(kernel_tmp2 == 255)
                # if edge_pixel_num2 > tolerance_size_max * tolerance_size_max * 0.8:
                #     edge_canny[x][y] = 0
    return edge_canny


def filter_canny_connectivity(canny_edge, min_thresh):
    labels = measure.label(canny_edge, connectivity=2)  # connectivity=2是8连通区域标记，等于1则是4连通
    props = measure.regionprops(labels)
    # print("total area:", len(props))
    # num = 0
    for each_area in props:
        if each_area.area <= min_thresh:
            # num += 1
            for each_point in each_area.coords:
                canny_edge[each_point[0]][each_point[1]] = 0
    return canny_edge


def merge_canny4pred(pred_data, threshold, img_data):
    # beta < 1则F更看中precision，beta > 1则更注重recall
    _, canny_range = search_best_range(pred_data, threshold, img_data, range_min=(10, 30), range_max=(40, 80), step=10, beta=2)
    # print("canny_range:", canny_range)
    edge_canny = cv2.Canny(img_data, canny_range[0], canny_range[1], apertureSize=3, L2gradient=False)
    merge_result = merge_dl_canny(pred_data, edge_canny, threshold=threshold, tolerance_size_min=3)
    return merge_result


def merge_canny4pred_filter_low_connect(pred_data, threshold, img_data):
    _, canny_range = search_best_range(pred_data, threshold, img_data, range_min=(10, 30), range_max=(40, 80), step=10, beta=2)
    # print("canny_range:", canny_range)
    edge_canny = cv2.Canny(img_data, canny_range[0], canny_range[1], apertureSize=3, L2gradient=False)
    merge_result = merge_dl_canny(pred_data, edge_canny, threshold=threshold, tolerance_size_min=3)
    merge_result = filter_canny_connectivity(merge_result, min_thresh=15)
    return merge_result


def get_edge_morphology(image):
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    dilate_img = cv2.dilate(image, kernel)
    erode_img = cv2.erode(image, kernel)
    absdiff_img = cv2.absdiff(dilate_img, erode_img);
    _, threshold_img = cv2.threshold(absdiff_img, 40, 255, cv2.THRESH_BINARY);
    # result = cv2.bitwise_not(threshold_img);
    return threshold_img


def get_connect_info(canny_edge):
    labels = measure.label(canny_edge, connectivity=2)  # connectivity=2是8连通区域标记，等于1则是4连通
    props = measure.regionprops(labels)
    # num = 0
    # thresh = 20
    # for each_area in props:
    #     # print("Eccentricity:", each_area.eccentricity)
    #     # print("centroid:", each_area.centroid)
    #     # print("total num:", each_area.area)
    #     # print("perimeter:", each_area.perimeter)
    #     # print("extent:", each_area.extent)
    #     if each_area.area <= thresh:
    #         num += 1
    return len(props)
    # print("total area:", len(props), "connectivity < " + str(thresh)+": " + str(num))


def get_area_nums(blur_image, split_size):
    pred = test_split_concat(blur_image, model, split_size)
    ret, pred_binary = cv2.threshold(pred, 0, 255, cv2.THRESH_OTSU)
    canny_merge = merge_canny4pred(pred_binary, ret, blur_image)
    labels = measure.label(canny_merge, connectivity=2)  # connectivity=2是8连通区域标记，等于1则是4连通
    props = measure.regionprops(labels)
    return len(props)


def multi_bilateralFilter(img, kernel_size, num):
    for _ in range(num):
        img = cv2.bilateralFilter(img, kernel_size, kernel_size * 2, kernel_size * 2)
    return img


if __name__ == '__main__':
    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")
    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')  # Get computing device

    # checkpoint_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge/checkpoints/RENDER/19/19_model.pth"
    checkpoint_path = "/home/speed_kai/yeyunfan_phd/edge_detection/codes/deep_learning_methods/Super_Edge_clean/checkpoints/RENDER/20/20_model.pth"
    model = load_model(checkpoint_path, device)

    image_dir = "/home/speed_kai/yeyunfan_phd/COCO_datasets/COCO_COCO_2017_Val_images/val2017/image"
    # image_dir = "/home/speed_kai/yeyunfan_phd/edge_detection/datasets/BIPED/edges/imgs/train/rgbr/real"
    imgs = get_imgs_list(image_dir)

    target_dir = os.path.join(image_dir + "_pseudo_label_strict", "edge")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    target_dir_img = os.path.join(image_dir + "_pseudo_label_strict", "image")
    if not os.path.exists(target_dir_img):
        os.makedirs(target_dir_img)

    k_size = 15
    max_iteration = 20
    step = 2   # 越大耗时越少

    total_info = []
    start = time.clock()
    for i in tqdm(range(len(imgs))):
        # if i < 40000:
        #     continue
        each_info = {}
        img_name = os.path.basename(imgs[i])
        each_info["img_name"] = img_name
        img = cv2.imread(imgs[i])
        each_info["height"] = img.shape[0]
        each_info["width"] = img.shape[1]
        img_area_num = get_area_nums(img, 1)
        each_info["image_area_num"] = img_area_num
        base_value = int(img.shape[0] * img.shape[1] / 100)
        # -------------------------------------以下是过滤条件，太复杂的图片没法玩的不要了-------------------------------------

        # 第一个条件，如果原图预测连通域数量大于正比于面积的某值，直接不要了
        if img_area_num > base_value * 0.5:
            continue
        # 第二个条件，如果中值滤波后的图像预测得到的连通域数量，与原图的比值，小于0.1，并且原图的数量要大于某最低阈值，那这个就不要了。
        # 这主要是为了滤掉那些有大量的草地，水面等类似的超高频且频繁出现的内容的图片，但又不至于滤掉只是比值小，但是图片本身其实并不复杂的图
        median = cv2.medianBlur(img, k_size)
        median_area_num = get_area_nums(median, 1)
        each_info["median_split_1_area_num"] = median_area_num
        if median_area_num * 0.1 > img_area_num > base_value * 0.2:
            continue

        # -------------------------------------以下是得到最终伪标签的策略-------------------------------------
        median2_area_num = get_area_nums(median, split_size=2)
        each_info["median_split_2_area_num"] = median2_area_num
        # print("median2_area_num:", median2_area_num)

        # bilateral = cv2.bilateralFilter(img, k_size, k_size * 2, k_size * 2)
        bilateral_blur_area_nums = []
        bilateral_blur_merges = []
        for blur_num in range(1, max_iteration, step):
            bilateral = multi_bilateralFilter(img, k_size, blur_num)
            bilateral_pred = test_split_concat(bilateral, model, split_size=2, concat=False)
            ret, bilateral_binary = cv2.threshold(bilateral_pred, 0, 255, cv2.THRESH_OTSU)
            bilateral_merge = merge_canny4pred(bilateral_binary, ret, bilateral)
            bilateral_blur_merges.append(bilateral_merge)
            labels = measure.label(bilateral_merge, connectivity=2)
            props = measure.regionprops(labels)
            bilateral2_area_num = len(props)
            bilateral_blur_area_nums.append(bilateral2_area_num)

            if bilateral2_area_num - median2_area_num < 50:
                break

        min_index = bilateral_blur_area_nums.index(min(bilateral_blur_area_nums))
        min_bilateral_merge = bilateral_blur_merges[min_index]

        each_info["bilateral_blur_area_nums"] = bilateral_blur_area_nums
        total_info.append(each_info)
        final_result = filter_canny_connectivity(min_bilateral_merge, min_thresh=15)

        cv2.imwrite(os.path.join(target_dir, img_name), final_result)
        shutil.copy(imgs[i], os.path.join(target_dir_img, img_name))
        # shutil.copy(imgs[i], os.path.join(target_dir_img, img_name)[:-4] + ".png")


    print("totol time:", time.clock() - start)
    json_path = os.path.join(image_dir, "_total_info_strict.json")
    with open(json_path, 'w') as f:
        json.dump(total_info, f)
    print("json file saved in " + json_path)



