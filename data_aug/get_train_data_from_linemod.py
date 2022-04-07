# 使用进过筛选（手动）linemod在各个工件上的匹配结果，作为边缘检测的数据
import os
import json
import tqdm
import cv2
import shutil
import numpy as np

def get_json_list(json_dir):
    return [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]


def get_txt_list(txt_dir):
    return [os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if f.endswith('.txt')]


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


def main():
    fenshi_1 = "D:/gzr/CadAlign_LXF_linemod_fusion-release/CadAlign/config/change_roi"
    list_name_1 = get_fenshi_name(fenshi_1)
    fenshi_2 = "D:/gzr/CadAlign_LXF_linemod_fusion-release/CadAlign/config/change_roi_2"
    list_name_2 = get_fenshi_name(fenshi_2)
    fenshi_3 = "D:/gzr/CadAlign_LXF_linemod_fusion-release/CadAlign/config/change_roi_3"
    list_name_3 = get_fenshi_name(fenshi_3)

    camera_rois_dir = "D:/gzr/CadAlign_LXF_linemod_fusion-release/CadAlign/data/undistortImg_json"
    camera_rois = get_json_list(camera_rois_dir)

    camera_img_dir = "D:/gzr/CadAlign_LXF_linemod_fusion-release/CadAlign/0524data/undistortImg"
    save_path_img = "E:/pytorch_model/Super_Edge_clean/datasets/linemod_data/img/real"
    save_path_edge = "E:/pytorch_model/Super_Edge_clean/datasets/linemod_data/edge/real"
    save_path_label = "E:/pytorch_model/Super_Edge_clean/datasets/linemod_data/label"

    featrue_label_path = "D:/gzr/CadAlign_LXF_linemod_fusion-release/CadAlign/test/visualize_calibra24"
    for i in range(len(camera_rois)):
        print(camera_rois[i])

        camera_roi = os.path.basename(camera_rois[i][:-5])
        roi_id = int(camera_roi[10:])
        print(camera_roi)

        if roi_id < 40 or roi_id > 62:
            continue
        with open(camera_rois[i], 'r') as f:
            roi_data = json.load(f)
            img0_path = os.path.join(camera_img_dir, 'CCD'+str(roi_id)+'.bmp')
            img1_path = os.path.join(camera_img_dir, 'CCD'+str(roi_id+100)+'.bmp')
            img2_path = os.path.join(camera_img_dir, 'CCD'+str(roi_id+200)+'.bmp')
            img3_path = os.path.join(camera_img_dir, 'CCD'+str(roi_id+300)+'.bmp')

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
                roi_img = cv2.resize(roi_img, (0, 0), fx=4, fy=4)
                n = roi_img.shape[0] / 16
                m = roi_img.shape[1] / 16
                n = int(n)
                m = int(m)
                roi_img = roi_img[0:16 * n, 0:16 * m]
                featrue_txt_path = os.path.join(featrue_label_path, "camera-CCD"+str(roi_id), part_name+".txt")
                if os.path.exists(featrue_txt_path):
                    cv2.imwrite(os.path.join(save_path_img, '24_'+"CCD" + str(roi_id) + '_' + part_name + '.png'), roi_img)
                    shutil.copyfile(featrue_txt_path, os.path.join(save_path_label,'24_'+"CCD" + str(roi_id) + '_' + part_name +'.txt'))
                    with open(featrue_txt_path,'r') as f:
                        data = f.readlines()
                        points = []
                        for line in data:
                            inf = line.split(" ")
                            points.append([float(inf[0]), float(inf[1])])
                        ellipse = cv2.fitEllipse(np.array(points, dtype=np.int32))
                        edge_img = np.zeros([roi_img.shape[0], roi_img.shape[1]], np.uint8)
                        axesSize = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
                        rotateAngle = ellipse[2]
                        startAngle = 0
                        endAngle = 360
                        point_color = (255, 255, 255)  # BGR
                        thickness = 1
                        lineType = 4
                        cv2.ellipse(edge_img, (int(ellipse[0][0]), int(ellipse[0][1])), axesSize, rotateAngle,
                                    startAngle, endAngle, point_color, thickness, lineType)
                    cv2.imwrite(os.path.join(save_path_edge,'24_'+"CCD" + str(roi_id) + '_' + part_name + '.png'), edge_img)

if __name__ == '__main__':
    main()