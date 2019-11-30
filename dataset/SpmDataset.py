from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os
import random

from .utils import read_cpn_json, prepare_bbox, prepare_kps, im_to_torch
from .data_aug import data_aug
from .spm import SingleStageLabel

colors = [[0,0,255],[255,0,0],[0,255,0],[0,255,255],[255,0,255],[255,255,0]]

class SpmDataset(Dataset):
    def __init__(self, params, train=True):
        self.params = params
        self.train = train
        if train:
            json_file = params['train_json_file']
            self.img_path = params['train_img_path']
        else:
            json_file = params['val_json_file']
            self.img_path = params['val_img_path']

        self.img_ids, self.id_bboxs_dict, self.id_kps_dict = read_cpn_json(json_file)

        if train:
            random.shuffle(self.img_ids)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_path, img_id))
        img_h, img_w = img.shape[:2]

        bboxs = self.id_bboxs_dict[img_id]
        kps = self.id_kps_dict[img_id]

        ############ show ori label #############
        # img_ori = img.copy()
        # for box in bboxs:
        #     print(box)
        #     cv2.rectangle(img_ori, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 0, 255))
        #
        # for j, kp in enumerate(kps):
        #     print (j, kp)
        #     for i in range(17):
        #         x = int(kp[i*3])
        #         y = int(kp[i*3+1])
        #         v = kp[i*3+2]
        #         cv2.circle(img_ori, (x,y),4,colors[j%3],thickness=-1)
        # cv2.imshow('ori %s'%img_id, img_ori)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        ###########################################################

        if self.train:
            # data aug
            img, bboxs, kps = data_aug(img, bboxs, kps, self.params['joints'])

        ############ show aug label #############
        # img_ori = img.copy()
        # for box in bboxs:
        #     # print(box)
        #     cv2.rectangle(img_ori, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 0, 255))
        #
        # for j, kp in enumerate(kps):
        #     # print (j, kp)
        #     for i in range(17):
        #         x = int(kp[i*3])
        #         y = int(kp[i*3+1])
        #         v = kp[i*3+2]
        #         cv2.circle(img_ori, (x,y),4,colors[j%3],thickness=-1)
        # cv2.imshow('ori', img_ori)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        ###########################################################

        # print(img_id)
        # print(kps)

        orih, oriw, oric = img.shape
        neth, netw = self.params['height'], self.params['width']                    # 网络输入 高、宽
        outh, outw = neth // self.params['scale'], netw // self.params['scale']     # 网络输出 高、宽

        centers, sigmas, whs = prepare_bbox(bboxs, orih, oriw, outh, outw)  # 转换为输出featuremap大小后框的中心点、sigma、框的宽高
        keypoints, kps_sigmas = prepare_kps(kps, orih, oriw, outh, outw)  # 转换为输出featuremap大小后得关键点、sigma均为2.5,没用到

        # print(img_id)
        # print(centers)

        spm_label = SingleStageLabel(outh, outw, centers, sigmas, keypoints, self.params['joints'])
        center_map, kps_map, kps_map_weight = spm_label()
        center_map = np.transpose(center_map, (2, 0, 1))
        kps_map = np.transpose(kps_map, (2, 0, 1))
        kps_map_weight = np.transpose(kps_map_weight, (2, 0, 1))  # kps_map_weight用来指示哪些关节点存在，不存在的不计算loss

        ################# show center_map and kps_map ####################
        # for map in center_map:
        #     cv2.imshow("center map %s"%img_id, map)
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()
        # for map in kps_map:
        #     cv2.imshow("kps map %s"%img_id, map)
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()
        ##################################################################

        # create img input
        img = cv2.resize(img, (netw, neth), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow('img', img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        img = img.astype(np.float32) / 255.
        img = im_to_torch(img)

        if self.train:
            return img, torch.Tensor(center_map), torch.Tensor(kps_map), torch.Tensor(kps_map_weight)
        else:
            return img, torch.Tensor(center_map), torch.Tensor(kps_map), torch.Tensor(kps_map_weight), img_w, img_h, img_id
