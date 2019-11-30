import os
import argparse
import json
import time

import numpy as np
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from dataset.SpmDataset import SpmDataset
from nets import network
from config.center_config import center_config
from dataset.decode_spm import SpmDecoder



os.environ['CUDA_VISIBLE_DEVICES'] = '4'

batch_size = 32
workers = 8
netH = center_config['height']
netW = center_config['width']
img_path = "/home/xsk/pycharmprojects/SPM_Pytorch/img/gymnastics_1.jpg"
checkpoint_file = "/dataset_sdc_5/xsk/SPM_person_model/SPM_resnet50_576_320_person_20191130_epoch5.pth.tar"
result_dir = "/home/xsk/pycharmprojects/SPM_Pytorch/result"
score = 0.5
dist = 20
colors = [[0,0,255],[255,0,0],[0,255,0],[0,255,255],[255,0,255],[255,255,0]]

def plot_landmark(image, single_result):
    # for i in range(int(len(single_result)/3)):
    #     # cv2.circle(image, (int(single_result[3*i]), int(single_result[3*i+1])), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
    #     cv2.putText(image, "{}".format(i), (int(single_result[3*i]), int(single_result[3*i+1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
    #             lineType=cv2.LINE_AA)

    # POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
    #               [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    # POSE_PAIRS = [[0, 1], [0, 2], [2, 4], [1, 3], [0, 5], [0, 6], [6, 8], [5, 7], [7, 9], [8, 10], [5, 11], [6, 12], [11, 12],
    #               [11, 13], [12, 14], [13, 15], [14, 16]]
    POSE_PAIRS = [[0, 1], [1, 2], [2, 4], [1, 3], [3, 5], [0, 6], [6, 8], [8, 10],[0, 7], [7, 9], [9, 11], [0, 12], [12, 14],
                  [14, 16], [0, 13], [13, 15], [15, 17]]
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        point_size = 8 if partA==0 else 4
        if single_result[2*partA]!=0 and single_result[2*partA+1]!=0 and single_result[2*partB]!=0 and single_result[2*partB+1]!=0:
            cv2.line(image, (int(single_result[2*partA]), int(single_result[2*partA+1])), (int(single_result[2*partB]), int(single_result[2*partB+1])), colors[j % 6], 2)
            cv2.circle(image, (int(single_result[2*partA]), int(single_result[2*partA+1])), point_size, colors[j % 6], thickness=-1, lineType=cv2.FILLED)
            cv2.circle(image, (int(single_result[2*partB]), int(single_result[2*partB+1])), 4, colors[j % 6], thickness=-1, lineType=cv2.FILLED)

    # cv2.imshow('landmark', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return

model = network.CPN50(center_config['out_shape'], center_config['joints']*2+1, pretrained = True)
model = torch.nn.DataParallel(model).cuda()

checkpoint = torch.load(checkpoint_file)
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

model.eval()

img = cv2.imread(img_path)
h, w, c = img.shape
resized = cv2.resize(img, (netW, netH), interpolation=cv2.INTER_LINEAR)
print(resized.shape)
norm = resized.astype(np.float32) / 255.
norm = np.transpose(norm, (2, 0, 1)) # C*H*W
norm = torch.from_numpy(norm).float()
input = torch.unsqueeze(norm, dim=0)

factor_x = w / (netW / 4)
factor_y = h / (netH / 4)

input_var = torch.autograd.Variable(input.cuda())
global_outputs, refine_output = model(input_var)
root_map_pred = torch.unsqueeze(refine_output[:, 0, :, :], dim=1)
kps_map_pred = refine_output[:, 1:, :, :]
center_map = root_map_pred.data.cpu().numpy()
center_map = np.transpose(center_map, (0, 2, 3, 1))     # n*h*w*c
kps_reg_map = kps_map_pred.data.cpu().numpy()
kps_reg_map = np.transpose(kps_reg_map, (0, 2, 3, 1))
print(center_map.shape)
print(kps_reg_map.shape)

spm_decoder = SpmDecoder(factor_x, factor_y, netH // 4, netW // 4)
joints, centers = spm_decoder([center_map[0], kps_reg_map[0]], score_thres=score, dis_thres=dist)

for j, single_person_joints in enumerate(joints):
    # print(centers[j][:2])
    # print(single_person_joints)
    plot_landmark(img, centers[j][:2]+single_person_joints)
cv2.imshow('landmark', img)
cv2.waitKey()
cv2.destroyAllWindows()

# for j, single_person_joints in enumerate(joints):
#     cv2.circle(img, (int(centers[j][0]), int(centers[j][1])), 8, colors[j % 6], thickness=-1)
#     for i in range(center_config['joints']):
#         x = int(single_person_joints[2 * i])
#         y = int(single_person_joints[2 * i + 1])
#         cv2.circle(img, (x, y), 4, colors[j % 6], thickness=-1)
#         # cv2.putText(img_show, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 250), 1)

# cv2.imshow('result', img)
# k = cv2.waitKey()


