#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: decode_spm.py
@time: 2019/9/9 下午4:32
@desc:
'''
import numpy as np
import math
from .utils import point_nms
from config.center_config import center_config

class SpmDecoder():
    def __init__(self, factor_x, factor_y, outw, outh):
        self.factor_x = factor_x
        self.factor_y = factor_y
        # self.level = [[0, 1, 2],
        #               [3, 4, 5],
        #               [6, 7, 8],
        #               [9, 10, 11],
        #               [13, 12]]
        self.level = [[0, 1, 3],  # COCO datasets
                      [0, 2, 4],
                      [5, 7, 9],
                      [6, 8, 10],
                      [11, 13, 15],
                      [12, 14, 16]]
        self.Z = math.sqrt(outw*outw + outh*outh)
        # self.Z = 1
        self.outw = outw
        self.outh = outh
        # print ('decoder self.z', self.Z)

    def __call__(self, spm_label, score_thres=0.9, dis_thres=5):

        center_map = spm_label[0]
        kps_map = spm_label[1]

        keep_coors = point_nms(center_map, score_thres, dis_thres)   # 要传入 h*w*c
        centers = keep_coors[0]       # 图片中所有中心点集合  (y, x, score)
        # print (len(centers))
        joints = []
        ret_centers = []
        for center in centers:        # 遍历每个人的中心点
            single_person_joints = [0 for i in range(center_config['joints']*2)]
            root_joint = [int(x) for x in center]    # 中心点作为根节点(y, x, score)
            ret_centers.append([center[1] * self.factor_x, center[0] * self.factor_y, center[2]])    # 中心点[x,y,score]的集合
            for single_path in self.level:
                # print (single_path)
                for i, index in enumerate(single_path):
                    # print (i, index)
                    if i == 0:
                        start_joint = [root_joint[1], root_joint[0]]    # (x,y)
                    if start_joint[0] >= kps_map.shape[1] or start_joint[1] >= kps_map.shape[0] \
                            or start_joint[0] < 0 or start_joint[1] < 0:                             # 超边界
                        break
                    offset = kps_map[start_joint[1], start_joint[0], 2*index:2*index+2] * self.Z     # 子joint的offset从父结点坐标处取值
                    # offset = kps_map[center[0], center[1], 2*index:2*index+2] * self.Z              # 子joint的offset从center点坐标处取值
                    # print (offset)
                    joint = [start_joint[0]+offset[0], start_joint[1]+offset[1]]    # (x,y)
                    # print ('start joint {} -> end joint {}'.format(start_joint, joint))
                    single_person_joints[2*index:2*index+2] = joint
                    start_joint = [int(x) for x in joint]           # (x,y)

            joints.append(single_person_joints)

        for single_person_joints in joints:
            for i in range(center_config['joints']):
                single_person_joints[2*i] *= self.factor_x
                single_person_joints[2*i+1] *= self.factor_y

        return joints, ret_centers