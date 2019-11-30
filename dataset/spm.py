#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: spm.py
@time: 2019/9/9 下午2:54
@desc:
'''
import numpy as np
import math
from .utils import draw_gaussian, clip, draw_ttfnet_gaussian

class SingleStageLabel():
    def __init__(self, height, width, centers, sigmas, kps, kps_num):   # 输出feature map的宽、高、目标中心、关键点...
        self.centers = centers
        self.sigmas = sigmas
        self.kps = kps
        self.height = height
        self.width = width
        self.Z = math.sqrt(height*height+width*width)        # 论文里说是图片的宽高平方根
        # self.Z = 1          # tensorflow实现里用了1

        # print ('encoder: self.Z', self.Z)

        self.center_map = np.zeros(shape=(height, width, 1), dtype=np.float32)       # opencv 的排列方式，h*w*c
        self.kps_map = np.zeros(shape=(height, width, kps_num*2), dtype=np.float32)       # 14个关键点，坐标需要28个map
        self.kps_count = np.zeros(shape=(height, width, kps_num*2), dtype=np.uint)
        self.kps_map_weight = np.zeros(shape=(height, width, kps_num*2), dtype=np.float32) # x, y 各对应一个weight？感觉有14个map表示weight就够了；这样做为了算loss的时候方便，直接与kps_map对应相乘就好了

        # hierarchical SPR, 0->1->2, 3->4->5, 6->7->8, 9->10->11, 13->12
        # self.level = [[0, 1, 2],
        #               [3, 4, 5],
        #               [6, 7, 8],
        #               [9, 10, 11],
        #               [13, 12]]
        self.level = [[0, 1, 3],      # COCO datasets
                      [0, 2, 4],
                      [5, 7, 9],
                      [6, 8, 10],
                      [11, 13, 15],
                      [12, 14, 16]]

        # self.body_level = {
        #     14:[0, 3, 6, 9, 13],
        #     0:[1],
        #     1:[2],
        #     3:[4],
        #     4:[5],
        #     6:[7],
        #     7:[8],
        #     9:[10],
        #     10:[11],
        #     13:[12]
        # }
        # # root joints -> [0, 3, 6, 9, 13]
        # self.root_joints_reg_map = np.zeros(shape=(height, width, 5*2), dtype=np.float32)
        # # joints [0->1, 1->2, 3->4, 4->5, 6->7, 7->8, 9->10, 10->11, 13->12]
        # self.body_jonts_reg_map = np.zeros(shape=(9, height, width, 2), dtype=np.float32)
        #
        # # 前5个是root joint对于[0, 3, 6, 9, 13]的offset，后面依次是：[0->1, 1->2, 3->4, 4->5, 6->7, 7->8, 9->10, 10->11, 13->12]
        # self.reg_map = np.zeros(shape=(height, width, 14, 2), dtype=np.float32)
        #
        # # 对于root joint，只生成位于 0 3 6 9 13的offset
        # self.reg_map = np.zeros(shape=(height, width, 14*2), dtype=np.float32)


    def __call__(self):

        for i, center in enumerate(self.centers):         # 遍历一张图中每个人
            sigma = self.sigmas[i]
            # sigma = [1, 1]
            kps = self.kps[i]
            if center[0] == 0 and center[1] == 0:
                continue
            # self.center_map[..., 0] = draw_gaussian(self.center_map[...,0], center, sigma, mask=None)
            self.center_map[..., 0] = draw_ttfnet_gaussian(self.center_map[...,0], center, sigma[0], sigma[1])    # center map 将当前人的中心点用高斯核处理，sigma大小为 框的宽高/9 + 0.8
            self.body_joint_displacement(center, kps, sigma)       # 根据关节点和中心点相对距离得到当前人的kps map

        # print (np.where(self.kps_count > 2))
        self.kps_count[self.kps_count == 0] += 1
        self.kps_map = np.divide(self.kps_map, self.kps_count)        # 逐像素除法，kps_map上每个点对该点处的人数取平均

        # return np.concatenate([self.center_map, self.kps_map, self.kps_map_weight], axis=-1)
        return self.center_map, self.kps_map, self.kps_map_weight     # 返回图片的center map, kps map, kps weight map； kps_map值是直接的坐标差，预测也直接预测坐标差，可能学起来有点问题

    def body_joint_displacement(self, center, kps, sigma):
        # taux = sigma[0]
        # tauy = sigma[1]
        taux = 2
        tauy = 2
        # taux = 1
        # tauy = 1

        for single_path in self.level:              # 计算每个关节相对于前一级关节点的offset    遍历每一条通道
            # print ('encoder single path: ', single_path)
            for i, index in enumerate(single_path):               # 遍历通道中每一级关节点
                # print ('i {} : index {}'.format(i, index))
                if i == 0:                         # 第一级关节点
                    start_joint = center           # 起点是目标中心点
                end_joint = kps[3*index:3*index+3]               # 关节点坐标,kps中关节点存放方式[x,y,signal]
                if start_joint[0] == 0 or start_joint[1] == 0:      # 关节点的参考点不存在
                    continue
                if end_joint[0] == 0 or end_joint[1] == 0:          # 关节点不存在
                    continue
                # if i == 0:       # 所有joint点均回归在center点的位置
                #     self.create_dense_displacement_map_l1(index, start_joint, end_joint, taux, tauy)
                # else:
                #     self.create_dense_displacement_map(index, center, start_joint, end_joint, taux, tauy)
                self.create_dense_displacement_map_l1(index, start_joint, end_joint, taux, tauy)                 # 子joint点回归在父节点的位置
                start_joint = end_joint

    def create_dense_displacement_map(self, index, center, start_joint, end_joint, sigmax=2, sigmay=2):

        # print('start joint {} -> end joint {}'.format(start_joint, end_joint))
        # center_x, center_y = int(start_joint[0]), int(start_joint[1])        # 这个实现中，关节点map上的取值中心采用的父节点的坐标，看论文里的意思应该始终是用了root点的坐标
        center_x, center_y = int(center[0]), int(center[1])
        th = 4.6052
        delta = np.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigmax + 0.5))                    # (x0,y0), (x1,y1)为参考点周围点的最外层点坐标   当前代码 joint map的中心点半径是13
        y0 = int(max(0, center_y - delta * sigmay + 0.5))

        x1 = int(min(self.width, center_x + delta * sigmax + 0.5))
        y1 = int(min(self.height, center_y + delta * sigmay + 0.5))

        # x0 = int(clip(start_joint[0], 0, self.width))
        # y0 = int(clip(start_joint[1], 0, self.height))
        # x1 = int(clip(x0+taux, 0, self.width))
        # y1 = int(clip(y0+tauy, 0, self.height))
        # print (x0,x1, y0,y1)
        for x in range(x0, x1):
            for y in range(y0, y1):                                          # 遍历(x0,y0), (x1,y1)内所有点进行赋值
                # x_offset = (end_joint[0] - x) / self.Z     # 关节点坐标与参考点周围点(x,y)的距离
                # y_offset = (end_joint[1] - y) / self.Z
                x_offset = (end_joint[0] - start_joint[0]) / self.Z                       # 关节点坐标与参考点的距离
                y_offset = (end_joint[1] - start_joint[1]) / self.Z
                # print (x_offset, y_offset)
                self.kps_map[y, x, 2*index] += x_offset                      # kps_map上(x,y)点值在原值基础上加上当前人的当前关节距离值
                self.kps_map[y, x, 2*index+1] += y_offset
                self.kps_map_weight[y, x, 2*index:2*index+2] = 1.            # kps_map_weight 上的取值为标志位，应该是在inference时作为一个人该关键点是否存在的判断阈值
                if end_joint[0] != x or end_joint[1] != y:                   # center周围点（x,y）不和关节点重合，则人数增加一个
                    self.kps_count[y, x, 2*index:2*index+2] += 1             # kps_count 上(x,y)的取值为参考点辐射到该点的人数


    def create_dense_displacement_map_l1(self, index, start_joint, end_joint, sigmax=2, sigmay=2):   # 第一级的joint map 中心点周围点的值取joint点到该周围点的实际距离

        # print('start joint {} -> end joint {}'.format(start_joint, end_joint))
        center_x, center_y = int(start_joint[0]), int(start_joint[1])        # 这个实现中，关节点map上的取值中心采用的父节点的坐标，看论文里的意思应该始终是用了root点的坐标
        th = 4.6052
        delta = np.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigmax + 0.5))                    # (x0,y0), (x1,y1)为参考点周围点的最外层点坐标   当前代码 joint map的中心点半径是13
        y0 = int(max(0, center_y - delta * sigmay + 0.5))

        x1 = int(min(self.width, center_x + delta * sigmax + 0.5))
        y1 = int(min(self.height, center_y + delta * sigmay + 0.5))

        # x0 = int(clip(start_joint[0], 0, self.width))
        # y0 = int(clip(start_joint[1], 0, self.height))
        # x1 = int(clip(x0+taux, 0, self.width))
        # y1 = int(clip(y0+tauy, 0, self.height))
        # print (x0,x1, y0,y1)
        for x in range(x0, x1):
            for y in range(y0, y1):                                          # 遍历(x0,y0), (x1,y1)内所有点
                x_offset = (end_joint[0] - x) / self.Z                       # 关节点坐标与参考点周围点(x,y)的距离
                y_offset = (end_joint[1] - y) / self.Z
                # print (x_offset, y_offset)
                self.kps_map[y, x, 2*index] += x_offset                      # kps_map上(x,y)点值在原值基础上加上当前人的当前关节距离值
                self.kps_map[y, x, 2*index+1] += y_offset
                self.kps_map_weight[y, x, 2*index:2*index+2] = 1.            # kps_map_weight 上的取值为标志位，应该是在inference时作为一个人该关键点是否存在的判断阈值
                if end_joint[0] != x or end_joint[1] != y:                   # center周围点（x,y）不和关节点重合，则人数增加一个
                    self.kps_count[y, x, 2*index:2*index+2] += 1             # kps_count 上(x,y)的取值为参考点辐射到该点的人数