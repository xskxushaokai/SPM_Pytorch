import os
import sys
sys.path.insert(0, './cocoapi/PythonAPI')
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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


os.environ['CUDA_VISIBLE_DEVICES'] = '4'

batch_size = 1
workers = 8
netH = center_config['height']
netW = center_config['width']
checkpoint_file = "/dataset_sdc_5/xsk/SPM_person_model/SPM_resnet50_576_320_person_20191127_epoch33.pth.tar"
result_dir = "/home/xsk/pycharmprojects/SPM_Pytorch/result"
score = 0.5
dist = 20
colors = [[0,0,255],[255,0,0],[0,255,0],[0,255,255],[255,0,255],[255,255,0]]

model = network.CPN50(center_config['out_shape'], center_config['joints']*2+1, pretrained = True)
model = torch.nn.DataParallel(model).cuda()

checkpoint = torch.load(checkpoint_file)
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

model.eval()

val_loader = torch.utils.data.DataLoader(
        SpmDataset(center_config, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )

predictions = []
for step, (inputs, root_map, kps_map, kps_weight, widths, heights, img_ids) in enumerate(val_loader):

    # test label
    root_map = root_map.numpy()
    root_map = np.transpose(root_map, (0, 2, 3, 1))
    kps_map = kps_map.numpy()
    kps_map = np.transpose(kps_map, (0, 2, 3, 1))

    s = time.time()
    input_var = torch.autograd.Variable(inputs.cuda())
    global_outputs, refine_output = model(input_var)
    root_map_pred = torch.unsqueeze(refine_output[:, 0, :, :], dim=1)
    kps_map_pred = refine_output[:, 1:, :, :]
    center_map = root_map_pred.data.cpu().numpy()
    center_map = np.transpose(center_map, (0, 2, 3, 1))     # n*h*w*c
    kps_reg_map = kps_map_pred.data.cpu().numpy()
    kps_reg_map = np.transpose(kps_reg_map, (0, 2, 3, 1))
    for b in range(len(img_ids)):
        factor_x = widths[b].numpy() / (netW / 4)
        factor_y = heights[b].numpy() / (netH / 4)
        spm_decoder = SpmDecoder(factor_x, factor_y, netH // 4, netW // 4)
        joints, centers = spm_decoder([center_map[b], kps_reg_map[b]], score_thres=score, dis_thres=dist)
        img_id = str(img_ids[b])

        # test label
        img = cv2.imread(os.path.join('/dataset_sdc_5/COCO/cpn_coco/data/COCO2014/val2014', img_id))
        joints_label, center_label = spm_decoder([root_map[b], kps_map[b]], score_thres=score, dis_thres=dist)
        for j, single_person_joints in enumerate(joints_label):
            cv2.circle(img, (int(center_label[j][0]), int(center_label[j][1])), 8, colors[j % 6], thickness=-1)
            for i in range(17):
                x = int(single_person_joints[2 * i])
                y = int(single_person_joints[2 * i + 1])
                cv2.circle(img, (x, y), 4, colors[j % 6], thickness=-1)
                cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 250), 1)
        cv2.imshow('result', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # predict = {}
        # predict['image_id'] = img_id
        kps = {}
        bbox = {}
        human = 1
        for j, single_person_joints in enumerate(joints):

            predict = {}
            predict['image_id'] = int(img_id.split('.')[0].split('_')[-1])
            predict['category_id'] = 1

            joints = []
            for i in range(center_config['joints']):
                x = int(single_person_joints[2 * i])
                y = int(single_person_joints[2 * i + 1])
                v = 1
                joints += [x, y, v]
            kps['human' + str(human)] = joints
            # bbox['human' + str(human)] = None
            human += 1

            predict['keypoints'] = joints
            predict['score'] = float(centers[j][2])
            predictions.append(predict)

        # predict['keypoint_annotations'] = kps
        # predictions.append(predict)
    e = time.time()
    print("processing.... {} / {}, time cost == {}".format(step, 30000 // batch_size, e - s))

print(len(predictions))
with open(os.path.join(result_dir, checkpoint_file.split('/')[-1].split('.')[0] + '_predicts.json'), 'w') as fw:
    json.dump(predictions, fw)
    print('done')

# evaluate on COCO
eval_gt = COCO(center_config['ori_gt_path'])
eval_dt = eval_gt.loadRes(os.path.join(result_dir, checkpoint_file.split('/')[-1].split('.')[0] + '_predicts.json'))
cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
