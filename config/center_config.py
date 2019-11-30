#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: center_config.py
@time: 2019/7/24 下午7:17
@desc:
'''

center_config = {}

center_config['height'] = 320         # 网络输入大小
center_config['width'] = 576
center_config['scale'] = 4            # 网络输出的缩放系数
center_config['out_shape'] = (80, 144)     # (out_h, out_w)

center_config['joints'] = 17
center_config['objs'] = 1

center_config['lr'] = 1e-4
center_config['weight_decay'] = 1e-4

center_config['train_json_file'] = '/dataset_sdc_5/COCO/cpn_coco/data/COCO2014/annotations/COCO_2014_train.json'
center_config['train_img_path'] = '/dataset_sdc_5/COCO/cpn_coco/data/COCO2014/train2014'

center_config['val_json_file'] = '/dataset_sdc_5/COCO/cpn_coco/data/COCO2014/annotations/COCO_2014_val.json'
center_config['val_img_path'] = '/dataset_sdc_5/COCO/cpn_coco/data/COCO2014/val2014'

center_config['ori_gt_path'] = '/dataset_sdc_5/COCO/cpn_coco/data/COCO2014/annotations/person_keypoints_val2014.json'

# center_config['finetune'] = '/media/hsw/E/ckpt/spm_net/2019-09-12-14-19'
# center_config['finetune'] = '/home/hsw/server/ckpt-74/ckpt-79'
# center_config['ckpt'] = '/media/hsw/E/ckpt/spm_net'