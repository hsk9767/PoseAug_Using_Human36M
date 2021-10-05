import os
import os.path as osp
from pycocotools.coco import COCO
from PIL import Image

import numpy as np
import cv2
import random
import json
from utils.data_utils import world2cam, cam2pixel, pixel2cam, process_bbox, cam2pixel_custom


class MOBIS_DATASET:
    def __init__(self, data_split, args):
        print(f'==> {data_split} dataset of MOBIS DATASET is being loaded..')
        self.data_split = data_split
        self.ann_path = args.mobis_dataset_ann_path
        
        ann_dir = osp.join(self.ann_path, 'result.json')
        with open(ann_dir, 'r') as f:
            anns = json.load(f)
        
        self.num_joints = 13
        self.joints_name = anns['categories'][0]['keypoints']
        if args.root_relative:
            self.root_relative = True
            self.root_idx = self.joints_name.index('nose')
        else:
            self.root_relative = False
        self.skeleton = anns['categories'][0]['skeleton']
        self.flip_pairs = ( (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12) )
        self.data = self.make_data(anns)

    def make_data(self, anns, to_eighty_flag=False):
        data = []
        
        # dataloader split
        to_eighty_len = int( len(anns['annotations']) * 0.8 )
        if self.data_split == 'train':
            to_eighty_flag = True
        
        for i, b in enumerate(anns['annotations']):
            if to_eighty_flag: 
                if to_eighty_len < i:
                    continue
            else:
                if to_eighty_len > i:
                    continue
            
            # file path
            frame_idx = str(b['frame_idx'])
            pre_numb = frame_idx[:3]
            pro_numb = frame_idx[3:]
            input_img_path = os.path.join(self.ann_path, 'input_images', \
                                        pre_numb + 'image_display_' + pro_numb + '.png')
            depth_img_path =  os.path.join(self.ann_path, 'depth_images', \
                                        pre_numb + 'depth_' + pro_numb + '.png')
            
            # bounding box
            bbox = np.array(b['bbox'])
            
            # segmentation
            segmentation = np.array(b['segmentation'])
            
            # keypoints
            keypoints = np.array(b['keypoints'])            
            # joint_vis = np.ones((self.num_joints))
            joint_vis = keypoints[:, -1].copy()
            for i, j_v in enumerate(joint_vis):
                if j_v == 2:
                    joint_vis[i] = 1.
                else:
                    joint_vis[i] = 0.
            
            ####depth
            depth_img = np.array(Image.open(depth_img_path))
            depth_points = np.zeros(shape=(self.num_joints))
            for i, k in enumerate(keypoints):
                depth_points[i] = depth_img[round(k[1]), round(k[0])]
            depth_points = np.clip(depth_points, 0, 2000) # clipping
            
            keypoints[:, 2] = depth_points
            if self.root_relative:
                keypoints[:, 2] = keypoints[:, 2] - keypoints[self.root_idx, 2]
            
            # data
            data.append({
                'img_path' : input_img_path,
                'joint_img' : keypoints,
                'joint_vis' : joint_vis,
                'bbox' : bbox,
                'segmentation' : segmentation,
            })
            
        return data