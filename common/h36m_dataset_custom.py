import os
import os.path as osp
from pycocotools.coco import COCO
import numpy as np
import cv2
import random
import json
from utils.data_utils import world2cam, cam2pixel, pixel2cam, process_bbox, cam2pixel_custom

class Human36M:
    # if original -> 18 points as 3DMPPE
    # else -> 16 keypoints as PoseAug
    def __init__(self, data_split, original=False):
        print(f'==> {data_split} dataset of Human3.6M is being loaded..')
        self.original = original
        self.data_split = data_split
        self.img_dir = osp.join('./data/Human3.6M/images')
        self.annot_path = osp.join('./data/Human3.6M/annotations')
        self.human_bbox_root_dir = osp.join('./data/bbox_root/bbox_root_human36m_output.json')
        # self.joint_num = 18 # original:17, but manually added 'Thorax'
        if original:
            self.joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax')
            self.flip_pairs = ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) )
            self.joint_num = 18
            self.skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
            self.eval_joint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 11, 12, 13, 14, 15, 16)
        else:
            self.joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax')
            self.flip_pairs = ( (1, 4), (2, 5), (3, 6), (12, 9), (13, 10), (14, 11) )
            self.joint_num = 16 # removed 'nose' and 'neck'
            self.skeleton = ( (8, 15), (15, 9), (9, 10), (10, 11), (15, 12), (12, 13), (13, 14), (15, 7), (7, 0), (0, 4), (4, 5), (5, 6), (0, 1), (1, 2), (2, 3) )
            self.eval_joint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        
        self.joints_have_depth = True
        self.joint_select = [0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17]
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
        self.root_idx = self.joints_name.index('Pelvis')
        self.lshoulder_idx = 11
        self.rshoulder_idx = 14
        self.protocol = 2
        self.data = self.load_data()
        

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test':
            return 64
        elif self.data_split == 'vis':
            return 2
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            if self.protocol == 1:
                subject = [1,5,6,7,8,9]
            elif self.protocol == 2:
                subject = [1,5,6,7,8]
        elif self.data_split == 'test':
            if self.protocol == 1:
                subject = [11]
            elif self.protocol == 2:
                subject = [9,11]
        elif self.data_split == 'vis':
            subject = [9]
        else:
            assert 0, print("Unknown subset")

        return subject
    
    def add_thorax(self, joint_coord):
        thorax = (joint_coord[self.lshoulder_idx, :] + joint_coord[self.rshoulder_idx, :]) * 0.5
        thorax = thorax.reshape((1, 3))
        joint_coord = np.concatenate((joint_coord, thorax), axis=0)
        return joint_coord

    def load_data(self):
        print('==> Load data of H36M Protocol ' + str(self.protocol))

        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()
        
        # aggregate annotations from each subject
        db = COCO()
        cameras = {}
        joints = {}
        for subject in subject_list:
            # data load
            # print(os.path.abspath(__file__))
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'),'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'),'r') as f:
                joints[str(subject)] = json.load(f)
        db.createIndex()
    
        # if self.data_split == 'test' and not True:
        #     print("Get bounding box and root from " + self.human_bbox_root_dir)
        #     bbox_root_result = {}
        #     with open(self.human_bbox_root_dir) as f:
        #         annot = json.load(f)
        #     for i in range(len(annot)):
        #         bbox_root_result[str(annot[i]['image_id'])] = {'bbox': np.array(annot[i]['bbox']), 'root': np.array(annot[i]['root_cam'])}
        # else:
        print("==> Get bounding box and root from groundtruth")

        data = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            img_width, img_height = img['width'], img['height']
        
            # check subject and frame_idx
            subject = img['subject']; frame_idx = img['frame_idx'];
            if subject not in subject_list:
                continue
            if frame_idx % sampling_ratio != 0:
                continue

            # camera parameter
            cam_idx = img['cam_idx']
            cam_param = cameras[str(subject)][str(cam_idx)]
            R,t,f,c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
            
            # project world coordinate to cam, image coordinate space
            action_idx = img['action_idx']; subaction_idx = img['subaction_idx']; frame_idx = img['frame_idx'];
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)
            joint_world = self.add_thorax(joint_world)
            joint_cam = world2cam(joint_world, R, t)
            # joint_img = cam2pixel(joint_cam, f, c)
            joint_img = cam2pixel_custom(joint_cam, f, c)
            joint_img[:,2] = joint_img[:,2] - joint_cam[self.root_idx,2]
            joint_vis = np.ones((self.joint_num,1))
            
            bbox = process_bbox(np.array(ann['bbox']), img_width, img_height)
            if bbox is None: continue
            root_cam = joint_cam[self.root_idx]
            
            if not self.original:
                joint_img = joint_img[self.joint_select]
                joint_cam = joint_cam[self.joint_select]
            
            data.append({
                'img_width' : img_width,
                'img_height' : img_height,
                'img_path': img_path,
                'img_id': image_id,
                'bbox': bbox,
                'joint_img': joint_img, # [org_img_x, org_img_y, depth - root_depth]
                'joint_cam': joint_cam, # [X, Y, Z] in camera coordinate
                'joint_vis': joint_vis,
                'root_cam': root_cam, # [X, Y, Z] in camera coordinate
                'f': f,
                'c': c})
        
        return data