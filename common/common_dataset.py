import numpy as np
import cv2
import random
import time
import torch
import copy
import math
from torch.utils.data.dataset import Dataset
from data_extra.dataset_converter import COCO2HUMAN, MPII2HUMAN

input_shape = (256, 256) 
bbox_3d_shape = (2000, 2000, 2000)

class DatasetLoader(Dataset):
    def __init__(self, db, ref_joints_name, is_train, transform, detection_2d=False):
        
        self.db = db.data
        self.joint_num = db.joint_num
        self.skeleton = db.skeleton
        self.flip_pairs = db.flip_pairs
        self.joints_have_depth = db.joints_have_depth
        self.joints_name = db.joints_name
        self.ref_joints_name = ref_joints_name
        
        self.transform = transform
        self.is_train = is_train

        if self.is_train:
            self.do_augment = True
        else:
            self.do_augment = False
        
        # to save 2D keypoints
        if detection_2d:
            self.bbox_output = True
        else:
            self.bbox_output = False

    def __getitem__(self, index):
        
        joints_have_depth = self.joints_have_depth
        data = copy.deepcopy(self.db[index])

        bbox = data['bbox'] # top_left_x, top_left_y, width, height
        joint_img = data['joint_img']
        joint_cam = data['joint_cam']
        joint_vis = data['joint_vis']
        img_width, img_height = data['img_width'], data['img_height']
        
        # 1. load image
        cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % data['img_path'])
        img_height, img_width, img_channels = cvimg.shape

        scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False

        # 3. crop patch from img and perform data augmentation (flip, rot, color scale, synthetic occlusion)
        img_patch, trans = generate_patch_image(cvimg, bbox, do_flip, scale, rot, do_occlusion)
        for i in range(img_channels):
            img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)
        
        # joint_img in image plane to bbox plane
        for i in range(len(joint_img)):
            joint_img[i, 0:2] = trans_point2d(joint_img[i, 0:2], trans)
            joint_img[i, 2] /= (2000/2.) # expect depth lies in -bbox_3d_shape[0]/2 ~ bbox_3d_shape[0]/2 -> -1.0 ~ 1.0
            joint_img[i, 2] = (joint_img[i,2] + 1.0)/2. # 0~1 normalize
            joint_vis[i] *= (
                            (joint_img[i,0] >= 0) & \
                            (joint_img[i,0] < input_shape[1]) & \
                            (joint_img[i,1] >= 0) & \
                            (joint_img[i,1] < input_shape[0]) & \
                            (joint_img[i,2] >= 0) & \
                            (joint_img[i,2] < 1)
                            )
        
        # normalize [-1, 1], because the 2D keypoints are input of the lifting network
        joint_img = joint_img[:, :2]
        joint_img = normalize_screen_coordinates(joint_img, input_shape[0], input_shape[1])
        ## to meter unit
        joint_cam = joint_cam / 1000.
        
        img_patch = self.transform(img_patch)
        joint_img = joint_img.astype(np.float32)
        joint_cam = joint_cam.astype(np.float32)
        joint_vis = (joint_vis > 0).astype(np.float32)
        joints_have_depth = np.array([joints_have_depth]).astype(np.float32)

        if not self.bbox_output:
            return img_patch, joint_img, joint_cam, joint_vis
        else:
            return img_patch, joint_img, joint_cam, joint_vis, bbox, img_width, img_height
    
    def __len__(self):
        return len(self.db)
    
class DatasetLoader_only_lifting(Dataset):
    def __init__(self, db, ref_joints_name, is_train, transform, path_2d=None, keypoints=None):
        
        self.db = db.data
        self.joint_num = db.joint_num
        self.skeleton = db.skeleton
        self.flip_pairs = db.flip_pairs
        self.joints_have_depth = db.joints_have_depth
        self.joints_name = db.joints_name
        self.ref_joints_name = ref_joints_name
        
        self.transform = transform
        self.is_train = is_train

        if self.is_train:
            self.do_augment = True
        else:
            self.do_augment = False
        self.joints_left = [4, 5, 6, 9, 10, 11]
        self.joints_right = [1, 2, 3, 12, 13, 14]
        
        # use the .npz file
        self.keypoints = keypoints
        if self.keypoints != 'gt':
            train_path = self.keypoints + '_train.npz'
            valid_path = self.keypoints + '_valid.npz'
            if self.is_train:
                path_2d = train_path
            else:
                path_2d = valid_path
            keypoints_2d = np.load(path_2d)
            x_2d = np.expand_dims(keypoints_2d['x'], 2)
            y_2d = np.expand_dims(keypoints_2d['y'], 2)
            self.keypoints_2d = np.concatenate([x_2d, y_2d], 2)
            assert len(self.keypoints) != len(self.db)

    def __getitem__(self, index):
        
        joint_num = self.joint_num
        skeleton = self.skeleton
        flip_pairs = self.flip_pairs
        joints_have_depth = self.joints_have_depth

        data = copy.deepcopy(self.db[index])

        bbox = data['bbox']
        joint_img = data['joint_img']
        joint_cam = data['joint_cam']
        joint_vis = data['joint_vis']

        # normalize
        joint_img = joint_img[:, :2]
        img_height, img_width = data['img_height'], data['img_width']
        joint_img = normalize_screen_coordinates(joint_img, img_width, img_height)

        ## to meter unit
        joint_cam = joint_cam / 1000.
        
        # not GT(2D)
        if self.keypoints == 'pelee':
            joint_img = COCO2HUMAN(self.keypoints_2d[index]) # already normalized
        elif self.keypoints == 'resnet':
            joint_img = MPII2HUMAN(self.keypoints_2d[index]) # already normalized
        
        joint_img = joint_img.astype(np.float32)
        joint_cam = joint_cam.astype(np.float32)
        joint_vis = (joint_vis > 0).astype(np.float32)
        joints_have_depth = np.array([joints_have_depth]).astype(np.float32)

        return [-1], joint_img, joint_cam, joint_vis
    
    def __len__(self):
        return len(self.db)

class DatasetLoader_only_inferencing(Dataset):
    def __init__(self, db, ref_joints_name, is_train, transform):
        
        self.db = db.data
        self.joint_num = db.joint_num
        self.skeleton = db.skeleton
        self.flip_pairs = db.flip_pairs
        self.joints_have_depth = db.joints_have_depth
        self.joints_name = db.joints_name
        self.ref_joints_name = ref_joints_name
        
        self.transform = transform
        self.is_train = is_train

        if self.is_train:
            self.do_augment = True
        else:
            self.do_augment = False

    def __getitem__(self, index):
        
        joint_num = self.joint_num
        skeleton = self.skeleton
        flip_pairs = self.flip_pairs
        joints_have_depth = self.joints_have_depth

        data = copy.deepcopy(self.db[index])

        bbox = data['bbox']
        joint_img = data['joint_img']
        joint_cam = data['joint_cam']
        joint_vis = data['joint_vis']
        
        scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False
        bb_c_x = float(bbox[0] + 0.5*bbox[2])
        bb_c_y = float(bbox[1] + 0.5*bbox[3])
        bb_width = float(bbox[2])
        bb_height = float(bbox[3])
        # trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0], scale, rot, inv=False)
        
        # joint_img in image plane to bbox plane
        # for i in range(len(joint_img)):
        #     joint_img[i, 0:2] = trans_point2d(joint_img[i, 0:2], trans)
        #     joint_img[i, 2] /= (2000/2.) # expect depth lies in -bbox_3d_shape[0]/2 ~ bbox_3d_shape[0]/2 -> -1.0 ~ 1.0
        #     joint_img[i, 2] = (joint_img[i,2] + 1.0)/2. # 0~1 normalize
        #     joint_vis[i] *= (
        #                     (joint_img[i,0] >= 0) & \
        #                     (joint_img[i,0] < input_shape[1]) & \
        #                     (joint_img[i,1] >= 0) & \
        #                     (joint_img[i,1] < input_shape[0]) & \
        #                     (joint_img[i,2] >= 0) & \
        #                     (joint_img[i,2] < 1)
        #                     )
        
        # make the top right of the bounding box as (0, 0)
        # joint_img[:, 0] -= bbox[0]
        # joint_img[:, 1] -= bbox[1]
        
        # normalize [-1, 1], because the 2D keypoints are input of the lifting network
        joint_img = joint_img[:, :2]
        # joint_img = normalize_screen_coordinates(joint_img, input_shape[0], input_shape[1])
        # joint_img = normalize_screen_coordinates(joint_img, bb_width, bb_height)
        img_height, img_width = data['img_height'], data['img_width']
        # normalize
        joint_img = normalize_screen_coordinates(joint_img, img_width, img_height)
        
        ## to meter unit
        joint_cam = joint_cam / 1000.
            
        
        # img_patch = self.transform(img_patch)
        joint_img = joint_img.astype(np.float32)
        joint_cam = joint_cam.astype(np.float32)
        joint_vis = (joint_vis > 0).astype(np.float32)
        joints_have_depth = np.array([joints_have_depth]).astype(np.float32)

        return [-1], joint_img, joint_cam, joint_vis
    
    def __len__(self):
        return len(self.db)
    
    
# helper functions
def get_aug_config():
    
    scale_factor = 0.25
    rot_factor = 30
    color_factor = 0.2
    
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    do_flip = random.random() <= 0.5
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

    do_occlusion = random.random() <= 0.5

    return scale, rot, do_flip, color_scale, do_occlusion


def generate_patch_image(cvimg, bbox, do_flip, scale, rot, do_occlusion):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    # synthetic occlusion
    if do_occlusion:
        while True:
            area_min = 0.0
            area_max = 0.7
            synth_area = (random.random() * (area_max - area_min) + area_min) * bbox[2] * bbox[3]

            ratio_min = 0.3
            ratio_max = 1/0.3
            synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

            synth_h = math.sqrt(synth_area * synth_ratio)
            synth_w = math.sqrt(synth_area / synth_ratio)
            synth_xmin = random.random() * (bbox[2] - synth_w - 1) + bbox[0]
            synth_ymin = random.random() * (bbox[3] - synth_h - 1) + bbox[1]

            if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < img_width and synth_ymin + synth_h < img_height:
                xmin = int(synth_xmin)
                ymin = int(synth_ymin)
                w = int(synth_w)
                h = int(synth_h)
                img[ymin:ymin+h, xmin:xmin+w, :] = np.random.rand(h, w, 3) * 255
                break

    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1
    
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0], scale, rot, inv=False)
    img_patch = cv2.warpAffine(img, trans, (int(input_shape[1]), int(input_shape[0])), flags=cv2.INTER_LINEAR)

    img_patch = img_patch[:,:,::-1].copy()
    img_patch = img_patch.astype(np.float32)

    return img_patch, trans

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]

# def normalize_screen_coordinates(X, w, h):
#     X[:, 0] = X[:, 0] / w * 2 - 1
#     X[:, 1] = X[:, 1] / h * 2 -1
#     return X