import torch
import numpy as np
def COCO2HUMAN(coco_keypoints):
    pelvis_keypoints = np.expand_dims((coco_keypoints[:, 11, :] + coco_keypoints[:, 12, :]) / 2., 1)
    head_keypoints = np.expand_dims((coco_keypoints[:, 1, :] + coco_keypoints[:, 2, :]) / 2., 1)
    thorax_keypoints = np.expand_dims((coco_keypoints[:, 5, :] + coco_keypoints[:, 6, :]) / 2., 1)
    torso_keypoints = ((pelvis_keypoints + thorax_keypoints) / 2.)
    
    pelvis_torso = coco_keypoints[:,[12, 14, 16, 11, 13, 15], :]
    head_thorax = coco_keypoints[:, [5, 7, 9, 6, 8, 10], :]
    
    keypoints = np.concatenate([pelvis_keypoints, pelvis_torso, torso_keypoints, head_keypoints,head_thorax, thorax_keypoints], 1)
    return keypoints
    
def MPII2HUMAN(mpii_keypoints):
    '''
    0 - r_ankle
    1 - r_knee
    2 - r_hip
    3 - l_hip
    4 - l_knee
    5 - l_ankle
    6 - pelvis
    7 - thorax
    8 - neck
    9 - headtop
    10 - r_wrist
    11 - r_elbow
    12 - r_shoulder
    13 - l_shoulder
    14 - l_elbow
    15 - l_wrist
    '''
    
    torso_keypoints = np.expand_dims((mpii_keypoints[:, 6, :] + mpii_keypoints[:, 7, :]) / 2., 1)
    pelvis_torso = mpii_keypoints[:,[6, 2, 1, 0, 3, 4, 5], :]
    torso_thorax = mpii_keypoints[:, [9, 13, 14, 15, 12, 11, 10, 7], :]
    keypoints = np.concatenate([pelvis_torso, torso_keypoints, torso_thorax],1)
    
    return keypoints
    

def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]))

    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint
    