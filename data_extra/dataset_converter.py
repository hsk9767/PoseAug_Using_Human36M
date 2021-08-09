import torch

def COCO2HUMAN(coco_keypoints):
    pelvis_keypoints = ((coco_keypoints[:, 11, :] + coco_keypoints[:, 12, :]) / 2.).unsqueeze(1)
    head_keypoints = ((coco_keypoints[:, 1, :] + coco_keypoints[:, 2, :]) / 2.).unsqueeze(1)
    thorax_keypoints = ((coco_keypoints[:, 5, :] + coco_keypoints[:, 6, :]) / 2.).unsqueeze(1)
    torso_keypoints = ((pelvis_keypoints + thorax_keypoints) / 2.)
    
    pelvis_torso = coco_keypoints[:,[12, 14, 16, 11, 13, 15], :]
    head_thorax = coco_keypoints[:, [5, 7, 9, 6, 8, 10], :]
    
    keypoints = torch.cat([pelvis_keypoints, pelvis_torso, torso_keypoints, head_keypoints,head_thorax, thorax_keypoints], dim=1)
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
    torso_keypoints = ((mpii_keypoints[:, 6, :] + mpii_keypoints[:, 7, :]) / 2.).unsqueeze(1)
    pelvis_torso = mpii_keypoints[:,[6, 2, 1, 0, 3, 4, 5], :]
    torso_thorax = mpii_keypoints[:, [9, 13, 14, 15, 12, 11, 10, 7], :]
    keypoints = torch.cat([pelvis_torso, torso_keypoints, torso_thorax], dim=1)
    
    return keypoints
    
    
    