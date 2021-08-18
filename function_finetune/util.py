import torch
import torch.nn.functional as F
def soft_argmax(heatmaps):
    joint_num = heatmaps.shape[1]
    out_y, out_x = heatmaps.shape[-2:]
    
    heatmaps = heatmaps.reshape((-1, joint_num,out_x*out_y))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, out_y, out_x))

    accu_x = heatmaps.sum(dim=(2))
    accu_y = heatmaps.sum(dim=(3))

    accu_x = accu_x * torch.arange(out_x).float().cuda()[None,None,:]
    accu_y = accu_y * torch.arange(out_y).float().cuda()[None,None,:]
    
    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y), dim=2)

    return coord_out