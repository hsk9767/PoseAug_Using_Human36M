from pelee.lib.models.MOBIS_peleenet import get_pose_pelee_net
import torch

pelee_net = get_pose_pelee_net(is_train=False)
dummy_input = torch.rand(size=(1, 3, 256, 256))
pelee_net.load_state_dict(torch.load('pelee/output/coco/pose_resnet_50/256x192_d256x3_adam_lr1e-3/final_state.pth.tar', map_location='cpu'))

print(pelee_net(dummy_input).shape)

