import torch
from common import get_resnet
from function_baseline.config import get_parse_args
args = get_parse_args()
model = get_resnet(args)
dummy = torch.rand(size=(1, 3, 256, 256)).cuda()
print(model(dummy).shape)
