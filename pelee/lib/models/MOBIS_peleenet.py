from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from collections import OrderedDict
import logging
import math
import os
import sys
logger = logging.getLogger(__name__)
class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, **kwargs):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        out = self.norm(self.conv(x))
        if self.activation:
            out = F.relu(out, inplace=True)
        return out

class conv_relu(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        out = F.relu(self.conv(x), inplace=True)
        return out


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bottleneck_width, drop_rate):
        super(_DenseLayer, self).__init__()
        growth_rate = growth_rate // 2
        inter_channel = int(growth_rate * bottleneck_width / 4) * 4
        if inter_channel > num_input_features / 2:
            inter_channel = int(num_input_features / 8) * 4
            print('adjust inter_channel to: ', inter_channel)
        self.branch1a = conv_bn_relu(num_input_features, inter_channel, kernel_size=1)
        self.branch1b = conv_bn_relu(inter_channel, growth_rate, kernel_size=3, padding=1)

        self.branch2a = conv_bn_relu(num_input_features, inter_channel, kernel_size=1)
        self.branch2b = conv_bn_relu(inter_channel, growth_rate, kernel_size=3, padding=1)
        self.branch2c = conv_bn_relu(growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out1 = self.branch1a(x)
        out1 = self.branch1b(out1)

        out2 = self.branch2a(x)
        out2 = self.branch2b(out2)
        out2 = self.branch2c(out2)

        out = torch.cat([x, out1, out2], dim=1)
        return out


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

            
class _StemBlock(nn.Module):
    def __init__(self, num_input_channels, num_init_features):
        super(_StemBlock, self).__init__()
        num_stem_features = int(num_init_features / 2)

        self.stem1 = conv_bn_relu(num_input_channels,
                                  num_init_features,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1)
        self.stem2a = conv_bn_relu(num_init_features,
                                   num_stem_features,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        self.stem2b = conv_bn_relu(num_stem_features,
                                   num_init_features,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1)
        self.stem3 = conv_bn_relu(2*num_init_features,
                                  num_init_features,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        out = self.stem1(x)

        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)

        branch1 = self.pool(out)

        out = torch.cat([branch1, branch2], dim=1)
        out = self.stem3(out)

        return out


class PeleeNet(nn.Module):
    def __init__(self, nof_joints=17, bn_momentum=0.1):
        super(PeleeNet, self).__init__()
        self.deconv_with_bias = False
        #self.phase = 'train'
        self.num_pelee_out = 704
        self.num_init_feat = 32
        self.growth_rates = [32, 32, 32, 32]
        self.bottleneck_width = [1, 2, 4, 4]
        self.block_config = [3, 4, 8, 6]
        self.drop_rate = 0.05
        self.bn_momentum = bn_momentum
        self.features = nn.Sequential(
            OrderedDict([('stemblock', _StemBlock(3, self.num_init_feat))])
        )
        # Each denseblock
        num_features = self.num_init_feat
        for i, num_layers in enumerate(self.block_config):
            block = _DenseBlock(
                        num_layers=num_layers,
                        num_input_features=num_features,
                        bn_size=self.bottleneck_width[i],
                        growth_rate=self.growth_rates[i],
                        drop_rate=self.drop_rate)
            
            self.features.add_module(
                'denseblock%d' % (i + 1), 
                block
            )
            
            num_features = num_features + num_layers * self.growth_rates[i]
            
            self.features.add_module(
                'transition%d' % (i + 1),
                conv_bn_relu(num_features,
                             num_features,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            )
            
            if i != len(self.block_config) - 1:
                self.features.add_module(
                    'transition%d_pool' % (i + 1),
                    nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
                )
        self.deconv_layer1 = self._make_single_deconv(3, 128, 4, idx=0)
        self.deconv_layer2 = self._make_single_deconv(3, 128, 4, idx=1)
        self.deconv_layer3 = self._make_single_deconv(3, 128, 4, idx=2)

        self.final_layer = nn.Conv2d(
            in_channels=128,
            out_channels=nof_joints,
            kernel_size=1,
            padding=0
        )
        
    def _make_single_deconv(self, num_layer, num_filter, num_kernel, idx):
        layers = []
        if idx == 0:
            inplanes = 704
        else:
            inplanes = 128
        
        layers.append(
            nn.ConvTranspose2d(
                in_channels=inplanes,
                out_channels=num_filter,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False
            )
        )
        layers.append(nn.BatchNorm2d(num_filter, momentum=self.bn_momentum))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        for k, feat in enumerate(self.features):
            x = feat(x)

        x = self.deconv_layer1(x)
        x = self.deconv_layer2(x)
        x = self.deconv_layer3(x)
        x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            for key in list(pretrained_state_dict.keys()):
                val = pretrained_state_dict[key]
                if key.find('final_layer.weight') != -1:
                    del pretrained_state_dict[key]
                if key.find('final_layer.bias') != -1:
                    del pretrained_state_dict[key]
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)

def get_pose_pelee_net(is_train, num_joints=17):
    if is_train:
        print(">>Peleenet Start")
        model = PeleeNet(nof_joints=16)
        model.init_weights('./peleenet.pth')
    else:
        model = PeleeNet()

    return model
