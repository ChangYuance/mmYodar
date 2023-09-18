from collections import OrderedDict

import torch
import torch.nn as nn
import cv2
import matplotlib as plt
import numpy as np
from mini_net.darknet import darknet53,darknet33,darknet23

def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m


class depthpctBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False, trainmode = "pct23"):
        super(depthpctBody, self).__init__()

        
        if trainmode=="color23":
            self.backbone = darknet23()
            out_filters = self.backbone.layers_out_filters
        if trainmode=="color33":
            self.backbone = darknet33()
            out_filters = self.backbone.layers_out_filters
        if trainmode=="color53":
            self.backbone = darknet53()
            out_filters = self.backbone.layers_out_filters
        if trainmode=="fusion":
            self.depthbackbone = darknet33()
            self.pctbackbone   = darknet33()
            out_filters = self.depthbackbone.layers_out_filters
        if trainmode=="depth53":
            self.backbone = darknet53()
            out_filters = self.backbone.layers_out_filters
        if trainmode=="depth33":
            self.backbone = darknet33()
            out_filters = self.backbone.layers_out_filters
        if trainmode=="depth23":
            self.backbone = darknet23()
            out_filters = self.backbone.layers_out_filters
        if trainmode=="pct53":
            self.backbone = darknet53()
            out_filters = self.backbone.layers_out_filters
        if trainmode=="pct33":
            self.backbone = darknet33()
            out_filters = self.backbone.layers_out_filters
        if trainmode=="pct23":
            self.backbone = darknet23()
            out_filters = self.backbone.layers_out_filters
        # if trainmode=="fusioncolor":
        #     self.colorbackbone = darknet53()
        #     self.pctbackbone = darknet23()
        #     out_filters = self.colorbackbone.layers_out_filters
        # if trainmode=="earlyconcatcolor":
        #     self.earlybackbone = darknet53()
        #     out_filters = self.earlybackbone.layers_out_filters
        # self.backbone2 = darknet23()
        #33是fusion训练
        # self.backbone3 = darknet33()
        # if pretrained:
        #     self.backbone.load_state_dict(torch.load("logs/last_epoch_weights.pth"))
        if trainmode=="pretrained":
            self.backbone = darknet23()
            out_filters = self.backbone.layers_out_filters
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.last_layer0            = make_last_layers([256, 512], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))
        self.last_layer1_conv       = conv2d(256, 128, 1)
        self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1            = make_last_layers([128, 256], out_filters[-2] + 128, len(anchors_mask[1]) * (num_classes + 5))
        self.last_layer2_conv       = conv2d(128, 64, 1)
        self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2            = make_last_layers([64, 128], out_filters[-3] + 64, len(anchors_mask[2]) * (num_classes + 5))
        self.istrain = True
    def forward(self, x):
        x2, x1, x0 = self.backbone(x)

        out0_branch = self.last_layer0[:5](x0)
        out0        = self.last_layer0[5:](out0_branch)

        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)


        x1_in = torch.cat([x1_in, x1], 1)

        out1_branch = self.last_layer1[:5](x1_in)
        out1        = self.last_layer1[5:](out1_branch)


        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        x2_in = torch.cat([x2_in, x2], 1)

        out2 = self.last_layer2(x2_in)
        if self.istrain:
            return out0, out1, out2, x0, x1, x2
        else:
            return out0, out1, out2