from collections import OrderedDict, Iterable, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib
from .yolo_layer import *
from .yolov3_base import *


class Yolov3Tiny(Yolov3Base):

    def __init__(self, num_classes, use_wrong_previous_anchors=False):
        super().__init__()

        self.num_classes = num_classes
        self.return_out_boxes = False
        self.skip_backbone = False

        self.backbone = Yolov3TinyBackbone()

        anchors_per_region = 3
        self.yolo_0_pre = nn.Sequential(OrderedDict([
            ('14_convbatch',    ConvBN(256, 512, 3, 1, 1)),
            ('15_conv',         nn.Conv2d(512, anchors_per_region*(5+self.num_classes), 1, 1, 0)),
            # ('16_yolo',         YoloLayer()),
        ]))
        self.yolo_0 = YoloLayer(anchors=[(81.,82.), (135.,169.), (344.,319.)], stride=32, num_classes=num_classes)

        self.up_1 = nn.Sequential(OrderedDict([
            ('17_convbatch',    ConvBN(256, 128, 1, 1, 0)),
            ('18_upsample',     Upsample(2)),
        ]))

        self.yolo_1_pre = nn.Sequential(OrderedDict([
            ('19_convbatch',    ConvBN(128+256, 256, 3, 1, 1)),
            ('20_conv',         nn.Conv2d(256, anchors_per_region*(5+self.num_classes), 1, 1, 0)),
            # ('21_yolo',         YoloLayer()),
        ]))
        
        # Tiny yolo weights were originally trained using wrong anchor mask
        # https://github.com/pjreddie/darknet/commit/f86901f6177dfc6116360a13cc06ab680e0c86b0#diff-2b0e16f442a744897f1606ff1a0f99d3L175
        if use_wrong_previous_anchors:
            yolo_1_anchors = [(23.,27.),  (37.,58.),  (81.,82.)]
        else: 
            yolo_1_anchors = [(10.,14.),  (23.,27.),  (37.,58.)]

        self.yolo_1 = YoloLayer(anchors=yolo_1_anchors, stride=16.0, num_classes=num_classes)

    def get_loss_layers(self):
        return [self.yolo_0, self.yolo_1]

    # def boxes_from_output(self, outputs, conf_thresh=0.25):
    #     all_boxes = [[] for j in range(outputs[0].size(0))]
    #     for i, layer in enumerate(self.get_loss_layers()):
    #         layer_boxes = layer.get_region_boxes(outputs[i], conf_thresh=conf_thresh)
    #         for j, layer_box in enumerate(layer_boxes):
    #             all_boxes[j] += layer_box

    #     return all_boxes


    # def predict_img(self, imgs, conf_thresh=0.25):
    #     self.eval()
    #     if len(imgs.shape) == 3: imgs = imgs.unsqueeze(-1) 
        
    #     outputs = self.forward(imgs)
    #     return self.boxes_from_output(outputs, conf_thresh)


    def forward_yolo(self, xb):
        x_b_0, x_b_full = xb[0], xb[1]
        y0 = self.yolo_0_pre(x_b_full)

        x_up = self.up_1(x_b_full)
        x_up = torch.cat((x_up, x_b_0), 1)
        y1 = self.yolo_1_pre(x_up)

        return y0, y1


    # def forward(self, *x):
    #     #x_b_0, x_b_full = self.backbone(x) if not self.skip_backbone else x[0], x[1]
    #     if (isinstance(x, tuple) or isinstance(x, list)) and len(x) == 2:
    #         x_b_0, x_b_full = x[0], x[1]
    #     else:
    #         x_b_0, x_b_full = self.backbone(x[0])

    #     out = self.forward_yolo(x_b_0, x_b_full)
    #     return out


    # def freeze_backbone(self, requires_grad=False):
    #     for _, p in self.backbone.named_parameters():
    #         p.requires_grad = requires_grad
    # def unfreeze(self):
    #     for _, p in self.named_parameters():
    #         p.requires_grad = True
    # def freeze_info(self, print_all=False):
    #     d = defaultdict(set)
    #     for name, param in self.named_parameters():
    #         if print_all:
    #             print(f"{name}: {param.requires_grad}")
    #         else:
    #             d[name.split('.')[0]].add(param.requires_grad)
    #     if not print_all:
    #         for k,v in d.items():
    #             print(k, ': ', v)
        

    # def load_only_backbone(self, h5_path):
    #     state_dict = torch.load(h5_path)

    #     for k in list(state_dict.keys()):
    #         if k.startswith(('yolo_0_pre.15', 'yolo_1_pre.20')):
    #             del state_dict[k]

    #     # Renaming some keys if needed for compatibility
    #     # state_dict = type(state_dict_org)()
    #     # for k_old in list(state_dict.keys()):
    #     #     k_new = k_old.replace('backend', 'backbone')
    #     #     state_dict[k_new] = state_dict_org[k_old]

    #     return self.load_state_dict(state_dict, strict=False)


###################################################################
## Backbone and helper modules

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x


class Yolov3TinyBackbone(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.layers_list =  OrderedDict([
            ('0_convbatch',     ConvBN(input_channels, 16, 3, 1, 1)),
            ('1_max',           nn.MaxPool2d(2, 2)),
            ('2_convbatch',     ConvBN(16, 32, 3, 1, 1)),
            ('3_max',           nn.MaxPool2d(2, 2)),
            ('4_convbatch',     ConvBN(32, 64, 3, 1, 1)),
            ('5_max',           nn.MaxPool2d(2, 2)),
            ('6_convbatch',     ConvBN(64, 128, 3, 1, 1)),
            ('7_max',           nn.MaxPool2d(2, 2)),
            ('8_convbatch',     ConvBN(128, 256, 3, 1, 1)),
            ('9_max',           nn.MaxPool2d(2, 2)),
            # ('9_max',           nn.MaxPool2d(2, 2, ceil_mode=True)), 
            ('10_convbatch',    ConvBN(256, 512, 3, 1, 1)),
            ('11_max',          MaxPoolStride1()),
            ('12_convbatch',    ConvBN(512, 1024, 3, 1, 1)),
            ('13_convbatch',    ConvBN(1024, 256, 1, 1, 0)), # padding = kernel_size-1//2
        ])
        self.layers = nn.Sequential(self.layers_list)
        self.idx = 9
    
    def forward(self, x):
        x_b_0 = self.layers[:self.idx](x)
        x_b_full = self.layers[self.idx:](x_b_0)
        return x_b_0, x_b_full

