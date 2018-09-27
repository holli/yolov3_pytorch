import pytest, torch, unittest, bcolz, pickle
import numpy as np
# from unittest import mock
# from unittest.mock import Mock
# from testfixtures import tempdir

from yolov3_pytorch import utils
from yolov3_pytorch.yolo_layer import *


class YoloGetRegionBoxesTest(unittest.TestCase):
    
    def test_get_region_boxes_large(self):
        # output = model(img_torch)[0]
        # region_boxes = model.yolo_0.get_region_boxes(output, conf_thresh=.25)
        # region_boxes = [[[i.item() for i in box] for box in region_boxes[0]]]
        # attrs = {'conf_thresh': .25, 'num_classes': model.yolo_0.num_classes, 'stride': model.yolo_0.stride,
        #           'anchors': model.yolo_0.anchors}
        # key_val = {'output': output, 'region_boxes': region_boxes, 'attrs': attrs}
        # pickle.dump(key_val, open("tests/data/yololayer_tiny_0_get_region_boxes_0.p", "wb"))

        key_val = pickle.load(open("tests/mocks/yololayer_tiny_0_get_region_boxes_0.p", "rb"))
        attrs = key_val['attrs']; output = key_val['output']; target = key_val['region_boxes']

        yolo = YoloLayer(anchors=attrs['anchors'], stride=attrs['stride'], num_classes=attrs['num_classes'])
        
        region_boxes = yolo.get_region_boxes(output, conf_thresh=attrs['conf_thresh'])
        region_boxes = [[[float(i) for i in box] for box in region_boxes[0]]]

        region_boxes = sorted(region_boxes[0], key=lambda x: x[0])
        target = sorted(target[0], key=lambda x: x[0])

        self.assertEqual(region_boxes, target)


class YoloGetLossTest(unittest.TestCase):

    # Pickle file was created when it was working. This is mainly to check that refactorings wont break the code.
    def test_get_loss_large_data(self):
        key_val = pickle.load(open("tests/mocks/yololayer_tiny_0_get_loss_0.p", "rb"))
        attrs = key_val['yolo']; output = key_val['output']; target = key_val['target']; losses = key_val['losses']

        yolo = YoloLayer(anchors=attrs['anchors'], stride=attrs['stride'], num_classes=attrs['num_classes'])
        
        loss_total, loss_coord, loss_conf, loss_cls = yolo.get_loss(output, target, return_single_value=False)

        self.assertAlmostEqual(losses[0], loss_total.item())
        self.assertAlmostEqual(losses[1], loss_coord.item())
        self.assertAlmostEqual(losses[2], loss_conf.item())
        self.assertAlmostEqual(losses[3], loss_cls.item())

    def test_get_loss_large_data_cuda(self):
        key_val = pickle.load(open("tests/mocks/yololayer_tiny_0_get_loss_0.p", "rb"))
        attrs = key_val['yolo']; output = key_val['output']; target = key_val['target']; losses = key_val['losses']

        output.to('cuda')

        yolo = YoloLayer(anchors=attrs['anchors'], stride=attrs['stride'], num_classes=attrs['num_classes'])
        
        loss_total, loss_coord, loss_conf, loss_cls = yolo.get_loss(output, target, return_single_value=False)

        self.assertAlmostEqual(losses[0], loss_total.item())
        self.assertAlmostEqual(losses[1], loss_coord.item())
        self.assertAlmostEqual(losses[2], loss_conf.item())
        self.assertAlmostEqual(losses[3], loss_cls.item())


    # def test_get_loss_calculated(self):
    #     yolo = YoloLayer(anchors=attrs['anchors'], stride=attrs['stride'], num_classes=attrs['num_classes'])
        




