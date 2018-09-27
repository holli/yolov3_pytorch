
import pytest, torch, unittest, bcolz
import numpy as np
# from unittest import mock
# from unittest.mock import Mock
# from testfixtures import tempdir

from yolov3_pytorch import utils


class NmsTest(unittest.TestCase):
    def test_empty(self):
        result = utils.nms([], .4)
        self.assertEqual(len(result), 0)

    def test_basic_big_array(self):
        boxes = [[.2, .2, .2, .2, .8], [.22, .22, .2, .2, 0.7], [.6, .6, .2, .2, 0.5], [.22, .22, .22, .22, 0.2], [.61, .61, .2, .2, 0.2],
                 [.2, .2, .01, .01, .2]]
        result = utils.nms(boxes, .4)
        self.assertEqual(result, [[.2, .2, .2, .2, .8], [.6, .6, .2, .2, 0.5], [.2, .2, .01, .01, .2]])
        
    def test_basic_array(self):
        boxes = [[.2, .2, .2, .2, .8], [.3, .2, .2, .2, 0.7]]
        result = utils.nms(boxes, .2)
        self.assertEqual(boxes, [[.2, .2, .2, .2, .8], [.3, .2, .2, .2, 0.7]], "nms should not change input")
        self.assertEqual(result, [[.2, .2, .2, .2, .8]])
        
        boxes = [[.2, .2, .2, .2, .8], [.3, .2, .2, .2, 0.7]]
        result = utils.nms(boxes, .7)
        self.assertEqual(boxes, [[.2, .2, .2, .2, .8], [.3, .2, .2, .2, 0.7]], "nms should not change input")
        self.assertEqual(result, boxes)


class BboxIouTest(unittest.TestCase):
    def test_bbox_iou_xywh_torch(self):
        box1 = torch.FloatTensor([.2, .2, .2, .2]) # from 0.1 to 0.3
        box2 = torch.FloatTensor([.3, .3, .2, .2]) # from 0.2 to 0.4
        iou = utils.bbox_iou(box1, box2, x1y1x2y2=False)

        intersect = (.1**2)
        iou_expected = intersect/(.2**2+.2**2-intersect)
        self.assertAlmostEqual(iou, iou_expected, places=5)

    def test_bbox_iou_xywh_numpy(self):
        box1 = np.array([.2, .2, .2, .2]) # from 0.1 to 0.3
        box2 = np.array([.3, .3, .2, .2]) # from 0.2 to 0.4
        iou = utils.bbox_iou(box1, box2, x1y1x2y2=False)

        intersect = (.1**2)
        iou_expected = intersect/(.2**2+.2**2-intersect)
        self.assertAlmostEqual(iou, iou_expected, places=5)

    def test_bbox_iou_xyxy(self):
        box1 = torch.FloatTensor([.1, .1, .3, .3])
        box2 = torch.FloatTensor([.2, .2, .4, .4])
        iou = utils.bbox_iou(box1, box2, x1y1x2y2=True)

        intersect = (.1**2)
        iou_expected = intersect/(.2**2+.2**2-intersect)
        self.assertAlmostEqual(iou, iou_expected, places=5)

class MultiBboxIouTest(unittest.TestCase):
    def test_multi_bbox_iou_xywh(self):
        box1 = torch.FloatTensor([[.2, .2, .2, .2], [.8,.8,.2,.2]])
        box2 = torch.FloatTensor([[.2, .2, .2, .2], [.7,.7,.2,.2]])

        iou_from_singles_0 = utils.bbox_iou(box1[0], box2[0], x1y1x2y2=False)
        iou_from_singles_1 = utils.bbox_iou(box1[1], box2[1], x1y1x2y2=False)
        
        box1 = box1.t().view(4,2)
        box2 = box2.t().view(4,2)
        iou = utils.multi_bbox_ious(box1, box2, x1y1x2y2=False)
        
        self.assertAlmostEqual(iou[0], iou_from_singles_0, places=5)
        self.assertAlmostEqual(iou[1], iou_from_singles_1, places=5)

    def test_multi_bbox_iou_xyxy(self):
        box1 = torch.FloatTensor([[.1, .1, .2, .2], [.7,.7,.9,.9]])
        box2 = torch.FloatTensor([[.1, .1, .2, .2], [.6,.6,.8,.8]])

        iou_from_singles_0 = utils.bbox_iou(box1[0], box2[0], x1y1x2y2=True)
        iou_from_singles_1 = utils.bbox_iou(box1[1], box2[1], x1y1x2y2=True)
        
        box1 = box1.t().view(4,2)
        box2 = box2.t().view(4,2)
        iou = utils.multi_bbox_ious(box1, box2, x1y1x2y2=True)
        
        self.assertAlmostEqual(iou[0], iou_from_singles_0, places=5)
        self.assertAlmostEqual(iou[1], iou_from_singles_1, places=5)

