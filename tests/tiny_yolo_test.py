import pytest, torch, unittest, bcolz, pickle
import numpy as np
# from unittest import mock
# from unittest.mock import Mock
# from testfixtures import tempdir
from PIL import Image

from yolov3_pytorch import utils
from yolov3_pytorch.yolo_layer import *
from yolov3_pytorch.tiny_yolo import *

class IntegrationTinyYoloTest(unittest.TestCase):
    
    def test_basic_process(self):
        model = TinyYolov3(num_classes=80, use_wrong_previous_anchors=True)
        model.load_state_dict(torch.load('data/models/tiny_yolo_converted.h5'))
        _ = model.eval() # .cuda()

        sz = 416
        imgfile = "tests/mocks/person.jpg"
        img_org = Image.open(imgfile).convert('RGB')
        img_resized = img_org.resize((sz, sz))
        img_torch = utils.image2torch(img_resized)

        output_all = model(img_torch)
        self.assertEqual(len(output_all), 2, "Basic output should be for two yolo layers")
        self.assertEqual(list(output_all[0].shape), [1, 255, 13, 13], "Basic output shape should be correct")

        all_boxes = model.predict_img(img_torch)[0]
        self.assertTrue(len(all_boxes) > 2, "Should detect something in img")
        self.assertTrue(len(all_boxes) < 20, "Should not detect too much in img")

        nms_boxes = utils.nms(all_boxes, .4)
        persons = [a for a in nms_boxes if a[-1] == 0]
        self.assertEqual(1, len(persons), "Should detect one person in img")


