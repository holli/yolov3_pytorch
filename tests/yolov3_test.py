import pytest, torch, unittest, bcolz, pickle
import numpy as np
# from unittest import mock
# from unittest.mock import Mock
# from testfixtures import tempdir
from PIL import Image

from yolov3_pytorch import utils
# from yolov3_pytorch.yolo_layer import *
from yolov3_pytorch.yolov3 import *

# import pdb; pdb.set_trace()

class IntegrationYolov3Test(unittest.TestCase):
    
    def test_basic_process(self):
        model = Yolov3(num_classes=80)
        model.load_state_dict(torch.load('data/models/yolov3_coco_01.h5'))
        _ = model.eval() # .cuda()

        sz = 416
        imgfile = "tests/mocks/person.jpg"
        img_org = Image.open(imgfile).convert('RGB')
        img_resized = img_org.resize((sz, sz))
        img_torch = utils.image2torch(img_resized)

        output_all = model(img_torch)
        self.assertEqual(len(output_all), 3, "Basic output should be for 3 yolo layers")
        self.assertEqual(list(output_all[0].shape), [1, 255, 13, 13], "Basic output shape should be correct")

        all_boxes = model.predict_img(img_torch)[0]
        self.assertTrue(len(all_boxes) > 2, "Should detect something in img")
        self.assertTrue(len(all_boxes) < 20, "Should not detect too much in img")

        nms_boxes = utils.nms(all_boxes, .4)
        persons = [a for a in nms_boxes if a[-1] == 0]
        
        self.assertEqual(1, len(persons), "Should detect one person in img")

        # If there is a little difference it might not be a bug, but its here to alert.
        # You can comment away tests that are failing if things are otherwise correct
        self.assertEqual(13, len(all_boxes), "Something has changed in predictions")
        previous_person = [0.36519092321395874, 0.5594505071640015, 0.13189448416233063, 0.6678988933563232, 0.9999792575836182, 1.0, 0]
        self.assertEqual(previous_person, persons[0], "Something has changed in predictions")


