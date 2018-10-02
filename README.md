# Yolov3_pytorch

Yolov3 (+tiny) object detection - object oriented pythonic pytorch implementation.

Tested with pytorch 0.4.0 and python>3.5. Some basic tests are included in [tests](https://github.com/holli/yolov3_pytorch/tree/master/test) folder.

This repo has a goal to have simple pythonic object oriented implementation that can be easily used as it is and also easy to modify the model.

See https://pjreddie.com/darknet/yolo/ for better explanation of how yolov3 object detection system differs from others.

# Pretrained Weights

Pretrained weights are available at: **http://www.ollihuotari.com/data/yolov3_pytorch/** . They are converted from https://pjreddie.com/darknet/yolo/. Check out the notebooks for examples how to use them.

# Notebook Examples

- **https://github.com/holli/yolov3_pytorch/blob/master/notebooks/basic_prediction.ipynb**
  - show's basic loading of model and prediction
- **https://github.com/holli/yolov3_pytorch/blob/master/notebooks/eval_coco_map.ipynb**
  - map metric on coco evaluation data set. Just to make sure that this implementation is close enough to original implementation

# Support / Commits

Submit suggestions or feature requests as a GitHub Issue or Pull Request. Preferably create a test to show whats happening and what should happen.

# Other Implementations

There are some good pytorch implementations previously but many of them were using original cfg files to create the model. This works well but it's harder to modify and test other approaches. Some of them didn't include yolov3-tiny model or didn't work with using images of different sizes (e.g. 608 pixel sizes instead of default 416). Some nicer ones include:

- https://github.com/marvis/pytorch-yolo3
- https://github.com/andy-yun/pytorch-0.4-yolov3
- https://github.com/jiasenlu/YOLOv3.pytorch

# Licence

Released under the MIT license (http://www.opensource.org/licenses/mit-license.php)



