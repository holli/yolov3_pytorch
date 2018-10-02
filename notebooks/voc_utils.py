import torch.nn as nn
from fastai.imports import *
from fastai.dataset import *
from matplotlib import patches, patheffects

def get_voc_md(data_filenames, sz=416, data_max_lines=False, tfms_trn=None, tfms_val=None):
    if not tfms_trn:
        #tfms_trn = [RandomRotate(10, tfm_y=TfmType.COORD), RandomLighting(0.20, 0.20), RandomBlur()]
        #tfms_trn = [RandomLighting(0.20, 0.20), RandomBlur()]
        tfms_trn = [RandomBlur()]
        # tfms_trn = image_gen(normalizer=None, denorm=None, sz=sz, crop_type=CropType.RANDOM,
        tfms_trn = image_gen(normalizer=None, denorm=None, sz=sz, crop_type=CropType.NO,
                        max_zoom=1.2, tfm_y=TfmType.COORD, tfms=tfms_trn)
    if not tfms_val:
        tfms_val = image_gen(normalizer=None, denorm=None, sz=sz, crop_type=CropType.NO,
        # tfms_val = image_gen(normalizer=None, denorm=None, sz=sz, crop_type=CropType.CENTER,
                        max_zoom=1,   tfm_y=TfmType.COORD, tfms=[])

    data_lines = []
    for f in data_filenames:
        with open(f, 'r') as file:
            arr = file.readlines()
        arr = [s.rstrip('\n') for s in arr]
        data_lines.append(arr)
        
    if data_max_lines:
        if type(data_max_lines) == int:
            data_max_lines = [data_max_lines, data_max_lines]
        for i in range(len(data_lines)):
            data_lines[i] = data_lines[i][:data_max_lines[i]]
            len(data_lines[i])

    datasets = [
        VocDataset(data_lines[0], transform=tfms_trn, path='', sz=sz), # train
        VocDataset(data_lines[1], transform=tfms_val, path='', sz=sz), # valid
        VocDataset(data_lines[0], transform=tfms_val, path='', sz=sz), # fix
        VocDataset(data_lines[1], transform=tfms_trn, path='', sz=sz), # aug
        None, None # test datasets
    ]

    md = ImageData(path = "/tmp", datasets=datasets, bs=32, num_workers=2, classes=VocDataset.CLASS_NAMES)
    md.trn_dl.pre_pad = md.val_dl.pre_pad = md.fix_dl.pre_pad = md.aug_dl.pre_pad = False

    return md


# To be used for example with https://github.com/rafaelpadilla/Object-Detection-Metrics
# python pascalvoc.py --gtfolder /tmp/ai_mAP_1/ground --detfolder /tmp/ai_mAP_1/pred -gtcoords rel -detcoords rel -imgsize 416,416 --noplot
def create_detection_files(validation_ds, tmp_dir='/tmp/ai_mAP_1', remove_old=True):
    for p in ['pred', 'ground']:
        p = os.path.join(tmp_dir, p)
        if os.path.exists(p):
            if remove_old:
                for f in glob.glob(os.path.join(p, "*.txt")):
                    os.remove(f)
        else:
            os.makedirs(p)
    
    for i in range(len(md.val_ds.fnames)):
        imgfile = md.val_ds.fnames[i]
        img_org = Image.open(imgfile).convert('RGB')
        img_resized = img_org.resize((sz, sz))
        img_torch = image2torch(img_resized).cuda()
        all_boxes = model.predict_img(img_torch)[0]
        boxes = nms(all_boxes, 0.4)

        fname = os.path.split(imgfile)[-1]
        fname = fname.replace('.png','.txt').replace('.jpg','.txt')
        det_fname = os.path.join(tmp_dir, 'pred', fname)
        with open(det_fname, 'w') as f:
            for box in boxes:
                box = np.array([b.item() for b in box])
                box[:2] -= box[2:4]/2
                arr = [int(box[-1]), box[-2]] + list(box[0:4])
                s = ' '.join([str(a) for a in arr]) + '\n'
                _ = f.write(s)

        g_fname = os.path.join(tmp_dir, 'ground', fname)
        with open(g_fname, 'w') as f:
            for box in md.val_ds.get_y(i):
                box = np.array(box)
                box[1:3] -= box[-2:]/2
                arr = [int(box[0])] + list(box[1:5])
                s = ' '.join([str(a) for a in arr]) + '\n'
                _ = f.write(s)



# class VocDataset(Dataset):
# /home/ohu/koodi/data/voc/VOCdevkit/VOC2007/JPEGImages/000012.jpg
# /home/ohu/koodi/data/voc/VOCdevkit/VOC2007/labels/000012.txt
# Parsing from https://pjreddie.com/media/files/voc_label.py
class VocDataset(FilesDataset):
    CLASS_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    
    def __init__(self, fnames, transform, path, sz):
        super().__init__(fnames, transform, path)
        self.sz = sz

    # Data is in center_x, center_y, width, height
    # VocDataset.read_labels('/home/ohu/koodi/data/voc/VOCdevkit/VOC2007/labels/000009.txt', 0.03)
    @staticmethod
    def read_labels(lab_path, min_box_scale=0.03):
        if os.path.exists(lab_path) and os.path.getsize(lab_path):
            all_truths = np.loadtxt(lab_path)
            all_truths = all_truths.reshape(all_truths.size//5, 5) # to avoid single truth problem
        else:
            all_truths = np.array([])
        
        truths = []
        for t in all_truths:
            if t[3] < min_box_scale or t[4] < min_box_scale:
                continue
            #truths.append([all_truths[i][0], all_truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
            truths.append(t)
        return np.array(truths)

    
    def get_y(self, i):
        path = os.path.join(self.path, self.fnames[i])
        path = path.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
        # print(path)
        arr = self.read_labels(path, 0.03)
        return arr
            
    def get_c(self):
        return 20 # class numbers gmm?
    
    def get(self, tfm, x, y): # override so that tfm only handels part of the thingie
#         return (x,y) if tfm is None else tfm(x,y)
        w,h = x.shape[0], x.shape[1]
        #return x, y
    
        y1 = y[:, 0:1]
        y2 = y[:, 1:]
        y2[:, :2] -= y2[:, 2:]/2 # x1, y1, w, h
        y2[:, 2:] += y2[:, :2]   # x1, y1, x2, y2
        y2[:, :] *= [h, w, h, w] # pixels
        #y2 *= model.width
        
        # swap y,x to x,y
        y2[:, 0], y2[:, 1] = y2[:, 1].copy(), y2[:, 0].copy()
        y2[:, 2], y2[:, 3] = y2[:, 3].copy(), y2[:, 2].copy()
        
        y2 = y2.reshape(-1)
        
        x, y2 = tfm(x,y2)
        
        y2 = y2.reshape(-1, 4)
                        
        y2[:, 2:] -= y2[:, :2] 
        y2[:, :2] += y2[:, 2:]/2
        
        # y2 /= model.width
        y2 /= self.sz
        # swap y,x to x,y
        y2[:, 1], y2[:, 0] = y2[:, 0].copy(), y2[:, 1].copy()
        y2[:, 3], y2[:, 2] = y2[:, 2].copy(), y2[:, 3].copy()
        
        y = np.concatenate((y1, y2), axis=1)[:50] # max 50 items
        y = y[(y[:, 3] > 0.001) & (y[:, 4] > 0.001)]
        
        if y.shape[0] < 50:
            y_pad = np.zeros((50-y.shape[0], 5))
            y = np.concatenate((y, y_pad), 0)
        
        y = y.reshape(-1)
        return x, y



