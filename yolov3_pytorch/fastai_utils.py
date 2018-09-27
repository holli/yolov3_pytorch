import fastai
from fastai.imports import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
import collections


def learn_sched_plot(learn):
    type(learn.sched)
    fig, axs = plt.subplots(1,2,figsize=(12,4))
    plt.sca(axs[0])
    learn.sched.plot_loss(0, 1)
    plt.sca(axs[1])
    learn.sched.plot_lr()


class MultiArraysDataset(BaseDataset):
    def __init__(self, y, xs, sz=None):
        self.xs,self.y = xs,y
        self.sz = sz
        assert(all([len(y)==len(x) for x in xs]))
        
        #super().__init__(None)
        self.n = self.get_n()
        self.sz = self.get_sz()
        self.transform = None
        
    def get1item(self, idx):
        x,y = self.get_x(idx), self.get_y(idx)
        return (*x, y)
    
    def __getitem__(self, idx):
        return self.get1item(idx)
    
    def get_x(self, i): return [x[i] for x in self.xs]
    def get_y(self, i): return self.y[i]
    def get_n(self): return len(self.y)
    def get_sz(self): return self.sz


class YoloLearner(Learner):
    def __init__(self, data, model, **kwargs):
        self.precompute = False
        super().__init__(data=data, models=None, tmp_name='/tmp/yolo_tmp/', **kwargs)
        self._model = model
        self.name = 'yolo_test_1'

    @property
    def model(self):
        if self.precompute:
            self._model.skip_backbone = True
        else:
            self._model.skip_backbone = False
        return self._model
    
    @property
    def data(self): return self.backbone_data if self.precompute else self.data_
    
    def get_layer_groups(self):
        modules = list(self.model.children())
        groups = list(split_by_idxs(modules, [1]))
        return groups
    
    def set_precomputed(self, force_predict=False):
        self.save_backbone_data(force_predict=force_predict)
        self.precompute = True
    
    @staticmethod
    def create_empty_bcolz(n, rootdir):
        if not os.path.exists(rootdir): os.makedirs(rootdir)
        return bcolz.carray(np.zeros(n, np.float32), chunklen=1, mode='w', rootdir=rootdir)
    
    @staticmethod
    def predict_backbone_to_bcolz(m, gen, arrs, workers=4):
        device = next(m.parameters()).device
        #arrs.trim(len(arr))
        lock=threading.Lock()
        m.eval()
        for x_org, target in tqdm(gen):
            x_acts = m(x_org.to(device))
            with lock:
                arrs[0].append(target)
                for i, x in enumerate(x_acts):
                    arrs[i+1].append(to_np(x))
                [a.flush() for a in arrs]
    
    def save_backbone_data(self, force_predict=False):
        tmpl = f'_{self.name}_{self.data_.sz}.bc'
        activations = []
        
        # Checking what shape of activations backbone outputs
        sample_batch = next(iter(self.data_.trn_dl))
        data_shapes = [(sample_batch[1][0].size(0),)]
        for out in self._model.backbone(sample_batch[0][0:1].to(next(self._model.backbone.parameters()).device)):
            data_shapes.append(out[0].detach().cpu().numpy().shape)
        
        # Creating or loading bcolz files
        dls = (self.data_.fix_dl, self.data_.val_dl)
        paths = ('x_act_trn', 'x_act_val')
        for i in range(len(dls)):
            name = os.path.join(self.tmp_path, paths[i]+tmpl)
            shapes_r = range(len(data_shapes))
            
            if os.path.exists(os.path.join(name,str(1))) and not force_predict:
                acts = [bcolz.open(os.path.join(name,str(i))) for i in shapes_r]
            else:
                acts = [self.create_empty_bcolz((0,*data_shapes[i]),os.path.join(name,str(i))) for i in shapes_r]
                    
            activations.append(acts)
        
        # Validate or predict activations
        m = self._model.backbone
        for acts, dl in zip(activations, dls):
            if any([len(a)>0 for a in acts]):
                # compare loaded activations to expected
                for a, s in zip(acts, data_shapes):
                    exp_shape = (len(dl.dataset), *s)
                    if a.shape != exp_shape:
                        raise TypeError(f"Previous backbone activations won't match. You might want to call learn.save_backbone_data(force_predict=True) to erase previous or change the name of the model. {a.shape} != {exp_shape}: {a.rootdir}")
            else:
                self.predict_backbone_to_bcolz(m, dl, acts)
        
        ds_trn = MultiArraysDataset(activations[0][0], activations[0][1:], sz=self.data_.fix_dl.dataset.sz)
        ds_val = MultiArraysDataset(activations[1][0], activations[1][1:], sz=self.data_.fix_dl.dataset.sz)
        dl_trn = DataLoader(ds_trn, batch_size=self.data_.bs, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=self.data_.bs, shuffle=False)
        self.backbone_data = ModelData(self.data_.path, dl_trn, dl_val)
        

class YoloLoss():
    def __init__(self, model, max_history=10000, model_reset_overwrite=True):
        self.model, self.max_history = model, max_history
        self.reset()
        if model_reset_overwrite:
            model.reset = self.reset # fastai call's this before validation run
        
    def __call__(self, output, target):
        losses_all = []
        total_losses = []
        
        for i, layer in enumerate(self.model.get_loss_layers()):
            losses = layer.get_loss(output[i], target, return_single_value=False)
            total_losses.append(losses[0])            
            losses_all.append([l.item() for l in losses])

        if self.max_history:
            self.history.append(losses_all)
        
        return sum(total_losses)
    
    def reset(self):
        if self.max_history:
            self.history = collections.deque(maxlen = self.max_history)
        else:
            self.history = None


class YoloLossMetrics():
    def __init__(self, yolo_loss):
        self.yolo_loss = yolo_loss
        self.set_n_layers()
        
    def set_n_layers(self):
        self.n_layers = len(self.yolo_loss.model.get_loss_layers())
    
    def layer_losses(self):
        arr = []
        for i in range(self.n_layers):
            l = lambda a=0,b=0,i=i: sum([h[i][0] for h in self.yolo_loss.history])/len(self.yolo_loss.history)
            l.__name__ = f"yolo_l_{i}"
            arr.append(l)
        return arr
    
    def individual_losses(self):
        arr = []
        for i in range(1,4):
            l = lambda a=0,b=0,i=i: sum([sum([h[i] for h in his]) for his in self.yolo_loss.history])/(len(self.yolo_loss.history))
            l.__name__ = ['total_loss', 'loss_coord', 'loss_conf', 'loss_cls'][i]
            arr.append(l)
        return arr
