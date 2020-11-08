import os
import numpy as np
import torch
import shutil
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import torchvision.transforms as transforms
from torch.autograd import Variable


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    state_dict = torch.load(model_path)
    model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()})


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def plot(out):
    size = 26
    scale_lst = [out]
    pylab.rcParams['figure.figsize'] = size / 2, size / 3
    plt.figure()
    for i, image in enumerate(scale_lst):
        if len(image.shape) == 4:
            image = image.data[0, 0].cpu().numpy().astype(np.float32)
        if len(image.shape) == 3:
            image = image.data[0].cpu().numpy().astype(np.float32)
        if len(image.shape) == 2:
            image = image
        s = plt.subplot(1, 1, i + 1)
        plt.imshow(1 - image, cmap=cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
    plt.tight_layout()
