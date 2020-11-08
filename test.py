import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import time
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy.io as scio
import scipy.misc as mc
import importlib
from datasets.sklarge import TestDataset


network = 'hed'
gpu_id = 0

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.set_device(gpu_id)

Network = getattr(importlib.import_module('networks.' + network), 'Network')
net = Network().cuda(gpu_id).eval()
net.load_state_dict(torch.load('./weights/hed_sklarge/hed_30000.pth', map_location=lambda storage, loc: storage))

root = './OriginalSKLARGE/images/test'
files = './OriginalSKLARGE/test.lst'


dataset = TestDataset(files, root)
dataloader = list(DataLoader(dataset, batch_size=1))


def plot_single_scale(scale_lst, size):
    pylab.rcParams['figure.figsize'] = size / 2, size / 2.5
    plt.figure()
    for i, image in enumerate(scale_lst):
        image = image.data[0, 0].cpu().numpy().astype(np.float32)
        s = plt.subplot(1, 1, i + 1)
        plt.imshow(1 - image, cmap=cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()


idx = 16

inp, fname = dataloader[idx]
inp = Variable(inp.cuda(gpu_id))
out = net(inp)
scale_lst = [out]
plot_single_scale(scale_lst, 22)
plt.show()


output_dir = 'outputs/hed/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
start_time = time.time()
tep = 1
for inp, fname in dataloader:
    inp = Variable(inp.cuda(gpu_id))
    out = net(inp)
    fileName = output_dir + fname[0] + '.mat'
    # file_jpg = output_dir + fname[0] + '.jpg'
    tep += 1
    scio.savemat(fileName, {'sym': out.data[0, 0].cpu().numpy()})
    # mc.toimage(out.cpu().detach()[0, 0, :, :]).save(file_jpg)
diff_time = time.time() - start_time
print('Detection took {:.5f}s per image'.format(diff_time / len(dataloader)))


