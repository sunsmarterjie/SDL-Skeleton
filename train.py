import argparse
import importlib
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from datasets.sklarge import TrainDataset
# from datasets.sklarge_flux import DataLayer as TrainDataset  # for deep_flux
from engines.trainer import Trainer


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6


def parse_args():
    parser = argparse.ArgumentParser(description='TRAIN SKLARGE')
    parser.add_argument('--root', default='./SKLARGE', type=str)
    parser.add_argument('--files', default='./SKLARGE/aug_data/train_pair.lst', type=str)
    parser.add_argument('--network', default='hed', type=str)
    parser.add_argument('--pretrained_model', default='pretrained_model/vgg16_caffe.pth', type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--lr_step', default=20000, type=int)
    parser.add_argument('--lr_gamma', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0.0002, type=float)
    parser.add_argument('--iter_size', default=10, type=int)
    parser.add_argument('--max_step', default=25000, type=int)
    parser.add_argument('--save_interval', default=5000, type=int)
    parser.add_argument('--disp_interval', default=50, type=int)
    parser.add_argument('--resume_iter', default=0, type=int)
    parser.add_argument('--resume_path', default=None, type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(args.gpu_id)
    dataset = TrainDataset(args.files, args.root)
    dataloader = DataLoader(dataset, shuffle=True)  # batchsize=1

    Network = getattr(importlib.import_module('networks.' + args.network), 'Network')

    if args.resume_path is None:
        net = Network(args.pretrained_model).cuda(args.gpu_id)
        args.resume_iter = 0
    else:
        net = Network().cuda(args.gpu_id)
        resume = torch.load(args.resume_path, map_location=lambda storage, loc: storage)
        net.load_state_dict(resume)
    print('params:%.3fM' % count_parameters_in_MB(net))
    lr = args.lr / args.iter_size
    optimizer = optim.SGD(net.parameters(args.lr), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(net.parameters(args.lr), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    #  for deep_flux
    trainer = Trainer(net, optimizer, dataloader, args)
    trainer.train()
