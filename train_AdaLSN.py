import argparse
from torch import optim
from torch.utils.data import DataLoader
from datasets.sklarge_RN import TrainDataset
from engines.trainer_AdaLSN import Trainer, logging
from Ada_LSN.model import Network
from Ada_LSN.utils import *
import os
from Ada_LSN.genotypes import geno_inception as geno

parser = argparse.ArgumentParser(description='TRAIN SKLARGE')
parser.add_argument('--data1', default='./SKLARGE/', type=str)
parser.add_argument('--data2', default='./SKLARGE/train_pairRN60_255_s_all.lst', type=str)
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--INF', default=1e6, type=int)
parser.add_argument('--C', default=64, type=int)  # 32/64/128
parser.add_argument('--each_steps', default=1000, type=int)
parser.add_argument('--numu_layers', default=5, type=int)
parser.add_argument('--u_layers', default=[0, 1, 2, 3], type=list)
parser.add_argument('--lr', default=1e-6, type=float)
parser.add_argument('--lr_step', default=70000, type=int)
parser.add_argument('--lr_gamma', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=0.0002, type=float)
parser.add_argument('--iter_size', default=10, type=int)
parser.add_argument('--max_step', default=80000, type=int)  # train max_step each architecture
parser.add_argument('--disp_interval', default=100, type=int)
parser.add_argument('--resume_iter', default=0, type=int)
parser.add_argument('--save_interval', default=1000, type=int)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train_model(g, dataloader, args):
    try:
        logging.info(g[0])
        logging.info(g[1][0])
        logging.info(g[1][1])
        logging.info(g[1][2])
        logging.info(g[1][3])
        logging.info(g[1][4])
        logging.info(
            '%s %s %s %s %s' % (str(g[2][0]), str(g[2][1]), str(g[2][2]), str(g[2][3]), str(g[2][4])))
        logging.info(g[2][5])
        logging.info(g[2][6])
    except Exception as e:
        pass
    net = Network(args.C, args.numu_layers, args.u_layers, geno,
                  pretrained_model='./pretrained_model/inceptionV3.pth').cuda(args.gpu_id)
    logging.info('params:%.3fM' % count_parameters_in_MB(net))
    lr = args.lr / args.iter_size
    optimizer = optim.Adam(net.parameter(args.lr), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    trainer = Trainer(net, optimizer, dataloader, args)
    loss = trainer.train()


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(args.gpu_id)
    dataset = TrainDataset(args.data2, args.data1)
    dataloader = DataLoader(dataset, shuffle=True)  # batchsize=1

    train_model(geno, dataloader, args)
