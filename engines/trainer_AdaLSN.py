from PIL import Image
import math, os, time
import logging
import sys
from utils import *
from torch.autograd import Variable
import torch

save = 'eval-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


class Trainer(object):
    # init function for class
    def __init__(self, network, optimizer, dataloader, args):
        self.args = args
        self.network = network
        self.optimizer = optimizer
        self.dataloader = dataloader

        if not os.path.exists('weights'):
            os.makedirs('weights')

        self.timeformat = '%Y-%m-%d %H:%M:%S'

    def train(self):
        lossAcc = 0.0
        lossFuse = 0.0
        self.network.eval()  # if backbone has BN layers, freeze them
        dataiter = iter(self.dataloader)
        for _ in range(self.args.resume_iter // self.args.lr_step):
            self.adjustLR()
        self.optimizer.zero_grad()
        for step in range(self.args.resume_iter, self.args.max_step):
            
            for _ in range(self.args.iter_size):
                try:
                    data, target = next(dataiter)
                except StopIteration:
                    dataiter = iter(self.dataloader)
                    data, target = next(dataiter)

                data, target = data.cuda(self.args.gpu_id), target.cuda(self.args.gpu_id)
                data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)

                loss, fuse_loss = self.network(data, target)
                if np.isnan(float(loss.data[0])):
                    raise ValueError('loss is nan while training')
                loss /= self.args.iter_size
                loss.backward()
                lossAcc += loss.data[0]
                lossFuse += fuse_loss.data[0]
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # adjust hed learning rate
            if (step > 0) and (step % self.args.lr_step) == 0:
                self.adjustLR()
                self.showLR()

            # visualize the loss
            if (step + 1) % self.args.disp_interval == 0:
                timestr = time.strftime(self.timeformat, time.localtime())
                logging.info('{} iter={} totloss={:<8.2f} fuseloss={:<8.2f}'.format(
                    timestr, step + 1, lossAcc / self.args.disp_interval,
                             lossFuse / self.args.disp_interval / self.args.iter_size))
                if step < self.args.max_step - 1:
                    lossAcc = 0.0
                    lossFuse = 0.0

            if (step + 1) % self.args.save_interval == 0:
                torch.save(self.network.state_dict(),
                           './Ada_LSN/weights/inception_sklarge/skel_{}.pth'.format(step + 1))
        torch.save(self.network.state_dict(),
                   './Ada_LSN/weights/inception_sklarge/skel_{}.pth'.format(self.args.max_step))
        return lossAcc / self.args.disp_interval

    def adjustLR(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.args.lr_gamma

    def showLR(self):
        for param_group in self.optimizer.param_groups:
            logging.info(param_group['lr'])
        logging.info('')
