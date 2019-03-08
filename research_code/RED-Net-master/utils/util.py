# Xi Peng, Feb 2017
import os, sys
import numpy as np
from PIL import Image
from collections import OrderedDict
import torch

class TrainHistory():
    """store statuses from the 1st to current epoch"""
    def __init__(self):
        self.epoch = []
        self.lr = []
        self.loss = []
        self.rmse = []
        self.best_rmse = 1.
        self.is_best = False

    def update(self, epoch, lr, loss, rmse):
        # lr, epoch, loss, rmse (OrderedDict)
        # epoch = OrderedDict([('epoch',1)] )
        # loss = OrderedDict( [('train_loss',0.1),('val_loss',0.2)] )
        self.epoch.append(epoch)
        self.lr.append(lr)
        if len(self.loss)==0 or \
                loss['val_loss_reg'] < 3.*self.loss[-1]['val_loss_reg']:
            self.loss.append(loss)
            self.rmse.append(rmse)
        else:
            self.loss.append( self.loss[-1] )
            self.rmse.append( self.rmse[-1] )

        self.is_best = rmse['val_rmse'] < self.best_rmse
        self.best_rmse = min(rmse['val_rmse'], self.best_rmse)
        print(rmse['val_rmse'])
        print(self.best_rmse)
        print(self.is_best)
        print(self.best_rmse)

    def state_dict(self):
        dest = OrderedDict()
        dest['epoch'] = self.epoch
        dest['lr'] = self.lr
        dest['loss'] = self.loss
        dest['rmse'] = self.rmse
        dest['best_rmse'] = self.best_rmse
        dest['is_best'] = self.is_best
        return dest

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr']
        self.loss = state_dict['loss']
        self.rmse = state_dict['rmse']
        self.best_rmse = state_dict['best_rmse']
        self.is_best = state_dict['is_best']

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_lr(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Epoch:[%d]\tlr:[%f]' % (epoch, lr))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

