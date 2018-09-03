# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import time
import argparse
import numpy as np
import json
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from dataset import ImageDataset
import models
import ref
from utils import AverageMeter, load_network, save_network, get_accuracy, mkdir_if_missing, getValue



######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='baseline', type=str, help='output model name')
parser.add_argument('--arch',default='resnet18', type=str, help='which backbone to use')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--pretrain_epoch', default=0, type=int, help='0,1,2,3...or last')
parser.add_argument('--load_epoch',default=None, type=str, help='0,1,2,3...or last')
parser.add_argument('--evaluate', action='store_true', help='if evaluate' )
parser.add_argument('--test', action='store_true', help='if test' )
args = parser.parse_args()

# save argss
mkdir_if_missing(args.name)
with open('%s/args.json'%args.name,'w') as fp:
    json.dump(vars(args), fp, indent=1)

print("Currently using GPU {}".format(args.gpu_ids))
np.random.seed(ref.seed)
torch.manual_seed(ref.seed)
torch.cuda.manual_seed(ref.seed)
torch.cuda.manual_seed_all(ref.seed)
cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
use_gpu = torch.cuda.is_available()

######################################################################
# Load Data
# ---------
#
transform_train_list = [
        transforms.Resize((72,72), interpolation=3), #Image.BICUBIC
        transforms.RandomCrop((64,64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(72,72),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
    'test': transforms.Compose(transform_val_list),
}

train_all = ''
if args.train_all:
     train_all = '_all'

# 读取训练集和测试集
train_x = np.load('train_data.npy')
train_y = pd.read_pickle('train_y.pkl').values
test_x = np.load('test_data.npy')

# 划分训练集和验证集
if train_all == True:
    _, val_x, _, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=2018)
else:
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=2018)

print('划分后:')
print('train size is {}'.format(train_x.shape))
print('val size is {}'.format(val_x.shape))
print('test size is {}'.format(test_x.shape))

data_loader = {}
data_loader['train'] = torch.utils.data.DataLoader(ImageDataset(train_x, labels=train_y, transform=data_transforms['train']), \
                                                   batch_size=ref.batchsize, shuffle=True, num_workers=16, pin_memory=use_gpu)
data_loader['val'] = torch.utils.data.DataLoader(ImageDataset(val_x, labels=val_y, transform=data_transforms['val']), \
                                                   batch_size=ref.batchsize, shuffle=False, num_workers=16, pin_memory=use_gpu)
data_loader['test'] = torch.utils.data.DataLoader(ImageDataset(test_x, transform=data_transforms['test']),\
                                                   batch_size=ref.batchsize, shuffle=False, num_workers=16, pin_memory=use_gpu)



######################################################################
# Training the model
# ------------------
def step(model, data_loader, mode, criterion, optimizer=None):

    if mode == 'train':
        model.train(True)  # Set model to training mode
        is_valatile = False
    elif mode == 'val':
        model.train(False)  # Set model to evaluate mode
        is_valatile = True
    else: raise RuntimeError("'{}' is not approriate".format(mode))

    losses = AverageMeter()
    for batch_idx, data in enumerate(data_loader[mode]):
        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs = Variable(inputs, volatile=is_valatile)
        labels = Variable(labels, volatile=is_valatile)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        if mode == 'train':
            loss.backward()
            optimizer.step()

        losses.update(loss.data[0], inputs.size(0))

        if (batch_idx+1) % ref.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(data_loader[mode]), losses.val, losses.avg))

    return losses

def train(model, data_loader, criterion, optimizer, scheduler, start_epoch=0, max_epochs=100):
    since = time.time()
    for epoch in range(start_epoch, max_epochs, 1):
        print('Epoch {}/{}'.format(epoch, max_epochs - 1))
        print('-' * 10)

        step(model, data_loader, 'train', criterion, optimizer)

        if ref.stepsize > 0: scheduler.step()

        if ref.eval_step > 0 and (epoch + 1) % ref.eval_step == 0 or (epoch + 1) == ref.max_epochs:
            print("==> Eval")
            step(model, data_loader, 'val', criterion, optimizer)
            save_network(model, args.name, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    save_network(model, args.name, 'last')

def test(model, data_loader, file_path):
    model.eval()

    preds_all = np.zeros((test_x.shape[0],30))
    for batch_idx, imgs in enumerate(data_loader):
        # get the inputs

        if use_gpu:
            imgs = Variable(imgs.cuda(), volatile=True)
        else:
            imgs = Variable(imgs, volatile=True)

        preds = model(imgs)
        index = batch_idx*ref.batchsize
        preds_all[index:index+imgs.size(0)] = getValue(preds)

        if (batch_idx+1) % ref.print_freq == 0:
            print("Batch {}/{}".format(batch_idx+1, len(data_loader)))

    label_data = pd.read_csv(ref.file_attribute_pc, sep='\t', header=None)
    predict_label = get_accuracy(preds_all, label_data)
    data = pd.read_csv(ref.test_file_image, sep='\t', header=None)
    file_list = data[0].values.tolist()

    with open(file_path, 'w') as f:
        for name,label in zip(file_list,predict_label):
            line = ' '.join((name,label))
            f.write(line + '\n')


model = models.init_model(name=args.arch, output_dim=30)
print(model)
if use_gpu:
    model = model.cuda()

start_epoch = 0
if args.load_epoch != None:
    model = load_network(model,args.name, args.load_epoch)
    start_epoch = int(args.load_epoch) + 1

if args.test == True:
    test(model, data_loader['test'], file_path= os.path.join(args.name,args.load_epoch+'out.txt'))
    os._exit()

# Decay LR by a factor of 0.1 every 40 epochs
optimizer = torch.optim.Adam(model.parameters(), lr=ref.lr, weight_decay=ref.weight_decay)
if ref.stepsize > 0:
    scheduler = lr_scheduler.StepLR(optimizer, step_size=ref.stepsize, gamma=ref.gamma)

criterion = nn.MSELoss(reduce=True)
train(model, data_loader, criterion, optimizer, scheduler, start_epoch, ref.max_epochs)
test(model, data_loader['test'] , os.path.join(args.name,'last_'+'out.txt'))




