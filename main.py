import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision.models import resnet18, resnet50
#from densenet import densenet161
from torch.autograd import Variable
from datasets import ImageDataset
from sklearn.model_selection import train_test_split, KFold
from utils import get_accuracy, get_label
EPOCH = 20
BATCH_SIZE = 128
LR = 1e-4
out_dim = 30
use_all_data = True
import torchvision.transforms as transforms
from PIL import Image
import os.path as osp
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
transforms_train_list = [#transforms.Resize((72,72), interpolation=3), #Image.BICUBIC
                         #transforms.RandomCrop((64, 64)),
                         #transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),]
                         #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
transforms_val_list = [#transforms.Resize((72,72), interpolation=3), #Image.BICUBIC
                      transforms.ToTensor(),]
                      #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
data_transforms = {
    'train': transforms.Compose(transforms_train_list),
    'val': transforms.Compose(transforms_val_list),
    'test':  transforms.Compose(transforms_val_list),
}
root_dir = './raw_data'
train_dir = osp.join(root_dir, 'DatasetA_train_20180813')
test_dir = osp.join(root_dir, 'DatasetA_test_20180813')
use_gpu = torch.cuda.is_available()
print('use_gpu is ', use_gpu)

"""
attribute_list.txt: (30, 2) ->(index, attribute_name)
attributes_per_class: (-1, 31) ->(label, attribute30)
class_embedding: (-1, 301) -> (name, embedding300)
label_list: (230, 2) -> (label, name)
"""

def merge_data_label(data_file, image_path, label_file=None):
    # 获取训练集和测试集的数据
    data = pd.read_csv(data_file, sep='\t', header=None)
    file_list = data[0].values.tolist()
    image_data = np.zeros((len(file_list), 64, 64, 3))
    print('data size is {}'.format(len(image_data.shape)))
    for i, x in enumerate(file_list):
        path = osp.join(image_path, x)
        if osp.isfile(path):
            image = Image.open(path).convert('RGB')
            image_data[i, :, :, :] = np.array(image)

    if label_file is not None:
        print('merge train data....')
        data.columns = ['file_name', 'label']
        columns = pd.read_csv(osp.join(train_dir, 'attribute_list.txt'), header=None, sep='\t')[
            1].values.tolist()
        train_label = pd.read_csv(label_file, sep='\t', header=None)
        train_label.columns = ['label'] + columns
        train_all_data = pd.merge(data, train_label, on='label', how='left')
        np.save('exp/train_class_label.npy', train_all_data[['label']].values)
        train_all_data.iloc[:, -30:].to_pickle('exp/train_y.pkl')
        np.save('exp/train_data.npy', image_data)
    else:
        np.save('exp/test_data.npy', image_data)

def train():
    model = resnet18(pretrained=False)
    #model = densenet161(pretrained=False)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #即使B*C*M*H -> B*C*1*1
    model.fc = nn.Linear(model.fc.in_features, out_dim)
    if use_gpu:
        model = model.cuda()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    lossFunc = nn.MSELoss()  # the target is not one-hotted

    # 读取训练集和测试集
    data_x = np.load('./data/train_data.npy')
    data_class_label = np.load('./data/train_class_label.npy')
    data_y = pd.read_pickle('./data/train_y.pkl').values
    test_x = np.load('./data/test_data.npy')
    if use_all_data:
        train_x, train_y = data_x ,data_y
    else:
        kf = KFold(n_splits=5, random_state=2018).split(data_x)
         for idx, (train_index, val_index) in enumerate(kf):
             train_x, train_y, train_class_label = data_x[train_index], data_y[train_index], data_class_label[train_index]
             val_x, val_y, val_class_label = data_x[val_index], data_y[val_index], data_class_label[val_index]
             break
        train_x, val_x ,train_y, val_y = train_test_split(data_x, data_y, random_state=2018, test_size=0.2)
        #划分训练集和验证集
        print('After split train size is {}, val_size is {}'.format(data_x.shape, val_x.shape))
        
        val_loader = Data.DataLoader(dataset=ImageDataset(val_x,
                                                       labels=val_y,
                                                       transform=data_transforms['val']),
                                     batch_size=BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=15)
#

    
    train_loader = Data.DataLoader(dataset=ImageDataset(train_x,
                                                        labels=train_y,
                                                        transform=data_transforms['train']),
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=15)
    
    test_loader = Data.DataLoader(dataset=ImageDataset(test_x,
                                                       transform=data_transforms['test']),
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=15)

    # 标签和对应属性表
    label_data = pd.read_csv(osp.join(train_dir, 'attributes_per_class.txt'), sep='\t', header=None)

    for epoch in range(EPOCH):
        train_loss = 0.
        # model train
        print('Training...')
        train_predict = np.zeros((train_x.shape[0], out_dim))
        #val_predict = np.zeros((val_x.shape[0], out_dim))
        test_predict = np.zeros((test_x.shape[0], out_dim))
        for step, (batch_x, batch_y) in enumerate(train_loader):
            if use_gpu:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            predict = model(batch_x)
            loss = lossFunc(predict, batch_y)
            train_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            index = step * BATCH_SIZE
            train_predict[index:index + batch_x.size(0)] = predict.cpu().data.numpy()
            if step % 50 == 0:
                print('Train elapsed {} steps'.format(step))

        if not use_all_data:
            #model eval
            model.eval()
            eval_loss = 0.
            print('Validation....')
            for step, (batch_x, batch_y) in enumerate(val_loader):
                if use_gpu:
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                predict = model(batch_x)
                loss = lossFunc(predict, batch_y)
                eval_loss += loss.data[0]
                index = step * BATCH_SIZE
                val_predict[index:index + batch_x.size(0)] = predict.cpu().data.numpy()
                if step % 50 == 0:
                    print('Validation elapsed {} steps!'.format(step))

        #get accuracy of train_set and val_set
        #train_predict_labels = get_label(train_predict, label_data) # 获得距离最小的对应类的名称
        #train_accuracy = get_accuracy(train_predict_labels, train_class_label)
        #val_predict_labels = get_label(val_predict, label_data)  # 获得距离最小的对应类的名称
        #val_accuracy = get_accuracy(val_predict_labels, val_class_label)

        print('-'*50)
        print('Epoch: ', epoch, '|Train Loss: ', train_loss, '|Val Loss: ', eval_loss)
        print('-'*50)

        print('Testing....')
        model.eval()
        for step, batch_x in enumerate(test_loader):
            if use_gpu:
                batch_x = batch_x.cuda()
            predict = model(batch_x)
            index = step*BATCH_SIZE
            test_predict[index:index+batch_x.size(0), :] = predict.cpu().data.numpy()
        
        predict_label = get_label(test_predict, label_data)

        # submit
        result = pd.read_csv(osp.join(test_dir, 'image.txt'), sep='\t', header=None)
        result['label'] = predict_label
        print('num_classes are {}'.format(result['label'].nunique()))
        result.to_csv('./result/resnet18_submit.txt', header=None, index=False, sep='\t')

        torch.save(model.state_dict(), './params/resnet18_params.pkl')

if __name__ ==  '__main__':
    #merge_data_label(osp.join(train_dir, 'train.txt'), osp.join(train_dir, 'train'), osp.join(train_dir, 'attributes_per_class.txt'))
    #merge_data_label(osp.join(test_dir, 'image.txt'), osp.join(test_dir, 'test'))
    train()

