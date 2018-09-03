import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision.models import resnet18, resnet50
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

EPOCH = 30
BATCH_SIZE = 64
LR = 1e-4
out_dim = 30
import torchvision.transforms as transforms
from PIL import Image
import os.path as osp

TR = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
root_dir = './raw_data'
train_dir = osp.join(root_dir, 'DatasetA_train_20180813')
test_dir = osp.join(root_dir, 'DatasetA_test_20180813')

def merge_data_label(data_file, image_path, label_file=None):
    # 获取训练集和测试集的数据
    data = pd.read_csv(data_file, sep='\t', header=None)
    file_list = data[0].values.tolist()
    image_data = np.zeros((len(file_list), 3, 64, 64))
    print('data size is {}'.format(len(image_data.shape)))
    for i, x in enumerate(file_list):
        path = osp.join(image_path, x)
        if osp.isfile(path):
            image = Image.open(path).convert('RGB')
            image1 = TR(image)
            image_data[i, :, :, :] = image1.data.numpy()

    print('data size is {}'.format(i))
    if label_file is not None:
        print('merge train data....')
        data.columns = ['file_name', 'label']

        columns = pd.read_csv(osp.join(train_dir, 'attribute_list.txt'), header=None, sep='\t')[
            1].values.tolist()
        train_label = pd.read_csv(label_file, sep='\t', header=None)
        train_label.columns = ['label'] + columns
        train_all_data = pd.merge(data, train_label, on='label', how='left')
        train_all_data.iloc[:, -30:].to_pickle('./data/train_y.pkl')
        np.save('./data/train_data.npy', image_data)
    else:
        np.save('./data/test_data.npy', image_data)

def train():
    model = resnet50(pretrained=False)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.fc = nn.Linear(model.fc.in_features, out_dim)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    lossFunc = nn.MSELoss()  # the target is not one-hotted

    # 读取训练集和测试集
    train_x = np.load('./data/train_data.npy')
    train_y = pd.read_pickle('./data/train_y.pkl').values
    test_x = np.load('./data/test_data.npy')
    # 划分训练集和验证集
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=2018)
    print('划分后 train size is {}, test_size is {}'.format(train_x.shape, val_x.shape))

    # 转为tensor
    tensor_train_x = torch.from_numpy(train_x).float()
    tensor_train_y = torch.from_numpy(train_y).float()

    tensor_val_x = torch.from_numpy(val_x).float()
    tensor_val_y = torch.from_numpy(val_y).float()

    tensor_test_x = torch.from_numpy(test_x).float()

    train_data = Data.TensorDataset(tensor_train_x, tensor_train_y)
    val_data = Data.TensorDataset(tensor_val_x, tensor_val_y)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=15)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=15)

    # 标签和对应属性表
    label_data = pd.read_csv(osp.join(train_dir, 'attributes_per_class.txt'), sep='\t', header=None)

    def get_label(predict, data):
        print(data.columns)
        label_index = np.zeros((predict.shape[0],))
        label_name = data[0].values.tolist()
        data30 = data.iloc[:, 1:].values
        print(data30.shape)
        for i in range(predict.shape[0]):
            min_dist = 1000000.
            for j in range(len(label_name)):
                # 比较欧式距离
                cur_dist = np.sqrt(np.sum(np.square(data30[j] - predict[i])))
                if min_dist > cur_dist:
                    print('min_dist:', min_dist, ' |cur_dist: ', cur_dist, ' |label name: ', label_name[j])
                    min_dist = cur_dist
                    label_index[i] = j
        predict_label = [label_name[int(i)] for i in label_index]
        return np.array(predict_label)

    for epoch in range(EPOCH):
        train_loss = 0.
        # model train
        print('Training...')
        for step, (batch_x, batch_y) in enumerate(train_loader):
            predict = model(batch_x)
            loss = lossFunc(predict, batch_y)
            train_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # model eval
        model.eval()
        eval_loss = 0.
        print('Validation....')
        for step, (batch_x, batch_y) in enumerate(val_loader):
            predict = model(batch_x)
            loss = lossFunc(predict, batch_y)
            eval_loss += loss.data[0]

        print('Epoch: ', epoch, '|Train Loss: ', train_loss, '|Val Loss: ', eval_loss)
        print('-'*10)

        print('Testing....')
        model.eval()
        predict = model(tensor_test_x)
        predict_label = get_label(predict.data.numpy(), label_data)
        print(predict_label.shape)

        # submit
        result = pd.read_csv(osp.join(test_dir, 'image.txt'), sep='\t', header=None)
        result['label'] = predict_label
        result.to_csv('resnet18_submit.txt', header=None, index=False, sep='\t')

        torch.save(model.state_dict(), './resnet18_params.pkl')

if __name__ ==  '__main__':
    #merge_data_label(osp.join(train_dir, 'train.txt'), osp.join(train_dir, 'train'), osp.join(train_dir, 'attributes_per_class.txt'))
    #merge_data_label(osp.join(test_dir, 'image.txt'), osp.join(test_dir, 'test'))
    train()

