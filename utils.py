import torch
import os
import errno
import os.path as osp
from PIL import Image
import shutil
import numpy as np

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

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


def save_network(network, name, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = osp.join(name, save_filename)
    torch.save(network.state_dict(), save_path)

def load_network(network, name, epoch_label):
    save_path = osp.join(name,'net_%s.pth' % epoch_label)
    network.load_state_dict(torch.load(save_path, map_location=lambda storage, loc: storage))
    return network

def adjust_learning_rate(optimizer, epoch, dropLR, LR):
    lr = LR * (0.1 ** (epoch // dropLR))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_accuracy(predict, data):
    print(data.columns)
    label_index = np.zeros((predict.shape[0], ))
    label_name = data[0].values.tolist()
    data30 = data.iloc[:, 1:].values
    print(data30.shape)
    for i in range(predict.shape[0]):
        min_dist = 1000000.
        for j in range(len(label_name)):
            # 比较欧式距离
            cur_dist = np.sqrt(np.sum(np.square(data30[j] - predict[i])))
            if min_dist>cur_dist:
                print('min_dist:', min_dist, ' |cur_dist: ', cur_dist, ' |label name: ', label_name[j])
                min_dist = cur_dist
                label_index[i] = j
    predict_label = [label_name[int(i)] for i in label_index]
    return np.array(predict_label)

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def getValue(x):
    '''Convert Torch tensor/variable to numpy array/python scalar
    '''
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    elif isinstance(x, torch.autograd.Variable):
        x = x.data.cpu().numpy()
    if x.size == 1:
        x = x.item()
    return x
