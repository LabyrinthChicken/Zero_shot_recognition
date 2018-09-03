import os
import pandas as pd
import numpy as np
from utils import read_image
import ref

def merge_data_label(data_file, image_path, label_file=None):
    # 获取训练集和测试集的数据
    data = pd.read_csv(data_file, sep='\t', header=None)
    file_list = data[0].values.tolist()
    image_data = np.zeros((len(file_list), 64, 64, 3))
    print('data size is {}'.format(len(image_data.shape)))
    for i, x in enumerate(file_list):
        if (i % 50 == 0):
            print('{}th'.format(i));
        path = os.path.join(image_path, x)
        print('image_path ', path)
        if os.path.isfile(path):
            image = read_image(path)
            image_data[i, :, :, :] =  np.array(image)

    print('data size is {}'.format(i))
    if label_file is not None:
        print('merge train data....')
        data.columns = ['file_name', 'label']

        columns = pd.read_csv(ref.file_attribute, header=None, sep='\t')[
            1].values.tolist()
        train_label = pd.read_csv(label_file, sep='\t', header=None)
        train_label.columns = ['label'] + columns
        train_all_data = pd.merge(data, train_label, on='label', how='left')
        train_all_data.iloc[:, -30:].to_pickle('train_y.pkl')
        np.save('train_data.npy', image_data)
    else:
        np.save('test_data.npy', image_data)

if __name__ == '__main__':
    merge_data_label(ref.train_file_image, ref.train_image_dir, ref.file_attribute_pc)
    merge_data_label(ref.test_file_image, ref.test_image_dir)
