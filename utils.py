import pandas as pd
import numpy as np

def get_label(y_pred, y_true):
    label_index = np.zeros((y_pred.shape[0],))
    label_name = y_true[0].values.tolist()
    data30 = y_true.iloc[:, 1:].values
    # data30 [-1, 30] y_pred
    for i in range(y_pred.shape[0]):
        pre_m = np.array(y_pred[i])
        label_m = np.array(data30)
        dis_m = np.sqrt(np.sum(np.power(label_m-pre_m, 2), axis=1))
        dis_list = dis_m.tolist()
        min_index = dis_list.index(min(dis_list))
        label_index[i] = min_index
        # for j in range(len(label_name)):
        #     # 比较欧式距离
        #     cur_dist = np.sqrt(np.sum(np.square(data30[j] - y_pred[i])))
        #     if min_dist > cur_dist:
        #         min_dist = cur_dist
        #         label_index[i] = j
    predict_label = [label_name[int(i)] for i in label_index]
    return np.array(predict_label)

def get_accuracy(y_pred, y_true):
    """
    :param y_pred: list of str
    :param y_true: list of str
    :return: accuracy
    """
    count = 0
    all_num = y_pred.shape[0]
    for m,n in zip(y_pred, y_true):
        if m == n:
            count += 1
        else:
            pass
    return count*1.0/all_num

