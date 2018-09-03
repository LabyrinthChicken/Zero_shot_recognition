import os.path as osp

######################################################################
# path
# --------
root = './data'

train_dir = osp.join(root, 'DatasetA_train_20180813')
train_image_dir = osp.join(train_dir, 'train')
train_file_image = osp.join(train_dir, 'train.txt')
train_file_label = osp.join(train_dir, 'label_list.txt')

test_dir = osp.join(root, 'DatasetA_test_20180813', 'DatasetA_test')
test_image_dir = osp.join(test_dir, 'test')
test_file_image = osp.join(test_dir, 'image.txt')


file_attribute = osp.join(train_dir, 'attribute_list.txt')
file_attribute_pc = osp.join(train_dir, 'attributes_per_class.txt')

######################################################################
# Super parameter
# --------

lr = 1e-4
gamma = 0.1
weight_decay = 5e-04
stepsize = 40
seed = 1
batchsize = 256
max_epochs = 1000

######################################################################
# display parameter
# --------
eval_step = 10
print_freq = 10