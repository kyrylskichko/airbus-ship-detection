import os

batch_size = 8
val_size = 0.2
IMG_SIZE = 768

steps_per_epoch=20
epochs=100

train_dir = 'data/train_v2_small/'
test_dir = 'data/test_v2/'

dir_len = len(os.listdir(train_dir))
train_len = round(dir_len*val_size)