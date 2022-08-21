import os

batch_size = 16
val_size = 0.2
IMG_SIZE = 768

train_dir = 'drive/MyDrive/data/train_v2_small/'

dir_len = len(os.listdir(train_dir))
train_len = round(dir_len*val_size)