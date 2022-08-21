from config import IMG_SIZE, train_dir
import os
import numpy as np
from cv2 import imread
import pandas as pd

import keras.backend as backend
import keras.losses as losses


df = pd.read_csv('drive/MyDrive/data/train_ship_segmentations_v2.csv').dropna()

def rle_to_mask(array, rle):
    '''

    :param array: single image
    :param rle: list of rle masks, for every image
    :return: image, with applied masks
    '''
    if rle == False:
        return array

    for elem in rle:
        list0 = list(map(int, elem.split(" ")))
        elem = list(zip(list0[0::2], list0[1::2]))

        place = []
        for i in elem:
            start, length = i
            coordinate = (start % IMG_SIZE, start // IMG_SIZE, length)
            place.append(coordinate)

        for x, y, l in place:
            for i in range(0, l):
                array[x-1+i, y-1] = 255
    return array


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def train_data(batch_size, train_len):
    X_train = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    Y_train = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)

    n = 0

    for filename in os.listdir(train_dir)[:train_len]:
        img = imread(train_dir + "/" + filename)[:, :, :3]
        masks = list(df.loc[df['ImageId'] == filename, 'EncodedPixels'])
        X_train[n] = img
        Y_train[n] = rle_to_mask(np.zeros((768, 768, 1)), masks)
        n += 1

        if batch_size % n == 0:
            n = 1
            yield X_train, Y_train


def val_data(batch_size, train_len):
    X_train = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    Y_train = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)

    n = 0

    for filename in os.listdir(train_dir)[train_len:]:
        img = imread(train_dir + "/" +  filename)[:, :, :3]
        masks = list(df.loc[df['ImageId'] == filename, 'EncodedPixels'])
        X_train[n] = img
        Y_train[n] = rle_to_mask(np.zeros((768, 768, 1)), masks)
        n += 1

        if batch_size % n == 0:
            n = 1
            yield X_train, Y_train



