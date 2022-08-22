from config import IMG_SIZE
import numpy as np
import pandas as pd

import keras.backend as backend
import keras.losses as losses


df = pd.read_csv('data/train_ship_segmentations_v2.csv').dropna()

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


def mask_to_rle(array):
    """
        array: numpy array, 1 - mask, 0 - background
        Returns run length as string formatted
        """
    pixels = array.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


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




