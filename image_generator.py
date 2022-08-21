from bodt import rle_to_mask, df
import numpy as np
import os
from config import train_dir, IMG_SIZE
from PIL import Image

for filename in os.listdir(train_dir):
    masks = list(df.loc[df['ImageId'] == filename, 'EncodedPixels'])

    with Image.open(train_dir+filename) as im:
        zero = np.zeros((IMG_SIZE, IMG_SIZE))
        #pixels = np.array(im)
        pixels = rle_to_mask(zero, masks)
        PIL_image = Image.fromarray(pixels.astype('uint8'), 'L')
        #PIL_image.show()
        PIL_image = PIL_image.save("data/train_v2_labeled/"+filename)




