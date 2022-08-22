from model import model
import os
from config import IMG_SIZE
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
from utils import mask_to_rle
import pandas as pd

model.load_weights("model_save.hdf5")

test_datagen = ImageDataGenerator()

test_image_generator = test_datagen.flow_from_directory(
    'data/test_v2/',
    class_mode=None,
    batch_size=1,
    target_size=(IMG_SIZE, IMG_SIZE))

df = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])

k=0

for filename in os.listdir('data/test_v2/data/'):
    with Image.open('data/test_v2/data/' + filename) as im:
        pixels = np.array(im)

    pixels = pixels.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    prediction = model.predict(pixels)

    prediction = prediction.reshape(IMG_SIZE, IMG_SIZE)
    prediction = np.round(prediction)

    rle = mask_to_rle(prediction)

    df = df.append({"ImageId": filename, "EncodedPixels": rle}, ignore_index=True)

df.to_csv('sample_submission_v2.csv', index=False)




