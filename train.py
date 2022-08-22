import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import keras.losses as losses
from config import batch_size, IMG_SIZE, epochs, steps_per_epoch
from model import model

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer,
              loss = losses.BinaryCrossentropy(),
              metrics=['accuracy'])

image_datagen = ImageDataGenerator(validation_split=0.2)
mask_datagen = ImageDataGenerator(validation_split=0.2)

seed = 1

train_image_generator = image_datagen.flow_from_directory(
    'data/train_v2_small/',
    class_mode=None,
    batch_size=batch_size,
    target_size=(IMG_SIZE, IMG_SIZE),
    seed=seed,
    subset='training')
train_mask_generator = mask_datagen.flow_from_directory(
    'data/train_v2_l_small/',
    class_mode=None,
    batch_size=batch_size,
    target_size=(IMG_SIZE, IMG_SIZE),
    seed=seed,
    subset='training')

test_image_generator = image_datagen.flow_from_directory(
    'data/train_v2_small/',
    class_mode=None,
    batch_size=batch_size,
    target_size=(IMG_SIZE, IMG_SIZE),
    seed=seed,
    subset='validation')
test_mask_generator = mask_datagen.flow_from_directory(
    'data/train_v2_l_small/',
    class_mode=None,
    batch_size=batch_size,
    target_size=(IMG_SIZE, IMG_SIZE),
    seed=seed,
    subset='validation')

train_generator = zip(train_image_generator, train_mask_generator)
test_generator = zip(test_image_generator, test_mask_generator)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[es],
    validation_data = test_generator)







