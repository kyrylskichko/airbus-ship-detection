import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.losses as losses
from config import batch_size, IMG_SIZE, epochs, steps_per_epoch
from model import model

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer,
              loss = losses.BinaryCrossentropy(),
              metrics=['accuracy'])

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, gama=2):
        super(DiceLoss, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(
            y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * \
            tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(
            y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result

loss_f = DiceLoss()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07)

model.compile(optimizer=optimizer,
              loss = [loss_f],
              metrics=["accuracy"])

image_datagen = ImageDataGenerator()
mask_datagen = ImageDataGenerator()

seed = 1

train_image_generator = image_datagen.flow_from_directory(
    'drive/MyDrive/data/train_v2_small/',
    class_mode=None,
    batch_size=batch_size,
    target_size=(IMG_SIZE, IMG_SIZE),
    subset='training',
    seed=seed)
train_mask_generator = mask_datagen.flow_from_directory(
    'drive/MyDrive/data/train_v2_l_small/',
    class_mode=None,
    batch_size=batch_size,
    target_size=(IMG_SIZE, IMG_SIZE),
    subset='training',
    seed=seed)

train_generator = zip(train_image_generator, train_mask_generator)

es = ModelCheckpoint("last_model", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    callbacks=[es],
    epochs=epochs)






