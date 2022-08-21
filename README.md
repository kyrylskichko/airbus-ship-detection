# airbus-ship-detection
Image segmentation for Kaggle`s "Airbus Ship Detection Challenge" dataset.

Dataset contain images and rle masks for them, so the first thing to do is a function, that will convert rle masks into 2d 1-channel masks.

Model was trained for part of dataset(~12000 img) because of lack of computational power, the accuracy for validation datatset is 98%.

The implementation of YOLO model was taken from https://www.kaggle.com/code/advaitsave/tensorflow-2-nuclei-segmentation-unet/notebook
