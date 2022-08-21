# airbus-ship-detection
Image segmentation for Kaggle`s "Airbus Ship Detection Challenge" dataset.

Dataset contain images and rle masks for them, so the first thing to do is a function, that will convert rle masks into 2d 1-channel masks.

The implementation of YOLO model was taken from https://www.kaggle.com/code/advaitsave/tensorflow-2-nuclei-segmentation-unet/notebook
