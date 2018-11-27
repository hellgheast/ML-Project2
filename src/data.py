import os
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np

IMG_HEIGHT = 400
IMG_WIDTH = 400
IMG_CHANNELS = 3

def load_image_rgb(infilename):
    data = imread(infilename)[:,:,:IMG_CHANNELS]
    return data

def load_image_bw(infilename):
    data = imread(infilename)
    return data

def load_train_set():
    root_dir = "data/training/"

    image_dir = root_dir + "images/"
    files = os.listdir(image_dir)
    n = min(100, len(files)) #load the whole dataset
    print("Loading " + str(n) + " images")
    imgs = [load_image_rgb(image_dir + files[i]) for i in range(n)]
    print(files[0])

    gt_dir = root_dir + "groundtruth/"
    print("Loading " + str(n) + " images")
    gt_imgs = [load_image_bw(gt_dir + files[i]) for i in range(n)]
    print(files[0])

    X_train = np.array(imgs)
    Y_train = np.zeros((len(imgs), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for i in range(len(imgs)):
        Y_train[i] = np.expand_dims(gt_imgs[i], axis=-1)

    return (X_train, Y_train)

def load_test_set():
    test_path = "data/test_set_images/"

    files = os.listdir(test_path)
    print("Loading " + str(len(files)) + " images")
    imgs = [load_image_rgb(test_path + files[i] + "/" + files[i] + ".png") for i in range(len(files))]

    return imgs

def downscale_test_images(imgs):
    X_test = np.zeros((len(imgs), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    for i in range(len(imgs)):
        img = resize(imgs[i], (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[i] = img

    return X_test

def upscale_predictions(imgs):
    predictions = np.zeros((len(imgs), 608, 608, 1), dtype=np.uint8)

    for i in range(50):
        img = resize(imgs[i], (608, 608), mode='constant', preserve_range=True)
        predictions[i] = img

    return predictions.reshape((len(imgs), 608, 608))
    