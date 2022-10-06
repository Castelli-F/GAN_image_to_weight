import tensorflow as tf
import os
import pathlib
import time
import datetime
import imageio

from matplotlib import pyplot as plt
from datagenerator import *


BUFFER_SIZE = 175   # the weight+strawberry dataset consists of 175 images
BATCH_SIZE = 1  # in the paper, a batch_size of 1 produced better results
# each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],     # input image = strawberry image
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],       # real image = weight image
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):   # random cropping
    stacked_image = tf.stack([input_image, real_image], axis=0)
    # randomly crops a tensor to a given size
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    # Resizing to 286x286
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image