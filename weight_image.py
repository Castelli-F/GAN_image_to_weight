import tensorflow as tf
import numpy as np
import json

# from matplotlib import pyplot as plt
from PIL import Image

# w is the weight vector and s is the strawberry image
image_number = 175
for i in range(image_number):
    with open("/home/francesco/Desktop/weights/" + str(i) + ".json") as f:
        data_w = json.load(f)
    w = data_w['mean_weights']
    w = np.array(w)
    print(w)
    w = tf.convert_to_tensor(w, dtype='uint8')
    w = tf.transpose(w)
    w = tf.tile(w, [256, 32])
    w = tf.expand_dims(w, axis=2)
    w = tf.repeat(w, 3, axis=2)
    print(tf.shape(w))
    # plt.figure()
    # plt.imshow(w)
    # plt.show()
    f.close()
    s = Image.open("/home/francesco/Desktop/rgb_tot_white_gan/" + str(i) + "_image.png")
    # plt.figure()
    # plt.imshow(s)
    # plt.show()
    size = (256, 256)
    s = s.resize(size)    # s stands for strawberry
    s = np.array(s)       # conversion to array
    s = tf.convert_to_tensor(s, dtype='uint8')    # conversion to tensor
    print(s.shape)
    united_w_s = tf.concat([w, s], 1)   # concatenate the two tensors
    united_w_s = np.array(united_w_s)   # convert the united tensor into an array (image)
    print(united_w_s.shape)
    # plt.figure()
    # plt.imshow(united_w_s)
    # plt.show()

    # saving the image on a folder in the desktop
    tf.keras.preprocessing.image.save_img("/home/francesco/PycharmProjects/dataset/weight_strawberry/"
                                          + str(i) + "_weight_and_strawberry.png", united_w_s)
