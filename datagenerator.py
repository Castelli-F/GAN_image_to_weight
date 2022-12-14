# import tensorflow
import numpy as np
from skimage.io import imread
from skimage.transform import resize
# from glob import glob
from natsort import natsorted
import os
from abc import ABC, abstractmethod


def get_number_id(filename):
    """
    Gets the sample number
    """

    if filename.split(sep=".")[-1] == 'png':
        return filename.split(sep='/')[-1].strip('.png').split(sep='_')[3]


def get_number_ids(data_dir):
    """
    Gets a list of sample numbers
    """
    out = np.array([get_number_id(filename) for filename in natsorted(os.listdir(data_dir))])
    out = [i for i in out if i != None]
    out = np.asarray(out)
    return out


def get_id(filename):
    """
    Gets  the id of a .json file
    """
    if filename.split(sep=".")[-1] == 'png':
        return filename.split(sep='/')[-1].strip('.png').split(sep='_')[0]


def get_ids(data_dir):
    """
    Gets a list of ids of the .json files
    """
    out = np.array(natsorted([get_id(filename) for filename in os.listdir(data_dir)]))
    out = [i for i in out if i != None]
    out = np.asarray(out)
    return out


def load_image(img_path, resize_shape=(256, 256, 3)):   # in Alessandra's: resize_shape=(128, 128, 3)
    """
    Loads the original RGB image
    """
    img = imread(img_path, pilmode='RGB')
    if resize_shape:
        img = resize(img, resize_shape)
    return img.astype('float32')


def load_images(ids, img_path, resize_shape=(256, 256, 3)):     # in Alessandra's: resize_shape=(128, 128, 3)
    """
    Loads a set of RGB images (the ones listed in images ids)
    """
    return np.array([load_image(os.path.join(img_path, id + ".png"), resize_shape) for id in ids])


def get_images_id(filename):
    """
    Gets the single id of an image.
    """
    if filename.split(sep=".")[-1] == 'png':
        return filename.split(sep='/')[-1].strip('.png')


def get_images_ids(data_dir):
    """
    Gets the list of ids of images in a directory
    """
    out = np.array(natsorted([get_images_id(filename) for filename in os.listdir(data_dir)]))
    out = [i for i in out if i != None]
    out = np.asarray(out)
    return out


def get_berry_id(filename):
    """
    Gets the sample number
    """
    if filename.split(sep=".")[-1] == 'png':
        return filename.split(sep='/')[-1].strip('.png').split(sep='_')[4]


def get_berry_ids(data_dir):
    """
    Gets a list of sample numbers
    """
    out = np.array([get_berry_id(filename) for filename in natsorted(os.listdir(data_dir))])
    out = [i for i in out if i != None]
    out = np.asarray(out)
    return out


def get_config_id(filename):
    """
    Gets the sample number
    """

    if filename.split(sep=".")[-1] == 'png':
        return filename.split(sep='/')[-1].strip('.png').split(sep='_')[1].strip('conf')


def get_config_ids(data_dir):
    """
    Gets a list of sample numbers
    """
    out = np.array([get_config_id(filename) for filename in natsorted(os.listdir(data_dir))])
    out = [i for i in out if i != None]
    out = np.asarray(out)
    return out


class DatasetRGB(ABC):

    def __init__(self, rgb_dir):
        self.samples = {}
        self.rgb_dir = rgb_dir
        # Load all IDs, images IDs, mean of weights and fake L elements
        self.samples["id"] = get_ids(self.rgb_dir)
        self.samples["number_id"] = get_number_ids(self.rgb_dir)
        self.samples["images_id"] = get_images_ids(self.rgb_dir)
        self.samples["img_enc"] = load_images(self.samples["images_id"], self.rgb_dir)
        self.samples["config"] = get_config_ids(self.rgb_dir)
        self.samples["berries_id"] = get_berry_ids(self.rgb_dir)

    def _split_train_test_val(self):
        idx_all = np.arange(0, self.samples["id"].shape[0])
        idx_train = np.array([], dtype=np.int64)
        idx_test = np.array([], dtype=np.int64)
        idx_val = np.array([], dtype=np.int64)

        for index in idx_all:
            if (self.samples["config"][index] == '25') and (
                    self.samples["berries_id"][index] == 'berry1' or self.samples["berries_id"][index] == 'berry2' or
                    self.samples["berries_id"][index] == 'berry3' or self.samples["berries_id"][index] == 'berry4' or
                    self.samples["berries_id"][index] == 'berry5'):
                idx_test = np.append(idx_test, index)   # add the image index to the testing index batch
                idx_val = np.append(idx_val, index)     # add the (same as test) image index to the validation index batch
            else:
                idx_train = np.append(idx_train, index)  # if the previous if is not satisfied -> image index added to train index batch
        return idx_train, idx_val, idx_test

    def prepare_data(self):
        idx_train, idx_val, idx_test = self._split_train_test_val()     # calls the previous function to divide indexs
        X_train = {"RGB": self.samples["img_enc"][idx_train].astype('float32')}     # crea image train batch
        X_val = {"RGB": self.samples["img_enc"][idx_val].astype('float32')}         # crea image val batch
        X_test = {"RGB": self.samples["img_enc"][idx_test].astype('float32')}       # crea image test batch

        self.data = (X_train), (X_val), (X_test)
        self.data_names = {'train_ids': [self.samples["images_id"][x] for x in idx_train],
                           'val_ids': [self.samples["images_id"][x] for x in idx_val],
                           'test_ids': [self.samples["images_id"][x] for x in idx_test]}
