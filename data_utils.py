from __future__ import division
import os
from glob import glob
from time import time

import numpy as np
import pandas as pd
from PIL import Image
import skimage
from skimage.transform import resize
from skimage.data import imread
import skimage.transform
from skimage.transform._warps_cy import _warp_fast

import multiprocessing as mp
from multiprocessing.pool import Pool
from functools import partial
from math import sin, cos

import pdb
import pickle


def get_image_files(datadir, left_only=False, shuffle=False):
    files = glob('{}/*'.format(datadir))
    if left_only:
        files = [f for f in files if 'left' in f]
    if shuffle:
        return np.random.shuffle(files)
    return sorted(files)



def compute_mean_across_channels(files, batch_size=512):
    ret = np.zeros(3)
    shape = None
    for i in range(0, len(files), batch_size):
        print('processing from {}'.format(i))
        images = load_images(files[i : i + batch_size])
        images=resize_same_images(images)
        shape = images.shape
        ret += images.sum(axis=(0, 2, 3))
    n = len(files) * shape[2] * shape[3]
    return (ret / n).astype(np.float32)


def compute_std_across_channels(files, batch_size=512):
    s = np.zeros(3)
    s2 = np.zeros(3)
    shape = None
    for i in range(0, len(files), batch_size):
        print('processing from {}'.format(i))
        #images = np.array(load_images_uint(files[i : i + batch_size]), dtype=np.float64)
        images = load_images(files[i : i + batch_size])
        images=resize_same_images(images)
        shape = images.shape
        s += images.sum(axis=(0, 2, 3))
        s2 += np.power(images, 2).sum(axis=(0, 2, 3))
    n = len(files) * shape[2] * shape[3]
    var = (s2 - s**2.0 / n) / (n - 1)
    return np.sqrt(var).astype(np.float32)

def compute_stat_pixel(files, batch_size=512):
    dummy_img = load_images(files[0])
    shape = dummy_img.shape[1:]
    mean = np.zeros(shape)
    batches = []

    for i in range(0, len(files), batch_size):
        images = load_images(files[i : i + batch_size])
        batches.append(images)
        mean += images.sum(axis=0)
    n = len(files)
    mean = (mean / n).astype(np.float32)

    std = np.zeros(shape)
    for b in batches:
        std += ((b - mean) ** 2).sum(axis=0)
    std = np.sqrt(std / (n - 1)).astype(np.float32)
    return mean, std

class re_resize(object):
    def __init__(self,output_shape):
        self.output_shape=output_shape
    
    def __call__(self,image):
        img=resize(image,output_shape=self.output_shape)
        return img
    
def load_images(files):
    p = Pool()
    process = imread
    results = p.map(process, files)
    #results=list(map(process,files))
    #images = np.array(results, dtype=np.float32)
    p.close()
    p.join()
    #print('number of images:%d'%len(results))
    return results

def resize_same_images(images_list,shape=(224,224)):
    Resize=re_resize(shape)
    p = Pool()
    res=p.map(Resize,images_list)
    #res=list(map(Resize,images_list))
    p.close()
    p.join()
    images=np.array(res,dtype=np.float32)
    images = images.transpose(0, 3, 1, 2)
    #print(images.shape)
    return images
    

def get_labels(label_file):
    labels = pd.read_csv(label_file)
    labels=[int(x) for x in labels.iloc[:,1].values]
    return labels

def one_hot(labels):
    identity = np.eye(max(labels) + 1)
    return identity[labels].astype(np.int32)


def get_files(basedir,label_file,shuffle=False):
    file=pd.read_csv(label_file)
    names=[os.path.join(basedir,str(x))+'.jpeg' for x in file.iloc[:,0].values]
    labels=[int(x) for x in file.iloc[:,1].values ]
    
    image_list=np.array([names,labels])
    image_list=image_list.transpose()
    print(image_list.shape)
    if shuffle:
        np.random.shuffle(image_list)
    
    #image_list=image_list.transpose()
    images=[str(x) for x in image_list[:,0]]
    labels=[int(i)for i in image_list[:,1]]
    return  images,labels


def fast_warp(img, tf, mode='constant', order=0):
    m = tf.params
    t_img = np.zeros(img.shape, img.dtype)
    for i in range(t_img.shape[0]):
        t_img[i] = _warp_fast(img[i], m, mode=mode, order=order)
    return t_img


def build_augmentation_transform(test=False):
    pid = mp.current_process()._identity[0]
    randst = np.random.mtrand.RandomState(pid + int(time() % 3877))
    if not test:
        r = randst.uniform(-0.1, 0.1)  # scale
        rotation = randst.uniform(0, 2 * 3.1415926535)
        skew = randst.uniform(-0.2, 0.2) + rotation
    else: # only rotate randomly during test time
        r = 0
        rotation = randst.uniform(0, 2 * 3.1415926535)
        skew = rotation

    homogenous_matrix = np.zeros((3, 3))
    c00 = (1 + r) * cos(rotation)
    c10 = (1 + r) * sin(rotation)
    c01 = -(1 - r) * sin(skew)
    c11 = (1 - r) * cos(skew)

    # flip every other time
    if randst.randint(0, 2) == 0:
        c00 *= -1
        c10 *= -1

    homogenous_matrix[0][0] = c00
    homogenous_matrix[1][0] = c10
    homogenous_matrix[0][1] = c01
    homogenous_matrix[1][1] = c11
    homogenous_matrix[2][2] = 1


    transform = skimage.transform.AffineTransform(homogenous_matrix)
    return transform


def build_center_uncenter_transforms(image_shape):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    """

    # need to swap rows and cols here apparently! confusing!
    center_shift = np.array([image_shape[1], image_shape[0]]) / 2.0 - 0.5
    tform_uncenter = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter


def augment(img, test=False):
    augment = build_augmentation_transform(test)
    center, uncenter = build_center_uncenter_transforms(img.shape[1:])
    transform = uncenter + augment + center
    img = fast_warp(img, transform, mode='constant', order=0)
    return img


def parallel_augment(images, normalize=None, test=False):
    if normalize is not None:
        mean, std = normalize
        images = images - mean[:, np.newaxis, np.newaxis] # assuming channel-wise normalization
        images = images / std[:, np.newaxis, np.newaxis]

    p = Pool()
    process = partial(augment, test=test)
    results = p.map(process, images)
    #results=list(map(process,images))
    p.close()
    p.join()
    augmented_images = np.array(results, dtype=np.float32)
    return augmented_images

def oversample_set(files, labels, coefs):
    """
    files: list of filenames in the train set
    labels: the corresponding labels for the files
    coefs: list of oversampling ratio for each class
    Code modified from github.com/JeffreyDF.`
    """

    train_1 = list(np.where(np.apply_along_axis(
        lambda x: 1 == x,
        0,
        labels))[0])
    train_2 = list(np.where(np.apply_along_axis(
        lambda x: 2 == x,
        0,
        labels))[0])
    train_3 = list(np.where(np.apply_along_axis(
        lambda x: 3 == x,
        0,
        labels))[0])
    train_4 = list(np.where(np.apply_along_axis(
        lambda x: 4 == x,
        0,
        labels))[0])

    print(len(train_1), len(train_2), len(train_3), len(train_4))
    X_oversample = list(files)
    X_oversample += list(np.array(files)[coefs[1] * train_1])
    X_oversample += list(np.array(files)[coefs[2] * train_2])
    X_oversample += list(np.array(files)[coefs[3] * train_3])
    X_oversample += list(np.array(files)[coefs[4] * train_4])

    y_oversample = np.array(labels)
    y_oversample = np.hstack([y_oversample, [labels[i] for i in coefs[1] * train_1]])
    y_oversample = np.hstack([y_oversample, [labels[i] for i in coefs[2] * train_2]])
    y_oversample = np.hstack([y_oversample, [labels[i] for i in coefs[3] * train_3]])
    y_oversample = np.hstack([y_oversample, [labels[i] for i in coefs[4] * train_4]])

    return X_oversample, y_oversample


def oversample_set_pairwise(files, labels, merged, coefs):
    """
    files: list of paired filenames in the train set
    labels: the corresponding label pairs for the file pairs
    merged: merged labels
    coefs: list of oversampling ratio for each class
    Code modified from github.com/JeffreyDF.`
    """

    train_1 = list(np.where(np.apply_along_axis(
        lambda x: 1 == x,
        0,
        merged))[0])
    train_2 = list(np.where(np.apply_along_axis(
        lambda x: 2 == x,
        0,
        merged))[0])
    train_3 = list(np.where(np.apply_along_axis(
        lambda x: 3 == x,
        0,
        merged))[0])
    train_4 = list(np.where(np.apply_along_axis(
        lambda x: 4 == x,
        0,
        merged))[0])

    print(len(train_1), len(train_2), len(train_3), len(train_4))
    X_oversample = list(files)
    X_oversample += list(np.array(files)[coefs[1] * train_1])
    X_oversample += list(np.array(files)[coefs[2] * train_2])
    X_oversample += list(np.array(files)[coefs[3] * train_3])
    X_oversample += list(np.array(files)[coefs[4] * train_4])

    y_oversample = list(labels)
    y_oversample += list(np.array(labels)[coefs[1] * train_1])
    y_oversample += list(np.array(labels)[coefs[2] * train_2])
    y_oversample += list(np.array(labels)[coefs[3] * train_3])
    y_oversample += list(np.array(labels)[coefs[4] * train_4])

    return X_oversample, y_oversample


from time import time
from queue import  Queue
import threading

class BatchIterator(object):

    def __init__(self, files, labels, batch_size, normalize=None, process_func=None, testing=None):
        self.files = np.array(files)
        self.labels = labels
        self.n = len(files)
        self.batch_size = batch_size
        self.testing = testing

        if normalize is not None:
            self.mean, self.std = normalize
            #self.mean = np.load(mean)
            #self.std = np.load(std)
        else:
            self.mean = np.array([0])
            self.std = np.array([1])

        if process_func is None:
            process_func = lambda x, y, z: x
        self.process_func = process_func

        if not self.testing:
            self.create_index = lambda: np.random.permutation(self.n)
        else:
            self.create_index = lambda: range(self.n)


        self.indices = self.create_index()
        assert self.n >= self.batch_size


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def get_permuted_batch_idx(self):
        if len(self.indices) <= self.batch_size:
            new_idx = self.create_index()
            self.indices = np.hstack([self.indices, new_idx])

        batch_idx = self.indices[:self.batch_size]
        self.indices = self.indices[self.batch_size:]

        return batch_idx

    def next(self):
        batch_idx = self.get_permuted_batch_idx()
        batch_files = self.files[batch_idx]
        image_files=load_images(batch_files)
        batch_X = resize_same_images(image_files)
        batch_X = self.process_func(batch_X, (self.mean, self.std), self.testing)
        batch_y = [self.labels[i] for i in batch_idx]
        return (batch_X, batch_y)
    
    
def threaded_iterator(iterator, num_cached=50):
    queue = Queue(maxsize=num_cached)
    sentinel = object()

    def producer():
        for item in iterator:
            queue.put(item)
        queue.put(sentinel)

    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()


def continuous_kappa(y, t, y_pow=1, eps=1e-15):
    if y.ndim == 1:
        y = one_hot(y)

    if t.ndim == 1:
        t = one_hot(t)

    # Weights.
    num_scored_items, num_ratings = y.shape
    ratings_mat = np.tile(np.arange(0, num_ratings)[:, None],
                          reps=(1, num_ratings))
    ratings_squared = (ratings_mat - ratings_mat.T) ** 2
    weights = ratings_squared / float(num_ratings - 1) ** 2

    if y_pow != 1:
        y_ = y ** y_pow
        y_norm = y_ / (eps + y_.sum(axis=1)[:, None])
        y = y_norm

    hist_rater_a = np.sum(y, axis=0)
    hist_rater_b = np.sum(t, axis=0)

    conf_mat = np.dot(y.T, t)

    nom = weights * conf_mat
    denom = (weights * np.dot(hist_rater_a[:, None],
                              hist_rater_b[None, :]) /
             num_scored_items)

    return 1 - nom.sum() / denom.sum(), conf_mat, \
        hist_rater_a, hist_rater_b, nom, denom

#datadir='/home/ye/user/yejg/database/Kaggle_Eye/train_001/train'
#label_file='/home/ye/user/yejg/database/Kaggle_Eye/train_001/trainLabels.csv'
#
#files,labels=get_files(basedir=datadir,label_file=label_file,shuffle=True)
#
#images_mean=compute_mean_across_channels(files,batch_size=32)
#images_std=compute_std_across_channels(files,batch_size=32)
#
#feed_dict={'mean':images_mean,'std':images_std}
#
#exp_pkl = open('features/'+'images_features.pkl', 'wb')
#data = pickle.dumps(feed_dict)
#exp_pkl.write(data)
#exp_pkl.close()
