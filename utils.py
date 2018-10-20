import numpy as np
import hyperparameter as hp
import tensorflow as tf


def get_noise(size):
    return np.random.uniform(-1,1,size=(size, hp.z_dim)).astype(np.float32)


def get_imgs_test(img_path):
    with tf.name_scope("Data_Processing"):
        img = tf.read_file(img_path)
        img = tf.image.decode_jpeg(img,channels=hp.c_dim)
        img = tf.cast(img,tf.float32)
        img = tf.divide(img,127.5)
        img = tf.subtract(img,1.0)
        return img
