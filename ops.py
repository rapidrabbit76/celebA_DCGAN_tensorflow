import tensorflow as tf
from tensorflow.nn import relu
from tensorflow.nn import leaky_relu as lrelu


flatten = tf.layers.flatten

def deconv2d(inputs, filters,kernel_size=5, strides=2, name=None):
    return tf.layers.conv2d_transpose(inputs,
                                      filters,
                                      kernel_size,
                                      strides,
                                      'SAME',
                                      kernel_initializer=tf.glorot_normal_initializer(),
                                      use_bias=False,
                                      name=name)


def conv2d(inputs, filters,kernel_size=5,strides=2, name=None):
    return tf.layers.conv2d(inputs,
                            filters,
                            kernel_size,
                            strides,
                            'SAME',
                            kernel_initializer=tf.glorot_normal_initializer(),
                            use_bias=False,
                            name=name)

'''
Do not use [tf.layers.batch_normalization]
ues [contrib.layers.batch_norm]
'''


def bat_norm(inputs,is_training,name=None):
    return tf.contrib.layers.batch_norm(inputs,
                                        decay=0.9,
                                        updates_collections= None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=name)


def dense(inputs, units, activation=None, name=None):
    return tf.layers.dense(inputs,
                           units=units,
                           activation=activation,
                           kernel_initializer=tf.glorot_normal_initializer(),
                           use_bias = False,
                           name=name)


def deconv2d_layer(inputs, filters, is_training,name=None):
    tensor = deconv2d(inputs,filters, name=name+"Deconv2d")
    tensor = bat_norm(tensor, is_training=is_training, name=name+"bn")
    tensor = relu(tensor,name=name+"relu")
    return tensor


def conv2d_layer(inputs, filters, is_training, batch_norm=True, name=None):
    tensor = conv2d(inputs,filters,name=name+"conv2d")
    if batch_norm:
        tensor = bat_norm(tensor, is_training=is_training,name=name+"bn")
    tensor = lrelu(tensor,name=name+"lrelu")
    return tensor


def sigmoid_cross_entropy(logits, labels):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                labels=labels))

