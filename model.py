import hyperparameter as hp
import os
import time
from ops import *
import tensorflow as tf


class DCGAN(object):

    def generator(self,inputs, is_train, reuse):
        with tf.variable_scope("generator", reuse=reuse):
            # input = ? , 100
            # ? , 1024
            tensor = dense(inputs, hp.gf_dim * hp.s16 * hp.s16 * 8, name="h0/lin")

            # ? , 4, 4, 512
            tensor = tf.reshape(tensor,shape = [-1, hp.s16, hp.s16, hp.gf_dim * 8],name="h0/reshape")
            tensor = relu(bat_norm(tensor, is_training=is_train, name="h0/bn"),name="h0/relu")

            # ? , 8, 8, 256
            tensor = deconv2d_layer(tensor, filters=hp.gf_dim * 4, is_training= is_train, name="h1/")

            # ? , 16, 16, 128
            tensor = deconv2d_layer(tensor, filters=hp.gf_dim * 2, is_training= is_train, name="h2/")

            # ? , 32, 32, 64
            tensor = deconv2d_layer(tensor, filters=hp.gf_dim * 1, is_training= is_train, name="h3/")

            # ?, 64, 64, 3
            tensor = deconv2d(tensor, hp.c_dim, name="h4/Deconv2d")
            outputs = tf.tanh(tensor)

            return outputs

    def discriminator(self,inputs, is_train, reuse):
        with tf.variable_scope("discriminator", reuse=reuse):
            # input.shape = ?, 64, 64, 3
            # ? , 32, 32, 64
            tensor = conv2d_layer(inputs, hp.df_dim * 1, is_train, False,name="h0/")

            # ? , 16, 16, 128
            tensor = conv2d_layer(tensor, hp.df_dim * 2, is_train, True, name="h1/")

            # ? , 8, 8, 256
            tensor = conv2d_layer(tensor, hp.df_dim * 4, is_train, True, name="h2/")

            # ? , 4, 4, 512
            tensor = conv2d_layer(tensor, hp.df_dim * 8, is_train, True, name="h4/")

            # ?, 1
            logits = dense(flatten(tensor,name="h4/lin/flatten"), # ?, 8192
                           units = 1,
                           activation = tf.identity,
                           name="h4/lin/dense")

            outputs = tf.nn.sigmoid(logits)
            return outputs, logits

    def build_model(self):
        self.z = tf.placeholder(tf.float32, [None,hp.z_dim], name="noise_inputs")
        self.images= tf.placeholder(tf.float32, [None, hp.output_size, hp.output_size, hp.c_dim] , name="real_IMG_inputs")

        G = self.generator(self.z, is_train=True, reuse=False)
        self.d_r, self.d_r_logits = self.discriminator(self.images, is_train=True, reuse=False)
        self.d_g, self.d_g_logits = self.discriminator(G, is_train=True, reuse=True)
        self.pred_G = self.generator(self.z,is_train=False,reuse=True)

    def loss(self):
        self.d_loss_R = sigmoid_cross_entropy(logits= self.d_r_logits,
                                              labels= tf.ones_like(self.d_r))

        self.d_loss_F = sigmoid_cross_entropy(logits=self.d_g_logits,
                                              labels=tf.zeros_like(self.d_g))

        self.d_loss = self.d_loss_R + self.d_loss_F

        self.g_loss = sigmoid_cross_entropy(logits=self.d_g_logits,
                                            labels=tf.ones_like(self.d_g))

        return self.d_loss, self.g_loss

    def train_op(self):
        t_var = tf.trainable_variables()
        d_vars = [var for var in t_var if var.name.startswith("discriminator")]
        g_vars = [var for var in t_var if var.name.startswith("generator")]

        d_train_op = tf.train.AdamOptimizer(hp.lr, beta1= hp.beta1).minimize(self.d_loss, var_list=d_vars)
        g_train_op = tf.train.AdamOptimizer(hp.lr, beta1=hp.beta1).minimize(self.g_loss, var_list=g_vars)

        return d_train_op, g_train_op

    def predimg(self,feed_dict):
        img = self.sess.run(self.pred_G,feed_dict=feed_dict)
        return img

    def save(self,epoch, step):
        ckpt_path = os.path.join(hp.checkpoint_path,"DCGAN_{}.ckpt".format(epoch))
        self.saver.save(self.sess,ckpt_path, global_step=step,
                        latest_filename="epoch_{}".format(epoch))


    def load(self, ckpt_path):
        ckpt = tf.train.checkpoint_exists(ckpt_path)

        if ckpt:
            print("[ ] Load Check Point .....")
            self.saver.restore(self.sess,ckpt_path)
            global_steps = int(os.path.basename(ckpt_path).split('-')[1])
            print("[*] Load Check Point Success!!")
            return True, global_steps
        else:
            print("[ ] Load Check Point Failed!!")
            return False, 1

    def __init__(self,sess):
        self.sess = sess
        self.pred_G = None

    def init(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=hp.epoch)

        return self.sess