import os , time as T
import numpy as np
import tensorflow as tf

from glob import glob
from model import *
from tqdm import tqdm
from utils import *
import tensorlayer as tl
import hyperparameter as hp


def build_dataset():
    data_list = np.array(glob(hp.datasets_path))
    hp.num_files = len(data_list)
    hp.batch_steps = int(hp.num_files // hp.batch_size)

    datasets = tf.data.Dataset.from_tensor_slices(data_list)
    datasets = datasets.map(get_imgs_test)

    datasets = datasets.repeat() \
        .shuffle(buffer_size=100) \
        .repeat().batch(hp.batch_size)

    iterator = datasets.make_initializable_iterator()
    return iterator


def imgsave(model, sample_noise ,global_steps):
    fileName = os.path.join(hp.samples_path,"Sample_IMG_gs:{}.jpg".format(global_steps))
    img = model.predimg(feed_dict={model.z: sample_noise})
    tilesize = int(np.sqrt(hp.sample_size))
    tl.visualize.save_images(img, [tilesize,tilesize], fileName)

def imgsave_step(model, sample_noise ,step):
    fileName = os.path.join("step_IMG","Step :{}.jpg".format(step))
    img = model.predimg(feed_dict={model.z: sample_noise})
    tilesize = int(np.sqrt(hp.sample_size))
    tl.visualize.save_images(img, [tilesize,tilesize], fileName)

def main():
    datasets = build_dataset()

    with tf.device("/gpu:0"):
        sess = tf.Session()

        model = DCGAN(sess)
        model.build_model()
        loss_d,loss_g = model.loss()
        d_train_op, g_train_op = model.train_op()

    sess.run(datasets.initializer)
    sess = model.init()
    log = tf.summary.FileWriter(hp.log_path, sess.graph)

    z_summ = tf.summary.histogram("noise", model.z)
    d_loss_real_summ = tf.summary.scalar("d_loss_real_summ", model.d_loss_R)
    d_loss_fake_summ = tf.summary.scalar("d_loss_fake_summ", model.d_loss_F)
    d_loss_summ = tf.summary.scalar("d_loss_summ", loss_d)
    g_loss_summ = tf.summary.scalar("g_loss_summ", loss_g)

    d_summ = tf.summary.merge([z_summ,d_loss_real_summ,d_loss_summ])
    g_summ = tf.summary.merge([z_summ,d_loss_fake_summ,g_loss_summ])

    load, global_steps = model.load("./ckpt/DCGAN.ckpt-76500")


    get_batch_data = datasets.get_next()

    sample_img = sess.run(get_batch_data)
    sample_noise = get_noise(hp.sample_size)

    for step in range(hp.epoch):

        for bat_step in tqdm(range(hp.batch_steps)):
            # data_generate
            bat_noise = get_noise(hp.batch_size)
            bat_img = sess.run(get_batch_data)

            # [training] discriminator
            D_summ , _ = sess.run([d_summ, d_train_op], feed_dict={model.z: bat_noise, model.images: bat_img})
            log.add_summary(D_summ, global_steps)

            # [training] generator
            G_summ, _ = sess.run([g_summ, g_train_op], feed_dict={model.z: bat_noise})
            log.add_summary(G_summ, global_steps)

            # eval
            if global_steps % 100 == 0 :
                log_loss_d, log_loss_g = sess.run([loss_d,loss_g],feed_dict={model.z: sample_noise, model.images: sample_img})

                print("[step:{} gs:{}] loss_d:{:.4} , loss_g:{:.4}"
                      .format(step+1,global_steps,log_loss_d,log_loss_g))

                imgsave(model, sample_noise, global_steps)

            if global_steps % (hp.batch_steps/2) == 0:
                model.save(epoch=step+1, step=global_steps)

            global_steps += 1

        imgsave_step(model, sample_noise, step+1)

    sess.close()


if __name__ == '__main__':
    main()
