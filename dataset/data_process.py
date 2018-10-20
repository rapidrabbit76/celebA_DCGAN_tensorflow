from tqdm import tqdm
import os
from glob import glob
from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image as IM
import scipy.misc
import imageio as io
import numpy as np

IMAGE_PATH = "../../../dataset/celebA/*.jpg"
SAVE_PATH = "../../../dataset/celebA_crop"
NUM_THREAD = 16


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])


def transform(image, npx=64, resize_w=64):
    cropped_image = center_crop(image, npx, resize_w=resize_w)
    return np.array(cropped_image)


def imread(path, is_grayscale = False):
    if (is_grayscale):
        return io.imread(path).astype(np.float).flatten()
    else:
        return io.imread(path).astype(np.float)


def get_image(image_path, image_size, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, resize_w)


def data_process(data):
    filename = os.path.basename(data)
    filename = os.path.join(SAVE_PATH, filename)
    img = get_image(data, 108, resize_w=64, is_grayscale=0)
    im = IM.fromarray(img)
    im.save(filename)


def parallel_process(fn,item):
    pool = ThreadPool(NUM_THREAD)
    for _ in tqdm(pool.imap_unordered(fn, item), total=len(item)):
        pass


def main():
    dataset = glob(IMAGE_PATH)
    parallel_process(fn=data_process,item=dataset)


if __name__ == '__main__':
    main()