from tensorflow.image import ResizeMethod
from options.train_options import TrainOptions
import tensorflow as tf
import numpy as np
import random


opt = TrainOptions().parse()

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


@tf.function
def preprocess_image_A(img):
    transform_list = []
    # params = get_params(opt, img.shape)
    if opt.label_nc == 0:
        method = ResizeMethod.NEAREST_NEIGHBOR
    else:
        method = opt.resize_method
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        img = tf.image.resize(img, osize, method)
    elif 'scale_width' in opt.resize_or_crop:
        img = scale_width(img, opt.loadSize, method)

    # if 'crop' in opt.resize_or_crop:
    #     img = crop(img, params['crop_pos'], fineSize)

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        img = make_power_2(img, base, method)

    # if opt.isTrain and not opt.no_flip:
    #     img = flip(img, params['flip'])

    if opt.label_nc == 0:
        if not opt.no_normalize_img:
            img = (img - 0.5) / 0.5

    return img


@tf.function
def preprocess_image_B(img):
    # params = get_params(opt, size)
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        img = tf.image.resize(img, osize, opt.resize_method)
    elif 'scale_width' in opt.resize_or_crop:
        img = scale_width(img, opt.loadSize, opt.resize_method)

    # if 'crop' in opt.resize_or_crop:
    #     img = crop(img, params['crop_pos'], opt.fineSize)

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        img = make_power_2(img, base, opt.resize_method)

    # if opt.isTrain and not opt.no_flip:
    #     img = flip(img, params['flip'])

    if not opt.no_normalize_img:
        img = tf.cast(img, tf.float32)
        img = img / 255
        img = (img - 0.5) / 0.5

    return img


def make_power_2(img, base, method=ResizeMethod.BICUBIC):
    ow, oh = opt.inputW, opt.inputH
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return tf.image.resize(img, (w, h), method)

def scale_width(img, target_width, method=ResizeMethod.BICUBIC):
    ow, oh = opt.inputW, opt.inputH
    if (ow == target_width):
        return img
    w = target_width
    h = target_width * oh / ow
    return tf.image.resize(img, (w, h), method)

def crop(img, pos, size):
    ow, oh = opt.inputW, opt.inputH
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return tf.image.crop_to_bounding_box(img, y1, x1, th, tw)
    return img

def flip(img, flip):
    if flip:
        return tf.image.flip_left_right(img)
    return img
