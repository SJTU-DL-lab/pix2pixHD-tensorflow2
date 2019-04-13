import os
import tf2lib as tl
import tensorflow as tf
from data.transform_dataset import get_params, preprocess_image_A, preprocess_image_B

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def aligned_dataset(opt):
    opt = opt
    root = opt.dataroot

    ### input A (label maps)
    dir_A = '_A' if opt.label_nc == 0 else '_label'
    dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
    A_paths = sorted(make_dataset(dir_A))

    ### input B (real images)
    if opt.isTrain or opt.use_encoded_image:
        dir_B = '_B' if opt.label_nc == 0 else '_img'
        dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        B_paths = sorted(make_dataset(dir_B))

    assert len(A_paths) == len(B_paths)
    A_dataset = tl.disk_image_batch_dataset(A_paths,
                                            opt.batchSize,
                                            drop_remainder=not opt.no_drop_remainder,
                                            map_fn=preprocess_image_A,
                                            shuffle=not opt.no_shuffle,
                                            repeat=opt.repeat_num)

    B_dataset = tl.disk_image_batch_dataset(B_paths,
                                            opt.batchSize,
                                            drop_remainder=not opt.no_drop_remainder,
                                            map_fn=preprocess_image_B,
                                            shuffle=not opt.no_shuffle,
                                            repeat=opt.repeat_num)
    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = len(A_paths) // opt.batchSize
    return A_B_dataset, len_dataset


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images
