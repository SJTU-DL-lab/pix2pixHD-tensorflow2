import os
import tensorflow as tf
import tensorflow.keras as K
from options.train_options import TrainOptions
from data.aligned_dataset import aligned_dataset
from models.pix2pixHD_model import Pix2PixHDModel
from models.networks import VGGLoss
from util.image_pool import ImagePool

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
dataset, len_datset = aligned_dataset(opt)

model = Pix2PixHDModel()
model.initialize(opt)
# model.encode_input(label_map)
fake_pool = ImagePool(opt.pool_size)
start_epoch, epoch_iter = 1, 0

if opt.no_lsgan:
    criterionGAN = K.losses.BinaryCrossentropy()
else:
    criterionGAN = K.losses.MeanSquaredError()
criterionFeat = K.losses.MeanAbsoluteError()
criterionVGG = VGGLoss()

def train_D(fake_pair, real_pair):

    loss_object(tf.ones_like(real_pair), real_pair)


def train_G(input):
    pass


def discriminate(input, target_is_real):
    if isinstance(input[0], list):
        loss = 0
        for input_i in input:
            pred = input_i[-1]
            if target_is_real:
                target_tensor = tf.ones_like(pred)
            else:
                target_tensor = tf.zeros_like(pred)
            loss += criterionGAN(pred, target_tensor)
        return loss
    else:
        if target_is_real:
            target_tensor = tf.ones_like(input[-1])
        else:
            target_tensor = tf.zeros_like(input[-1])
        return criterionGAN(input[-1], target_tensor)


for ep in range(start_epoch, opt.niter + opt.niter_decay + 1):
    print(ep)
    for step, (label, real_img) in enumerate(dataset):
        print(step)
        input_label, inst_map, real_image, feat_map = model.encode_input(label)

        fake_img = model.netG(input_label)

        real_pair = tf.concat([input_label, real_img], axis=-1)
        fake_pair = tf.concat([input_label, fake_img], axis=-1)
        fake_pair = fake_pool.query(fake_pair)

        # Fake Detection and Loss
        pred_fake_pool = model.netD(fake_pair)
        loss_D_fake = discriminate(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = model.netD(real_pair)
        loss_D_real = discriminate(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = model.netD(fake_pair)
        loss_G_GAN = discriminate(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not opt.no_ganFeat_loss:
            feat_weights = 4.0 / (opt.n_layers_D + 1)
            D_weights = 1.0 / opt.num_D
            for i in range(opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        criterionFeat(pred_fake[i][j], pred_real[i][j]) * opt.lambda_feat

        if not opt.no_vgg_loss:
            loss_G_VGG = criterionVGG(fake_img, real_img) * opt.lambda_feat
