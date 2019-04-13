import os
import tensorflow as tf
import tensorflow.keras as K
import tf2lib as tl
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

# checkpoint
checkpoint_dir = os.path.join(self.opt.checkpoints_dir,
                              self.opt.name, 'train_ckpt')
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=model.optimizer_G,
                                 discriminator_optimizer=model.optimizer_D,
                                 generator=model.netG,
                                 discriminator=model.netD)
# summary
train_summary_writer = tf.summary.create_file_writer(os.path.join(self.opt.checkpoints_dir,
                                                     self.opt.name, 'summary'))

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


def train_step(inputs, real_img):
    input_label = inputs[0]
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
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

        loss_D = loss_D_fake + loss_D_real
        loss_G = loss_G_GAN + loss_G_GAN_Feat + loss_G_VGG

    gen_grads = gen_tape.gradient(loss_G, model.netG.trainable_variables)
    disc_grads = disc_tape.gradient(loss_D, model.netD.trainable_variables)

    model.optimizer_G.apply_gradients(zip(gen_grads,
                                          model.netG.trainable_variables))
    model.optimizer_D.apply_gradients(zip(disc_grads,
                                          model.netD.trainable_variables))

    loss_D_dict = {
        'D_fake': loss_D_fake,
        'D_real': loss_D_real
        }
    loss_G_dict = {
        'G_GAN': loss_G_GAN,
        'G_GAN_Feat': loss_G_GAN_Feat,
        'loss_G_VGG': loss_G_VGG
    }
    return loss_D_dict, loss_G_dict


# main loop
with train_summary_writer.as_default():
    for ep in range(start_epoch, opt.niter + opt.niter_decay + 1):
        print('Epoch: ', ep)
        for step, (label, real_img) in enumerate(dataset):
            input_label, inst_map, real_image, feat_map = model.encode_input(label)
            inputs = (input_label, inst_map, real_image, feat_map)
            loss_D_dict, loss_G_dict = train_step(inputs, real_img)

            if step % opt.save_epoch_freq == 0:
                checkpoint.save(checkpoint_prefix)

            tl.summary(loss_G_dict, step=model.netG.iterations, name='G_losses')
            tl.summary(loss_D_dict, step=model.netG.iterations, name='D_losses')
