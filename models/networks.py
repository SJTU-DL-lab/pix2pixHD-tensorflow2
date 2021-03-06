import tensorflow as tf
import tensorflow.keras as K
from tensorflow_addons.layers import InstanceNormalization
from options.train_options import TrainOptions
import functools
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg19 import VGG19


opt = TrainOptions().parse()
weight_init = {}
weight_init['conv'] = tf.random_normal_initializer(0.0, 0.02)
weight_init['bn_gamma'] = tf.random_normal_initializer(1.0, 0.02)
weight_init['bn_beta'] = tf.zeros_initializer()


# Functions
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        # BUG HERE
        norm_layer = functools.partial(layers.BatchNormalization,
                                       gamma_initializer=weight_init['bn_gamma'],
                                       beta_initializer=weight_init['bn_beta'])
    elif norm_type == 'instance':
        norm_layer = InstanceNormalization
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance'):

    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                                  n_local_enhancers, n_blocks_local, norm_layer)
    # elif netG == 'encoder':
    #     netG = Encoder(output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise('generator not implemented!')
    return netG


def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)

    return netD


# Losses
class VGGLoss(layers.Layer):
    def __init__(self):
        super(VGGLoss, self).__init__(name='VGGLoss')
        self.vgg = Vgg19()
        self.criterion = K.losses.MeanAbsoluteError()  # lambda ta, tb: tf.reduce_mean(tf.abs(ta - tb))
        self.layer_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def call(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            y_vgg_temp = tf.stop_gradient(y_vgg[i])
            loss += self.layer_weights[i] * self.criterion(x_vgg[i], y_vgg_temp)
        return loss


# Generator
class LocalEnhancer(K.Model):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=get_norm_layer('batch'), padding_type='REFLECT'):
        super(LocalEnhancer, self).__init__(name='LocalEnhancer')
        self.n_local_enhancers = n_local_enhancers
        paddings = (3, 3)

        # global generator
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).layers[0].layers
        model_global = model_global[:-3] # get rid of final convolution layers
        self.model = K.Sequential(model_global)

        # local enhancer layers
        for n in range(1, n_local_enhancers+1):
            # downsample
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = K.Sequential()
            model_downsample.add(ReflectionPad2d(paddings))
            model_downsample.add(layers.Conv2D(ngf_global, 7, kernel_initializer=weight_init['conv']))
            model_downsample.add(norm_layer())
            model_downsample.add(layers.ReLU())
            model_downsample.add(layers.Conv2D(ngf_global * 2, 3, strides=2, padding='same', kernel_initializer=weight_init['conv']))
            model_downsample.add(norm_layer())
            model_downsample.add(layers.ReLU())

            # residual blocks
            model_upsample = K.Sequential()
            for i in range(n_blocks_local):
                model_upsample.add(ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer))

            # upsample
            model_upsample.add(layers.Conv2DTranspose(ngf_global, 3, strides=2, padding='same', output_padding=1, kernel_initializer=weight_init['conv']))
            model_upsample.add(norm_layer())
            model_upsample.add(layers.ReLU())

            # final convolution
            if n == n_local_enhancers:
                model_upsample.add(ReflectionPad2d(paddings))
                model_upsample.add(layers.Conv2D(output_nc, 7, kernel_initializer=weight_init['conv']))
                model_upsample.add(Tanh(name='local_output'))

            setattr(self, 'model'+str(n)+'_1', model_downsample)
            setattr(self, 'model'+str(n)+'_2', model_upsample)

        self.downsample = layers.AveragePooling2D(3, strides=2, padding='same')

    def call(self, x):

        input_downsampled = [x]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        # output at coarest level
        output_prev = self.model(input_downsampled[-1])

        # build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):

            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)

        return output_prev

class GlobalGenerator(K.Model):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9,
                 norm_layer=get_norm_layer('batch'),
                 padding_type='REFLECT'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        self.padding_type = padding_type
        self.output_nc = output_nc
        paddings = (3, 3)
        activation = layers.ReLU()

        model = K.Sequential()
        model.add(layers.InputLayer([None, None, input_nc]))
        model.add(ReflectionPad2d(paddings))
        model.add(layers.Conv2D(ngf, 7, kernel_initializer=weight_init['conv']))
        model.add(norm_layer())
        model.add(activation)

        # downsample
        for i in range(n_downsampling):
            mult = 2**i
            model.add(layers.Conv2D(ngf * mult * 2, 3, strides=2, padding='same', kernel_initializer=weight_init['conv']))
            model.add(norm_layer())
            model.add(activation)

        # resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model.add(ResnetBlock(dim=ngf * mult, padding_type=padding_type, norm_layer=norm_layer, activation=activation))

        # upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model.add(layers.Conv2DTranspose(int(ngf * mult / 2), 3, strides=2, padding='same', output_padding=1, kernel_initializer=weight_init['conv']))
            model.add(norm_layer())
            model.add(activation)
        model.add(ReflectionPad2d(paddings))
        model.add(layers.Conv2D(self.output_nc, 7, kernel_initializer=weight_init['conv']))
        model.add(Tanh(name='global_output'))

        # inputs = K.Input(shape=(None, None, input_nc), name='GlobalGenerator_input')
        # outputs = model(inputs)
        # model = K.Model(inputs, outputs)

        self.model = model

    def call(self, x):
        # x = tf.pad(x, paddings, self.padding_type)
        # x = self.model(x)
        # x = tf.pad(x, paddings, self.padding_type)
        # x = layers.Conv2D(self.output_nc, 7, kernel_initializer=weight_init['conv'])(x)
        # x = K.activations.tanh(x)
        x = self.model(x)

        return x


class ResnetBlock(layers.Layer):
    def __init__(self, dim, padding_type, norm_layer,
                 activation=layers.ReLU(), use_dropout=False, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        # CONSTANT REFLECT SYMMETRIC
        model = K.Sequential((layers.ZeroPadding2D(padding=(1, 1)),
                              layers.Conv2D(dim, 3, kernel_initializer=weight_init['conv']),
                              norm_layer(),
                              activation
                              ))
        if use_dropout:
            model.add(layers.Dropout(0.5))
        model.add(layers.ZeroPadding2D([1, 1]))
        model.add(layers.Conv2D(dim, 3, kernel_initializer=weight_init['conv']))
        model.add(norm_layer())
        self.model = model

    def call(self, x):
        identity = x
        x = self.model(x)
        out = x + identity

        return out

# Discriminator
class MultiscaleDiscriminator(K.Model):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=get_norm_layer('batch'),
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__(name='MultiscaleDiscriminator')
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD)

        self.downsample = layers.AveragePooling2D(3, strides=2, padding='same')

    def singleD_forward(self, model, x):
        if self.getIntermFeat:
            result = [x]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(x)]

    def call(self, x):
        num_D = self.num_D
        result = []
        input_downsampled = x
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(K.Model):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=get_norm_layer('batch'), use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__(name='NLayerDiscriminator')
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        sequence = K.Sequential()
        sequence.add(layers.InputLayer([None, None, input_nc]))
        sequence.add(layers.Conv2D(ndf, kw, strides=2, padding='same', kernel_initializer=weight_init['conv']))
        sequence.add(layers.LeakyReLU(0.2))

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence.add(layers.Conv2D(nf, kw, strides=2, padding='same', kernel_initializer=weight_init['conv']))
            sequence.add(norm_layer())
            sequence.add(layers.LeakyReLU(0.2))

        ng_prev = nf
        nf = min(nf * 2, 512)
        sequence.add(layers.Conv2D(nf, kw, strides=1, kernel_initializer=weight_init['conv']))
        sequence.add(layers.ZeroPadding2D())
        sequence.add(norm_layer())
        sequence.add(layers.LeakyReLU(0.2))

        sequence.add(layers.Conv2D(1, kw, strides=1, kernel_initializer=weight_init['conv']))
        sequence.add(layers.ZeroPadding2D())

        if use_sigmoid:
            sequence.add(Sigmoid())

        if getIntermFeat:
            for n in range(len(sequence.layers)):
                setattr(self, 'model'+str(n), sequence.layers[n])
        else:
            self.model = sequence

    def call(self, x):
        if self.getIntermFeat:
            res = [x]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(x)

# the keras vgg19 do not count ReLU layer
# the index is a little different
class Vgg19(K.Model):
    def __init__(self, trainable=False):
        super(Vgg19, self).__init__(name='Vgg19')
        vgg_pretrained_features = VGG19(weights='imagenet', include_top=False)
        if trainable is False:
            vgg_pretrained_features.trainable = False
        vgg_pretrained_features = vgg_pretrained_features.layers
        self.slice1 = K.Sequential()
        self.slice2 = K.Sequential()
        self.slice3 = K.Sequential()
        self.slice4 = K.Sequential()
        self.slice5 = K.Sequential()
        for x in range(1, 2):
            self.slice1.add(vgg_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add(vgg_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add(vgg_pretrained_features[x])
        for x in range(8, 13):
            self.slice4.add(vgg_pretrained_features[x])
        for x in range(13, 18):
            self.slice5.add(vgg_pretrained_features[x])

    def call(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class ReflectionPad2d(layers.Layer):
    def __init__(self, paddings=(1, 1), **kwargs):
        self.paddings = tuple(paddings)
        self.input_spec = [layers.InputSpec(ndim=4)]
        super(ReflectionPad2d, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1]+self.paddings[0], s[2]+self.paddings[1], s[3])

    def call(self, x):
        w_pad, h_pad = self.paddings
        x = tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')
        return x


class Tanh(layers.Layer):
    def __init__(self, **kwargs):
        super(Tanh, self).__init__(**kwargs)

    def call(self, x):
        return K.activations.tanh(x)


class Sigmoid(layers.Layer):
    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)

    def call(self, x):
        return tf.keras.activations.sigmoid(x)


class InstanceNorm(layers.Layer):
    def __init__(self,
                 scale=True,
                 center=False,
                 gamma_initializer=None,
                 beta_initializer=None,
                 epsilon=1e-6,
                 trainable=True):
        super(InstanceNorm, self).__init__()
        self.scale = scale
        self.center = center
        self.gamma_initializer = gamma_initializer
        self.beta_initializer = beta_initializer
        self.trainable = trainable
        self.epsilon = epsilon

        if gamma_initializer is None:
            self.gamma_initializer = tf.ones_initializer()
        if beta_initializer is None:
            self.beta_initializer = tf.zeros_initializer()

    def build(self, input_shape):
        # input_shape = tensor_shape.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank:', input_shape)
        ndims = len(input_shape)
        assert ndims > 0

        reduction_axis = ndims - 1
        moments_axes = list(range(ndims))
        del moments_axes[reduction_axis]
        del moments_axes[0]
        self.moments_axes = tf.convert_to_tensor(moments_axes)
        param_shape = input_shape[reduction_axis:reduction_axis+1]

        if self.scale:
            self.gamma = self.add_weight(
              name='gamma',
              shape=param_shape,
              initializer=self.gamma_initializer,
              trainable=self.trainable)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                        name='beta',
                        shape=param_shape,
                        initializer=self.beta_initializer,
                        trainable=self.trainable)
        else:
            self.beta = None

    def call(self, inputs):
        # Calculate the moments
        mean, variance = tf.nn.moments(inputs, self.moments_axes, keepdims=True)

        # Compute instance normalization
        outputs = tf.nn.batch_normalization(
            inputs, mean, variance, self.beta, self.gamma, self.epsilon)
        return outputs



if __name__ == '__main__':
    test_inp = tf.random.normal((128, 128, 3))
    # res_block = ResnetBlock(3, 'CONSTANT', get_norm_layer('batch'))
    # res_out = res_block(test_inp[tf.newaxis, ...])
    # print(res_out.shape)
    #
    # print(tf.autograph.to_code(GlobalGenerator(3, 3).call.python_function, experimental_optional_features=None))
    glob_g = GlobalGenerator(3, 3, norm_layer=InstanceNormalization)
    glob_out = glob_g(test_inp[tf.newaxis, ...])
    print(glob_out)
    #
    # local_e = LocalEnhancer(3)
    # local_out = local_e(test_inp[tf.newaxis, ...])
    # print(local_e.layers)
    #
    # nlayer_disc = NLayerDiscriminator()
    # nlayer_disc_out = nlayer_disc(test_inp[tf.newaxis, ...])
    # print(nlayer_disc_out.shape)
    #
    # multi_disc = MultiscaleDiscriminator(3)
    # multi_disc_out = multi_disc(test_inp[tf.newaxis, ...])
    # print(multi_disc_out)

    # vgg19 = Vgg19()
    # vgg_out = vgg19(test_inp[tf.newaxis, ...])
    # print(len(vgg_out))
    # ins_norm = InstanceNorm()
    # ins_out = ins_norm(test_inp)
    # print(ins_out.shape)
