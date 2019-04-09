import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers

# Functions
weight_init = {}
weight_init['conv'] = tf.random_normal_initializer(0.0, 0.02)
weight_init['bn_gamma'] = tf.random_normal_initializer(1.0, 0.02)
weight_init['bn_beta'] = tf.zeros_initializer()


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = layers.BatchNormalization(gamma_initializer=weight_init['bn_gamma'],
                                               beta_initializer=weight_init['bn_beta'])
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    return norm_layer


class ReflectionPad2d(layers.Layer):
    def __init__(self, paddings):
        super(ReflectionPad2d, self).__init__(name='ReflectionPad2d')
        self.paddings = paddings

    def call(self, x):
        x = tf.pad(x, self.paddings, 'REFLECT')
        return x


class Tanh(layers.Layer):
    def __init__(self):
        super(Tanh, self).__init__(name='Tanh')

    def call(self, x):
        return K.activations.tanh(x)


# Generator
class LocalEnhancer(K.Model):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=get_norm_layer('batch'), padding_type='REFLECT'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        paddings = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])

        # global generator
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).layers[0].layers
        model_global = model_global[:-3] # get rid of final convolution layers
        self.model = K.Sequential(model_global)

        # local enhancer layers
        for n in range(1, n_local_enhancers+1):
            # downsample
            print(n)
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = K.Sequential()
            model_downsample.add(ReflectionPad2d(paddings))
            model_downsample.add(layers.Conv2D(ngf_global, 7, kernel_initializer=weight_init['conv']))
            model_downsample.add(norm_layer)
            model_downsample.add(layers.ReLU())
            model_downsample.add(layers.Conv2D(ngf_global * 2, 3, strides=2, padding='same', kernel_initializer=weight_init['conv']))
            model_downsample.add(norm_layer)
            model_downsample.add(layers.ReLU())

            # residual blocks
            model_upsample = K.Sequential()
            for i in range(n_blocks_local):
                model_upsample.add(ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer))

            # upsample
            model_upsample.add(layers.Conv2DTranspose(ngf_global, 3, strides=2, padding='same', output_padding=1, kernel_initializer=weight_init['conv']))
            model_upsample.add(norm_layer)
            model_upsample.add(layers.ReLU())

            # final convolution
            if n == n_local_enhancers:
                model_upsample.add(ReflectionPad2d(paddings))
                model_upsample.add(layers.Conv2D(output_nc, 7, kernel_initializer=weight_init['conv']))
                model_upsample.add(Tanh())

            setattr(self, 'model'+str(n)+'_1', model_downsample)
            setattr(self, 'model'+str(n)+'_2', model_upsample)

        self.downsample = layers.AveragePooling2D(3, strides=2, padding='same')

    def call(self, x):

        input_downsampled = [x]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        # output at coarest level
        output_prev = self.model(input_downsampled[-1])
        return self.model(x)

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
        paddings = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])
        activation = layers.ReLU()

        model = K.Sequential()
        model.add(ReflectionPad2d(paddings))
        model.add(layers.Conv2D(ngf, 7, kernel_initializer=weight_init['conv']))
        model.add(norm_layer)
        model.add(activation)

        # downsample
        for i in range(n_downsampling):
            mult = 2**i
            model.add(layers.Conv2D(ngf * mult * 2, 3, strides=2, padding='same', kernel_initializer=weight_init['conv']))
            model.add(norm_layer)
            model.add(activation)

        # resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model.add(ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer))

        # upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model.add(layers.Conv2DTranspose(int(ngf * mult / 2), 3, strides=2, padding='same', output_padding=1, kernel_initializer=weight_init['conv']))
            model.add(norm_layer)
            model.add(activation)
        model.add(ReflectionPad2d(paddings))
        model.add(layers.Conv2D(self.output_nc, 7, kernel_initializer=weight_init['conv']))
        model.add(Tanh())
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
                 activation = layers.ReLU(), use_dropout=False, name='ResnetBlock', **kwargs):
        super(ResnetBlock, self).__init__(name=name, **kwargs)
        self.use_dropout = use_dropout
        self.padding_type = padding_type
        self.dim = dim
        self.norm_layer = norm_layer
        self.activation = activation
        # CONSTANT REFLECT SYMMETRIC

    def call(self, x):
        identity = x
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

        x = tf.pad(x, paddings, self.padding_type)
        x = layers.Conv2D(self.dim, 3, kernel_initializer=weight_init['conv'])(x)
        x = self.norm_layer(x)
        x = self.activation(x)
        if self.use_dropout:
            x = layers.Dropout(0.5)(x)

        x = tf.pad(x, paddings, self.padding_type)
        x = layers.Conv2D(self.dim, 3, kernel_initializer=weight_init['conv'])(x)
        x = self.norm_layer(x)
        out = x + identity

        return out

if __name__ == '__main__':
    test_inp = tf.random.normal((256, 256, 3))
    res_block = ResnetBlock(3, 'CONSTANT', get_norm_layer('batch'))
    res_out = res_block(test_inp[tf.newaxis, ...])
    print(res_out.shape)

    glob_g = GlobalGenerator(3, 3, norm_layer=layers.BatchNormalization(gamma_initializer=weight_init['bn_gamma'], beta_initializer=weight_init['bn_beta']))
    glob_out = glob_g(test_inp[tf.newaxis, ...])
    print(glob_out.shape)

    local_e = LocalEnhancer(3, 3)
    local_out = local_e(test_inp[tf.newaxis, ...])
    print(local_out.shape)
    print(local_e.model1_2.layers)
