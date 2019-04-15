import random
import tensorflow as tf


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = tf.TensorArray(tf.float32, size=self.pool_size)

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = tf.TensorArray(tf.float32, size=tf.shape(images)[0])
        return_idx = 0
        for image in images:
            image = tf.expand_dims(image, 0)
            if self.num_imgs < self.pool_size:
                self.images.write(self.num_imgs, image)
                self.num_imgs = self.num_imgs + 1
                return_images.write(return_idx, image)
                return_idx = return_idx + 1
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images.read(random_id)
                    self.images.write(random_id, image)
                    return_images.write(return_idx, tmp)
                    return_idx = return_idx + 1
                else:
                    return_images.write(return_idx, image)
                    return_idx = return_idx + 1
        return_images = tf.squeeze(return_images.stack())
        # return_images = Variable(torch.cat(return_images, 0))c
        return return_images


if __name__ == '__main__':
    # img_pool = ImagePool(2)
    # inp = tf.random.normal((2, 2))
    # print(inp)
    # img_pool.query(inp)
    # print('first query:')
    # print(img_pool.images.stack())
    # print()
    # # print(img_pool.query)
    # for i in range(5):
    #     inp = tf.random.normal((2, 2))
    #     return_images = img_pool.query(inp)
    #     print('before: ', inp)
    #     print('after: ', return_images)

    img_pool = ImagePool(10)
    # print(img_pool.query)
    for i in range(5):
        inp = tf.random.normal((4, 32, 32, 2))
        return_images = img_pool.query(inp)
        print(return_images)
