import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions

class GaussianSmoothing2D(tf.keras.Model):
    def __init__(self, kernel_size, std, trainable, name='smoothing2D_layer', channels=2, **kwargs):
        super(GaussianSmoothing2D, self).__init__(self, name = name, **kwargs)
        init_s_std = tf.constant_initializer(np.float32(std))
        self.s_std = tf.Variable(initial_value=init_s_std(shape=(1,)),
                             trainable=trainable, dtype="float32", name="s_std")
        self.kernel_size = kernel_size
        gk = self.create_gaussian_kernel(self.kernel_size, 0.0, self.s_std)[:,:,None, None]
        self.gk = tf.tile(gk, multiples=[1,1,channels,1]) 

        self.gaussian_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Lambda(lambda img: tf.nn.depthwise_conv2d(img, self.gk, strides=[1,1,1,1], padding='SAME')))

    def create_gaussian_kernel(self, size, mean, std):
        """Generate a Gaussian kernel."""
        d = tfd.Normal(mean, std)
        vals = d.prob(tf.range(start = -size, limit = size + 1, dtype=tf.float32))
        kernel = tf.einsum('i,j->ij', vals, vals)
        return kernel / tf.reduce_sum(kernel)

    def call(self, image):        
        filtered_image = self.gaussian_layer(image)
        return filtered_image