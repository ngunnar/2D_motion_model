import tensorflow as tf
tfk = tf.keras

class AugmentationLayer(tfk.Model):
    def __init__(self, aug_config, name='augmentation_layer', **kwargs):
        super(AugmentationLayer, self).__init__(self, name = name, **kwargs)
        if aug_config is not None:
            self.add_flip=aug_config['add_flip']
            self.add_rot=aug_config['add_rot']
            self.add_noise=aug_config['add_noise']            
            self.add_zoom=aug_config['add_zoom']
            self.add_trans=aug_config['add_trans']
            self.add_noise_at_zeros=aug_config['add_noise_at_zeros']
        
        self.random_flip = tfk.layers.RandomFlip()
        self.random_rotation = tfk.layers.RandomRotation((-1.0, 1.0), fill_mode='nearest')
        self.random_noise = tfk.layers.GaussianNoise(0.1)
        self.random_zoom = tf.keras.layers.RandomZoom((-0.5,0.2),(-0.5,0.2))
        self.random_trans = tf.keras.layers.RandomTranslation((-0.1,0.1),(-0.1,0.1))

    def call(self, inputs, training):
        y = inputs[0]
        y0 = inputs[1]
        if training == True:
            y0y = tf.concat([y0[:,None,...],y], axis=1) # (bs, 1+length, h, w)
            y0y = tf.transpose(y0y, perm=[0,2,3,1]) # (bs, 1+length, h, w) ->  (bs, h, w, 1+length)
            if self.add_flip:
                y0y = self.random_flip(y0y)
            if self.add_rot:
                y0y = self.random_rotation(y0y)
            if self.add_noise:
                y0y = self.random_noise(y0y)
            if self.add_zoom:
                y0y = self.random_zoom(y0y)
            if self.add_trans:
                y0y = self.random_trans(y0y)
            
            y0 = y0y[...,0] # (bs, h, w)
            y = y0y[...,1:] # (bs, h, w, length)
            y = tf.transpose(y, perm=[0,3,1,2]) # (bs, h, w, length) ->  (bs, length, h, w)

        if self.add_noise_at_zeros:
            y0y = tf.concat([y0[:,None,...],y], axis=1) # (bs, 1+length, h, w)
            y0y = tf.transpose(y0y, perm=[0,2,3,1]) # (bs, 1+length, h, w) ->  (bs, h, w, 1+length)
            sum_all = tf.reduce_sum(y0y, axis=-1) # (bs, h, w)
            noise = self.random_noise(tf.zeros_like(sum_all), True) * tf.cast(sum_all == 0, tf.float32)
            noise = tf.tile(noise[...,None], [1,1,1,y0y.shape[-1]])
            y0y = y0y + noise
            y0 = y0y[...,0] # (bs, h, w)
            y = y0y[...,1:] # (bs, h, w, length)
            y = tf.transpose(y, perm=[0,3,1,2]) # (bs, h, w, length) ->  (bs, length, h, w)
        return y, y0