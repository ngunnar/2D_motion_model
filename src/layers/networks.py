import tensorflow as tf
import tensorflow_probability as tfp
import keras

layers = keras.layers
tfk = tf.keras
tfpl = tfp.layers
tfd = tfp.distributions

def conv_layer(filters, kernel_size=(3,3), strides = (1,1)):
    return layers.Conv2D(filters=filters,
                         activation='elu',
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same')

class Model(keras.Model):
    def __init__(self, config, length, name="model"):
        inputs_y = layers.Input(shape=(length, *config.dim_y), name='input_y')
        inputs_y0 = layers.Input(shape=config.dim_y, name='input_y0')
        
        x = Encoder(config, length, 0, 0, name="Encoder")([inputs_y, inputs_y0])
        s_feats = SpatialEncoder(config, 0, name='SpatialEncoder')(inputs_y0)

        phi = Decoder(config, length, 0, 0, name="Decoder")([x, s_feats])
        
        return super().__init__(inputs=[inputs_y, inputs_y0], outputs=phi, name=name)


class RepeatLayer(layers.Layer):
    def __init__(self, n, **kwargs):
        super(RepeatLayer, self).__init__(**kwargs)
        self.n = n

    def call(self, inputs):
        return tf.tile(inputs, [1, self.n, 1,1,1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n, input_shape[1])


def get_fc_block(length, dim_x, h, w, c, name='fc_block', dropout_prob=0.0, reg_factor=0.01):
    return tfk.Sequential([
                layers.InputLayer(input_shape=(length, dim_x)),
                layers.TimeDistributed(layers.Dense(h*w*c, 
                use_bias=True,
                bias_initializer='zeros',                                
                bias_regularizer=tfk.regularizers.l2(reg_factor),
                kernel_initializer='glorot_uniform',
                kernel_regularizer=tfk.regularizers.l2(reg_factor),
                activity_regularizer=tfk.regularizers.l2(reg_factor),
                name = name + '_dense')),
        tf.keras.layers.Dropout(dropout_prob, name=name + '_dropout'),
        layers.Reshape((length, h,w,c), name=name + '_reshape')            
    ], name=name)

class AttentionGateBlock(tfk.Model):
    def __init__(self, length, y_dim, level, filters_s_in, filters_x_in, filters_out, name='attention_block'):
        x_inputs = layers.Input(shape=(length,
                                     y_dim[0]//(2**level),
                                     y_dim[1]//(2**level),
                                     filters_x_in))
        s_feat_inputs = layers.Input(shape=(length,
                                          y_dim[0]//(2**(level-1)),
                                          y_dim[1]//(2**(level-1)),
                                           filters_s_in)) 
        
        phi_g = layers.TimeDistributed(conv_layer(filters_out), name='phi_g')(x_inputs)
        theta_x = layers.TimeDistributed(conv_layer(filters_out, strides=(2,2)), name='theta_x')(s_feat_inputs)
        add_xg = layers.TimeDistributed(layers.Activation('relu'))(phi_g + theta_x)

        psi = layers.TimeDistributed(conv_layer(1, kernel_size=(1,1)), name='psi')(add_xg)
        psi = layers.TimeDistributed(layers.UpSampling2D(size=(2,2)), name='psi_up')(psi)
        attn_coefficients = layers.Lambda(lambda x: tf.repeat(x, filters_out, axis=-1))(psi)
        x = layers.TimeDistributed(layers.Multiply(), name='multiply')([attn_coefficients, s_feat_inputs])
        return super().__init__(inputs=[x_inputs, s_feat_inputs], outputs=x, name=name)    

class DecoderMiniBlock(tfk.Model):
    def __init__(self, length, x_filters_in, x_skip_filters_in, filters, y_dim, level, include_xskip = True, name='up_block'):
        x_up_input = layers.Input(shape=(length,
                                       y_dim[0]//(2**level),
                                       y_dim[1]//(2**level),
                                       x_filters_in))
        
        x = layers.TimeDistributed(layers.Conv2DTranspose(input_shape=x_up_input.shape,
                                    filters=filters,
                                    activation='elu',
                                    kernel_size=(3,3),
                                    padding='same',
                                    strides = (2,2), 
                                    name= name + '_up'))(x_up_input)
        
        if include_xskip:
            x_skip_input = layers.Input(shape=(length,
                                               y_dim[0]//(2**(level-1)),
                                               y_dim[1]//(2**(level-1)),
                                               x_skip_filters_in))
            x = layers.TimeDistributed(layers.Concatenate(axis=-1))([x, x_skip_input])

        x = layers.TimeDistributed(conv_layer(filters))(x)
        x = layers.TimeDistributed(conv_layer(filters))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        if include_xskip:
            return super().__init__(inputs=[x_up_input, x_skip_input], outputs=x, name=name)
        return super().__init__(inputs=x_up_input, outputs=x, name=name)

class Decoder(tfk.Model):
    def __init__(self, config, length, dropout_prob, reg_factor, include_sfeats=True, name='Decoder', **kwargs):        
        inputs_x = layers.Input(shape=(length, config.dim_x), name='x')
        if include_sfeats:
            inputs_sfeats = [layers.Input(shape=(1,
                                                 config.dim_y[0]//(2**i),
                                                 config.dim_y[1]//(2**i),
                                                 f), name=f's_{i+1}') for i, f in enumerate(config.enc_filters)]
            
        h = int(config.dim_y[0] / (2**(len(config.enc_filters))))
        w = int(config.dim_y[1] / (2**(len(config.enc_filters))))
        self.out_dim = (h*2**(len(config.dec_filters)), w*2**(len(config.dec_filters)))
        x = get_fc_block(length,
                         config.dim_x,
                         h,w,config.dim_x, 
                         reg_factor = reg_factor,
                         dropout_prob=dropout_prob, 
                         name='{0}_fc_block'.format(name))(inputs_x)
        

        x = layers.TimeDistributed(conv_layer(config.dim_x), name='conv1')(x)
        x = layers.TimeDistributed(conv_layer(config.dec_filters[-1]), name='conv2')(x)
        if include_sfeats:
            sfeat = RepeatLayer(length, name=f'repeat_s{len(inputs_sfeats)}')(inputs_sfeats[-1])
            x_feat = AttentionGateBlock(length,
                                        config.dim_y,
                                        len(config.dec_filters),
                                        config.enc_filters[-1], 
                                        config.dec_filters[-1],
                                        config.dec_filters[-1],
                                        name=f'AttentionBlock_{len(inputs_sfeats)}')([x, sfeat])
            x = [x, x_feat]
        x = DecoderMiniBlock(length,
                             config.dec_filters[-1], 
                             config.dec_filters[-1],
                             config.dec_filters[-1],
                             config.dim_y,
                             len(config.dec_filters),
                             include_xskip=include_sfeats,
                             name=f'DecoderBlock_{len(config.dec_filters)}')(x)
        k = 1
        for i in reversed(range(1, len(config.dec_filters))):
            if include_sfeats:
                sfeat = RepeatLayer(length, name=f'repeat_s{i}')(inputs_sfeats[-(k+1)])
                x_feat = AttentionGateBlock(length,
                                            config.dim_y,
                                    len(config.dec_filters)-k,
                                    config.enc_filters[i-1], 
                                    config.dec_filters[i],
                                    config.dec_filters[i-1],
                                    name=f'AttentionBlock_{i}')([x, sfeat])
                x = [x, x_feat]
            x = DecoderMiniBlock(length,
                                 config.dec_filters[i],
                                 config.dec_filters[i-1],
                                 config.dec_filters[i-1],
                                 config.dim_y,
                                 len(config.dec_filters)-k,
                                 include_xskip=include_sfeats,
                                 name=f'DecoderBlock_{i}')(x)
            k += 1
        
        if include_sfeats:
            return super().__init__(inputs=[inputs_x, inputs_sfeats], outputs=x, name=name)
        return super().__init__(inputs=inputs_x, outputs=x, name=name)

class DownBlock(tfk.Model):
    def __init__(self, length, y_dim, level, filters_in, filters, dropout, name='DownBlock'):
        inputs = layers.Input(shape=(length, 
                                   y_dim[0]//(2**level),
                                   y_dim[1]//(2**level),
                                   filters_in))
        x = layers.TimeDistributed(conv_layer(filters))(inputs)
        x = layers.TimeDistributed(layers.Dropout(dropout))(x)
        x = layers.TimeDistributed(conv_layer(filters))(x)
        x = layers.TimeDistributed(layers.Dropout(dropout))(x)
        x = layers.TimeDistributed(layers.LeakyReLU(0.2, name = name + '_leakyRelu2'))(x)
        x = layers.TimeDistributed(layers.BatchNormalization(momentum=0.99, epsilon=0.001, name = name + '_bn2'))(x) # TODO
        down_x = layers.TimeDistributed(conv_layer(filters, strides=(2,2)))(x)
        
        return super().__init__(inputs=inputs, outputs=[x, down_x], name=name)
class Downsampling(tfk.Model):
    def __init__(self, length, y_dim, channels, filters, dropout_prob, name="Downsampling"):
        y_inputs = layers.Input(shape=(length, *y_dim, channels))
        
        x_feat, x = DownBlock(length, y_dim, 0, channels, filters[0], dropout_prob, f'DownBlock_{0}')(y_inputs)
        feats = [x_feat]
        for i in range(1, len(filters)):
            x_feat, x = DownBlock(length, y_dim, i, filters[i-1], filters[i], dropout_prob, f'DownBlock_{i}')(x)
            feats.append(x_feat)

        return super().__init__(inputs=y_inputs, outputs=[x, feats], name=name)

class Encoder(tfk.Model):
    def __init__(self, config, length, dropout_prob, reg_factor, include_y0=True, name="Encoder"):        
        inputs_y = layers.Input(shape=(length, *config.dim_y), name='input_y')
        y = layers.Lambda(lambda y: tf.expand_dims(y, axis=-1), name='expand_channel_y')(inputs_y)
        channels = 1
        if include_y0:
            inputs_y0 = layers.Input(shape=config.dim_y, name='input_y0')        
            y0 = layers.Lambda(lambda y: tf.expand_dims(y, axis=1), name='expand_channel_y0')(inputs_y0)
            y0 = layers.Lambda(lambda y: tf.expand_dims(y, axis=-1), name='expand_length_y0')(y0)        

            Y0 = RepeatLayer(length, name='y0_repeat')(y0)
            y = layers.Concatenate(axis=-1, name='y0y_merge')([Y0, y])
            channels = 2

        x, _ = Downsampling(length, config.dim_y, channels, config.enc_filters, dropout_prob)(y)
        x = layers.Reshape((length, -1), name='reshape')(x)

        use_dist = tfpl.MultivariateNormalTriL
        activity_regularizer = None 

        x = layers.TimeDistributed(layers.Dense(use_dist.params_size(config.dim_x),
                                kernel_regularizer=tfk.regularizers.l2(reg_factor),      
                                bias_regularizer=tfk.regularizers.l2(reg_factor),
                                activation=None), name='dense')(x) 
        x = use_dist(config.dim_x, 
                     activity_regularizer = activity_regularizer, 
                     name='encoder_dist')(x)

        if include_y0:
            return super().__init__(inputs=[inputs_y,inputs_y0], outputs=x, name=name)        
        else:
            return super().__init__(inputs=inputs_y, outputs=x, name=name)
    
    
class SpatialEncoder(tfk.Model):
    def __init__(self, config, dropout_prob, name="SpatialEncoder"):
        inputs_y0 = layers.Input(shape=config.dim_y, name='input_y0')
        y0 = layers.Lambda(lambda y: tf.expand_dims(y, axis=1), name='expand_length')(inputs_y0)
        y0 = layers.Lambda(lambda y: tf.expand_dims(y, axis=-1), name='expand_channels')(y0)
                         
        _, s = Downsampling(1, config.dim_y, 1, config.enc_filters, dropout_prob)(y0)
        super().__init__(inputs=inputs_y0, outputs=s, name=name)  
