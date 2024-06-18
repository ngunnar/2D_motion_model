import tensorflow as tf
import tensorflow_probability as tfp

from voxelmorph.tf.layers import SpatialTransformer as SpatialTransformer
from voxelmorph.tf.layers import SpatialTransformer as SpatialTransformer
from voxelmorph.tf.layers import VecInt, RescaleTransform
from voxelmorph.tf.losses import Grad, NCC, MSE

from .layers.networks import Encoder, Decoder, SpatialEncoder
from .layers.augmentation_layer import AugmentationLayer
from .layers.gaussian_smoother import GaussianSmoothing2D

tfd = tfp.distributions
tfk = tf.keras
tfpl = tfp.layers

class BaseModel(tfk.Model):
    def __init__(self, config, name="base_model", **kwargs):
        super(BaseModel, self).__init__(name=name, **kwargs)
        self.augmentation_layer = AugmentationLayer(config.use_aug)
        self.output_dist = tfpl.IndependentNormal(config.dim_y, name='output_dist')
        sigma = 0.035 # TODO
        self.y_sigma = lambda y: tf.ones_like(y, dtype='float32') * tfp.math.softplus_inverse(sigma)

        self.dropout_prob = 0.25 #TODO
        self.reg_factor = 0.05 #TODO

        self.grad_loss = Grad(penalty='l2')
        
        self.grad_flow_metric = tfk.metrics.Mean(name = 'grad flow ↓')
        self.loss_metric = tfk.metrics.Mean(name = 'loss ↓')

        self.log_py_metric = tfk.metrics.Mean(name = 'log p(y|x) ↑')
        self.kld_metric = tfk.metrics.Mean(name = 'KLD ↓')
        self.elbo_metric = tfk.metrics.Mean(name ='ELBO ↑')        
        self.ssim_metric = tfk.metrics.Mean(name = 'ssim ↑')
        self.mse_metric = tfk.metrics.Mean(name = 'MSE ↓')
        self.ncc_metric = tfk.metrics.Mean(name = 'NCC ↑')


    def parse_inputs(self, inputs):
        y = inputs['input_video']
        y0 = inputs['input_ref']
        mask = inputs['input_mask']
        return y, y0, mask
    
    def call(self, inputs, training):
        y, y0, mask = self.parse_inputs(inputs)
        if self.config.use_aug:
            y, y0 = self.augmentation_layer([y, y0], training)

        outs = self.forward(y, y0, mask, training)
        self.set_loss(y, y0, mask, outs)
        return 

    def update_matrices(self, y, p_y, mask_ones, length, log_p_y, elbo, kld):
        y_true = tf.reshape(y, (-1, *y.shape[-2:], 1)) # (bs*length, *dim_y, 1)
        y_pred = p_y.sample()
        y_pred = tf.reshape(y_pred, (-1, *y.shape[-2:], 1)) # (bs*length, *dim_y, 1)
        
        ncc = self.sim_metric(y_true, y_pred, mask_ones, NCC().ncc, length, True, True)
        mse = self.sim_metric(y_true, y_pred, mask_ones, MSE().mse, length, True, True)
        ssim_m = lambda t, p: tf.image.ssim(p, t, max_val=tf.reduce_max(t), filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, return_index_map=True)
        ssim = self.sim_metric(y_true, y_pred, mask_ones, ssim_m, length, False, False)
        
        self.mse_metric.update_state(mse)
        self.ncc_metric.update_state(ncc)
        self.ssim_metric.update_state(ssim)
        self.log_py_metric.update_state(log_p_y)
        self.elbo_metric.update_state(elbo)
        self.kld_metric.update_state(kld)

    def forward(self, y, y0, mask, training):
        raise NotImplementedError
    
    def set_loss(self, y, y_0, mask, outs):
        raise NotImplementedError
    
    def compile(self, num_batches, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, jit_compile=None, **kwargs):        
        optimizer = tf.keras.optimizers.Adam(self.config.init_lr)

        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, **kwargs)

class SimpleVAE(BaseModel):
    def __init__(self, 
                 config,
                 length,
                 name="vae",
                 include_sfeats = False,
                 include_y0 = False,
                 decoder_out_channels = 1,
                 output_activation = 'sigmoid',
                 **kwargs):
        super(SimpleVAE, self).__init__(config, name=name, **kwargs)        
        self.config = config
        self.encoder = Encoder(config, length, self.dropout_prob, self.reg_factor, include_y0=include_y0) 
        dec = Decoder(config, length, self.dropout_prob, self.reg_factor, include_sfeats=include_sfeats)
        cnn = tfk.layers.TimeDistributed(tfk.layers.Conv2D(filters=decoder_out_channels,
                                        kernel_size=(3,3),
                                        padding='same',
                                        kernel_initializer=tfk.initializers.RandomNormal(mean=0.0, stddev=1e-5),
                                        activation= output_activation), name='cnn_out')
        self.decoder = tfk.Model(inputs = dec.inputs, outputs = cnn(dec.output), name='decoder')

        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(self.config.dim_x), scale=1.), 
                                reinterpreted_batch_ndims=1)
        
        self.log_py_metric = tfk.metrics.Mean(name = 'log p(y|x) ↑')
        self.kld_metric = tfk.metrics.Mean(name = 'KLD ↓')
        self.elbo_metric = tfk.metrics.Mean(name ='ELBO ↑')
    
    def forward(self, y, y0, mask, training=None):        
        q_x = self.encoder(y, training)           
        x = q_x.sample()        
        y_mu = self.decoder(x)[...,0]

        y_mu = tf.reshape(y_mu, (-1, y.shape[1], y.shape[2] * y.shape[3]))
        y_sigma = self.y_sigma(y_mu)
        p_y =  self.output_dist(tf.concat([y_mu, y_sigma], axis=-1))
        return p_y, q_x, x
    
    def set_loss(self, y, y0, mask, outs):
        p_y = outs[0]
        q_x = outs[1]

        length = y.shape[1]

        mask_ones = tf.cast(mask == False, dtype='float32') # (bs, length)    
        log_p_y = p_y.log_prob(y) # (bs, length)        
        log_p_y = tf.multiply(log_p_y, mask_ones) 
        log_p_y = tf.reduce_sum(log_p_y, axis=1)

        kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
        kld = kl(q_x.sample(), self.prior.sample(q_x.shape[1])) #(bs, length)
        kld = tf.multiply(kld, mask_ones)
        kld = tf.reduce_sum(kld, axis=1)

        elbo = log_p_y - kld
        loss = -log_p_y + kld
        
        # LOSS
        self.add_loss(tf.reduce_mean(loss)) # on batch level

        # METRICES
        self.update_matrices(y, p_y, mask_ones, length, log_p_y, elbo, kld)

    def eval(self, inputs):
        y = inputs['input_video']
        mask = inputs['input_mask'] 
        p_y, q_x, x = self.forward(y, False)
        y_vae = p_y.sample()

        return {'image_data': {'vae': {'images' : y_vae}},
                'x_obs': x}

    def sample(self, y):
        x = self.prior((tf.shape(y)[0:2]))
        p_y = self.decoder(x, training=False)
        return p_y


class DisplacementVAE(SimpleVAE):
    def __init__(self, 
                 config,
                 length,
                 name="vae",                 
                 include_sfeats = False,
                 include_y0 = True,
                 decoder_out_channels = 2,      
                 **kwargs):
        super(DisplacementVAE, self).__init__(config, 
                                              length,
                                              name=name,
                                              include_y0=include_y0,
                                              decoder_out_channels=decoder_out_channels,
                                              include_sfeats=include_sfeats,
                                              output_activation = None,
                                              **kwargs)                

        if config.spatial_smoothing > 0:
            self.gaussian_kernel = GaussianSmoothing2D(15, config.spatial_smoothing, channels=2, trainable=False)
        self.stn = SpatialTransformer()
        self.warp = tf.keras.layers.Lambda(lambda x: self.warping(x), name='warping')

        if self.config.int_steps > 0:            
            self.vecInt = VecInt(method='ss', name='s_flow_int', int_steps=self.config.int_steps)

        if len(config.enc_filters) > len(config.dec_filters):
            rescale_factor = 2**(len(config.enc_filters) - len(config.dec_filters))
            self.rescale = RescaleTransform(rescale_factor, name=f'{name}_svf_resize')

    
    def forward(self, y, y0, mask, training=None):        
        q_x = self.encoder([y, y0], training)           
        x = q_x.sample()        
        phi = self.get_phi(x, y0)
        y_mu = self.warp([phi, y0])
        y_mu = tf.reshape(y_mu, (-1, y.shape[1], y.shape[2] * y.shape[3]))
        y_sigma = self.y_sigma(y_mu)
        p_y =  self.output_dist(tf.concat([y_mu, y_sigma], axis=-1))
        return p_y, q_x, x, phi
    
    def get_phi(self, x, y0):
        dim_y = y0.shape[1:3]        
        phi = self.decoder(x)
        if self.config.spatial_smoothing > 0:
            phi = self.gaussian_kernel(phi)
        
        if self.config.int_steps > 0:
            phi = self.diff_steps(phi)

        if phi.shape[2] < y0.shape[1]:            
            length = phi.shape[1]
            phi = tf.reshape(phi, (-1, phi.shape[2], phi.shape[3], 2))
            phi = self.rescale(phi)
            phi = tf.reshape(phi, (-1, length, *dim_y, 2))
        return phi
    
    def diff_steps(self, phi):
        dim_y = phi.shape[2:4]
        length = phi.shape[1]
        phi = tf.reshape(phi, (-1, *dim_y, 2))
        phi= self.vecInt(phi)
        phi = tf.reshape(phi, (-1, length, *dim_y, 2))
        return phi
    
    def warping(self, inputs):
        phi = inputs[0]
        y_0 = inputs[1]
        _, length, dim_y, _, _ = phi.shape
        y_0 = tf.repeat(y_0[:,None,...], length, axis=1)
        images = tf.reshape(y_0, (-1, *(dim_y,dim_y), 1))

        flows = tf.reshape(phi, (-1, *(dim_y,dim_y), 2))
        y_pred = self.stn([images, flows])
        y_pred = tf.reshape(y_pred, (-1, length, *(dim_y,dim_y)))
        return y_pred

    def eval(self, inputs):
        y = inputs['input_video']
        mask = inputs['input_mask'] 
        p_y, q_x, x = self.forward(y, False)
        y_vae = p_y.sample()

        return {'image_data': {'vae': {'images' : y_vae}},
                'x_obs': x}

    def sample(self, y):
        x = self.prior((tf.shape(y)[0:2]))
        p_y = self.decoder(x, training=False)
        return p_y
    
class UnetVAE(DisplacementVAE):
    def __init__(self, 
                 config,
                 length,
                 name="vae",
                 include_sfeats = True,
                 include_y0 = True,
                 decoder_out_channels = 2,
                 **kwargs):
        super(UnetVAE, self).__init__(config, 
                                      length,
                                      name=name,
                                      include_y0=include_y0,
                                      decoder_out_channels=decoder_out_channels,
                                      include_sfeats=include_sfeats,
                                      **kwargs)   
        
        self.s_encoder = SpatialEncoder(config, self.dropout_prob)

        
        
    def forward(self, y, y0, mask, training=None):        
        q_x = self.encoder([y, y0], training)
        s_feats = self.s_encoder(y0)
        x = q_x.sample()        
        phi = self.get_phi([x, s_feats], y0)
        y_mu = self.warp([phi, y0])
        y_mu = tf.reshape(y_mu, (-1, y.shape[1], y.shape[2] * y.shape[3]))
        y_sigma = self.y_sigma(y_mu)
        p_y =  self.output_dist(tf.concat([y_mu, y_sigma], axis=-1))
        return p_y, q_x, x, phi, s_feats