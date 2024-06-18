import tensorflow as tf
import tensorflow_probability as tfp

from voxelmorph.tf.losses import Grad
from voxelmorph.tf.losses import NCC, MSE

tfk = tf.keras
tfpl = tfp.layers
tfd = tfp.distributions

from .vae import UnetVAE
from .lgssm import LGSSM

class fKVAE(UnetVAE):
    def __init__(self, config, length, name="fKVAE", **kwargs):
        super(fKVAE, self).__init__(length=length, name=name, config=config, **kwargs)
        self.lgssm = LGSSM(config)
                
        # Variables
        self.init_w_kf = tf.Variable(initial_value=config.kf_loss_weight, trainable=False, dtype="float32", name="init_w_kf")
        self.w_kf = tf.Variable(initial_value=config.kf_loss_weight, trainable=False, dtype="float32", name="w_kf")

        self.init_w_recon = tf.Variable(initial_value=config.scale_reconstruction, trainable=False, dtype="float32", name="init_w_kf")
        self.w_recon = tf.Variable(initial_value=config.scale_reconstruction, trainable=False, dtype="float32", name="w_kf")

        self.init_w_kl = tf.Variable(initial_value=config.kl_latent_loss_weight, trainable=False, dtype="float32", name="init_w_kf")
        self.w_kl = tf.Variable(initial_value=config.kl_latent_loss_weight, trainable=False, dtype="float32", name="w_kf")

        self.w_g = tf.Variable(initial_value=1., trainable=False, dtype="float32", name="w_g")
        
        # Metrics
        self.log_qx_metric = tfk.metrics.Mean(name = 'log q(x[t]|y[t]) ↓')
        self.log_pzx_metric = tfk.metrics.Mean(name = 'log p(z[t],x[t]) ↑')
        self.log_pz_x_metric = tfk.metrics.Mean(name = 'log p(z[t]|x[t]) ↓')
        self.log_px_x_metric = tfk.metrics.Mean(name = 'log p(x[t]|x[:t-1]) ↑')        
        
    def forward(self, y, y0, mask, training):
        q_x = self.encoder([y, y0], training)        
        s_feats = self.s_encoder(y0)
        
        x = q_x.sample()
        log_pred, log_filt, log_p_1, log_smooth, ll = self.lgssm([x, mask])

        phi = self.get_phi([x, s_feats], y0)
        y_mu = self.warp([phi, y0])
        y_mu = tf.reshape(y_mu, (-1, y.shape[1], y.shape[2] * y.shape[3]))
        y_sigma = self.y_sigma(y_mu)
        p_y =  self.output_dist(tf.concat([y_mu, y_sigma], axis=-1))
        return p_y, q_x, x, phi, s_feats, log_pred, log_filt, log_p_1, log_smooth, ll

    def set_loss(self, y, y_0, mask, out):
        p_y, q_x, x, phi, s_feats, log_pred, log_filt, log_p_1, log_smooth, ll = out
        length = y.shape[1]
        mask_ones = tf.cast(mask == False, dtype='float32')

        # log p(y|x)        
        log_p_y_x = self.get_log_dist(p_y, y, mask_ones, True)    
        
        # log q(x|y)       
        log_q_x_y = self.get_log_dist(q_x, x, mask_ones, True)
        
        # log p(x, z)
        log_p_xz = tf.reduce_sum(log_filt, axis=1) + log_p_1 + tf.reduce_sum(log_pred, axis=1)
        
        # log p(z|x)
        log_p_z_x = tf.reduce_sum(log_smooth, axis=1)

        # Image simularity metrics
        y_true = tf.reshape(y, (-1, *y.shape[-2:], 1)) # (bs*length, *dim_y, 1)
        if ('ncc' in self.config.losses) or ('mse' in self.config.losses):
            y_pred = p_y.mean()
        else:
            y_pred = p_y.sample()
        y_pred = tf.reshape(y_pred, (-1, *y.shape[-2:], 1)) # (bs*length, *dim_y, 1)        

        if 'kvae_loss' in self.config.losses:
            if 'ncc' in self.config.losses:
                recon = self.sim_metric(y_true, y_pred, mask_ones, NCC().ncc, length, True, True)
            elif 'mse' in self.config.losses:
                recon = -self.sim_metric(y_true, y_pred, mask_ones, MSE().mse, length, True, True)  
            else:
                recon = log_p_y_x

            loss = -(self.w_recon * recon - self.w_kl*log_q_x_y + self.w_kf * (log_p_xz - log_p_z_x))
            self.add_loss(tf.reduce_mean(loss))            
        
        if 'lgssm_ml' in self.config.losses:                        
            loss = -tf.reduce_mean(ll, axis=1)
            self.add_loss(tf.reduce_mean(loss))
        
        self.loss_metric.update_state(loss)
        self.add_metric(self.w_kl, "W_kl")
        self.add_metric(self.w_kf, "W_kf")
        self.add_metric(self.w_recon, "W_recon")
        
        # METRICES
        ## ELBO
        kld = log_q_x_y - log_p_xz + log_p_z_x
        elbo = log_p_y_x - kld
        self.elbo_metric.update_state(elbo)

        ## Image sequence
        self.update_matrices(y, p_y, mask_ones, length, log_p_y_x, elbo, kld)
        
        ## Latent space
        self.log_qx_metric.update_state(log_q_x_y)
        self.log_pzx_metric.update_state(log_p_xz)
        self.log_pz_x_metric.update_state(log_p_z_x)        
        self.log_px_x_metric.update_state(tf.reduce_sum(ll, axis=1))

        # G
        grad = self.grad_loss.loss(None, tf.reshape(phi, (-1, *phi.shape[2:])))
        self.grad_flow_metric.update_state(grad)
        if 'grad' in self.config.losses:
            self.add_loss((tf.reduce_mean(self.w_g * grad)))

    @tf.function
    def eval(self, inputs):
        y_org, y0_org, mask = self.parse_inputs(inputs)
        if self.config.use_aug:
            y, y0 = self.augmentation_layer([y_org, y0_org], False) # TODO: how to handle this fór single run?
        else:
            y, y0 = y_org, y0_org
        
        s_feats = self.s_encoder(y0)
        q_x = self.encoder([y, y0], training=False)
              
        x = q_x.sample()
        
        # Latent distributions 
        #p_obssmooth, p_obsfilt, p_obspred = self.lgssm.get_distribtions(x, mask)
        mask_samples = tf.ones((x.shape[0], x.shape[1]), dtype=tf.bool)
        mask_half = tf.concat([tf.zeros((x.shape[0], y_org.shape[1]//2), dtype=tf.bool), tf.ones((x.shape[0], (x.shape[1] - x.shape[1]//2)), dtype=tf.bool)], axis=1)
        mask_full = tf.zeros((x.shape[0], x.shape[1]), dtype=tf.bool)

        x_samples, mu_samples, std_samples = self.lgssm.get_sequence(x, mask=mask_samples)
        x_half, mu_half, std_half = self.lgssm.get_sequence(x, mask=mask_half)
        x_full, mu_full, std_full = self.lgssm.get_sequence(x, mask=mask_full)

        # Flow
        phi_x = self.get_phi([x, s_feats], y0)
        phi_samples = self.get_phi([x_samples, s_feats], y0)
        phi_half = self.get_phi([x_half, s_feats], y0)
        phi_full = self.get_phi([x_full, s_feats], y0)

        # Image distributions and samples
        last_channel = y.shape[2] * y.shape[3]
        y_mu_x = self.warp([phi_x, y0_org])
        y_mu_x = tf.reshape(y_mu_x, (-1, y.shape[1], last_channel))
        y_sigma = self.y_sigma(y_mu_x)
        p_y_x=  self.output_dist(tf.concat([y_mu_x, y_sigma], axis=-1))
        y_x = p_y_x.sample()

        y_mu_samples = self.warp([phi_samples, y0_org])
        y_mu_samples = tf.reshape(y_mu_samples, (-1, y.shape[1], last_channel))
        p_y_samples =  self.output_dist(tf.concat([y_mu_samples, y_sigma], axis=-1))
        y_samples = p_y_samples.sample()

        y_mu_half = self.warp([phi_half, y0_org])
        y_mu_half = tf.reshape(y_mu_half, (-1, y.shape[1], last_channel))
        p_y_half =  self.output_dist(tf.concat([y_mu_half, y_sigma], axis=-1))
        y_half = p_y_half.sample()

        y_mu_full = self.warp([phi_full, y0_org])
        y_mu_full = tf.reshape(y_mu_full, (-1, y.shape[1], last_channel))
        p_y_full =  self.output_dist(tf.concat([y_mu_full, y_sigma], axis=-1))
        y_full = p_y_full.sample()
        
        return {'image_data': {'vae': {'images' : y_x, 'flows': phi_x},
                               'sample': {'images': y_samples, 'flows': phi_samples},
                               'half': {'images': y_half, 'flows': phi_half},
                               'full': {'images': y_full, 'flows': phi_full}},
                'latent_dist': {'sample': [x_samples, mu_samples, std_samples], 
                                'half': [x_half, mu_half, std_half], 
                                'full': [x_full, mu_full, std_full]},
                'x_obs': x,
                's_feat': s_feats}

    def get_log_dist(self, dist, y, mask_ones, sum = True):
        log_dist = dist.log_prob(y)    
        log_dist = tf.multiply(log_dist, mask_ones)
        if sum:
            log_dist = tf.reduce_sum(log_dist, axis=1)
        else:
            log_dist = tf.reduce_mean(log_dist, axis=1)
        return log_dist
    
    def sim_metric(self, y_true, y_pred, mask_ones, metric, length, pixel_sum = True, seq_sum=True):
        val = metric(y_true, y_pred) # (bs*length, *dim_y, 1)        
        if pixel_sum:
            val = tf.reduce_sum(val, axis=tf.range(1, len(val.shape))) # (bs*length)
        else:
            val = tf.reduce_mean(val, axis=tf.range(1, len(val.shape))) # (bs*length)
        val = tf.reshape(val, (-1, length)) # (bs, length)
        val = tf.multiply(val, mask_ones)  # (bs, length)
        if seq_sum:
            return tf.reduce_sum(val, axis=1) # (bs)
        return tf.reduce_mean(val, axis=1) # (bs)

    @tf.function
    def reconstruct_phis(self, y, y0, mask, use_lgssm_samples=False):
        if self.config.use_aug:
            y, y0 = self.augmentation_layer([y, y0], False) # TODO: how to handle this fór single run?

        q_x = self.encoder([y, y0], False)        
        s_feats = self.s_encoder(y0)

        x = q_x.sample()
        self.lgssm.length = x.shape[1]
        z_samples = self.lgssm.lgssm.posterior_sample(x, mask=mask)
        x_samples = tf.matmul(self.lgssm.C, z_samples[...,None])[...,0]
        
        if use_lgssm_samples:
            phis = self.get_phi([x_samples, s_feats], y0)
        else:
            phis = self.get_phi([x, s_feats], y0)
        return phis, x, x_samples
    
    def compile(self, num_batches, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, jit_compile=None, **kwargs):        
        optimizer = tf.keras.optimizers.Adam(self.config.init_lr)

        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, **kwargs)

    @tf.function 
    def online_train_step(self, y, y0, s_feats, x_prev, m_prev):
        
        x = self.encoder([y, y0], False).sample()        
        X = tf.concat([x_prev, x], axis=1)
        mask = tf.concat([m_prev, tf.zeros((1, 1), dtype='bool')], axis=1)

        out, loss_value = self.lgssm.train_step([X,mask])
        phi = self.get_phi([x, s_feats], y0)
        return phi, out, loss_value, X, mask