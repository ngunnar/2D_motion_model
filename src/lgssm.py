import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfk = tf.keras
tfpl = tfp.layers

def get_cholesky(A):
    L = tfp.experimental.distributions.marginal_fns.retrying_cholesky(A, jitter=None, max_iters=5, name='retrying_cholesky')
    return L[0]

class LGSSM(tfk.Model):
    def __init__(self, config, name='LGSSM', **kwargs):
        super(LGSSM, self).__init__(name=name, **kwargs)
        self.dim_z = config.dim_z
        self.dim_x = config.dim_x
        self.config = config
        
        ## Parameters
        A_init = tf.random_normal_initializer()
        self.A = tf.Variable(initial_value=A_init(shape=(config.dim_z,config.dim_z)),
                             trainable=config.trainable_A, 
                             dtype="float32", 
                             name="A")
        
        C_init = tf.random_normal_initializer()
        self.C = tf.Variable(initial_value=C_init(shape=(config.dim_x, config.dim_z)), 
                             trainable=config.trainable_C, 
                             dtype="float32", 
                             name="C")
        # w ~ N(0,Q)        
        init_Q = tf.constant_initializer(np.eye(config.dim_z, dtype='float32')*config.noise_transition)
        self.Q = tf.Variable(initial_value=init_Q(shape=(config.dim_z,config.dim_z)), trainable=config.trainable_Q, dtype='float32', name="Q")
        # v ~ N(0,R)        
        init_R = tf.constant_initializer(np.eye(config.dim_x, dtype='float32')*config.noise_emission)
        self.R = tf.Variable(initial_value=init_R(shape=(config.dim_x, config.dim_x)), trainable=config.trainable_R, dtype='float32', name="R")
        
        # Initial prior
        sigma_0 = tf.constant_initializer(np.eye(config.dim_z, dtype='float32'))
        self.sigma_0 = tf.Variable(initial_value=sigma_0(shape=(config.dim_z, config.dim_z)), trainable=config.trainable_sigma0, dtype='float32', name="Sigma0")
        mu_0 = tf.constant_initializer(np.zeros(config.dim_z, dtype='float32'))
        self.mu_0 = tf.Variable(initial_value=mu_0(shape=(config.dim_z)), trainable=config.trainable_mu0, dtype='float32', name="mu0")

        # Trainable        
        self.trainable_params = []
        self.trainable_params.append(self.A) if config.trainable_A else None
        self.trainable_params.append(self.C) if config.trainable_C else None
        self.trainable_params.append(self.Q) if config.trainable_Q else None
        self.trainable_params.append(self.R) if config.trainable_R else None
        self.trainable_params.append(self.sigma_0) if config.trainable_sigma0 else None
        self.trainable_params.append(self.mu_0) if config.trainable_mu0 else None

        self.length = config.length
    
    @property
    def transition_noise(self):
        return tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(self.dim_z, dtype='float32'), 
                                                        scale_tril=tf.linalg.LinearOperatorLowerTriangular(self.Q).to_dense())    
    @property
    def observation_noise(self):
        return tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(self.dim_x, dtype='float32'),
                                                                        scale_tril=tf.linalg.LinearOperatorLowerTriangular(self.R).to_dense()) 
    
    @property
    def initial_prior(self):
        return tfp.distributions.MultivariateNormalTriL(loc = self.mu_0, scale_tril=tf.linalg.LinearOperatorLowerTriangular(self.sigma_0).to_dense())        
    
    @property
    def lgssm(self):
        return tfp.distributions.LinearGaussianStateSpaceModel(num_timesteps = self.length,
                                                               transition_matrix = self.A, 
                                                               transition_noise = self.transition_noise, 
                                                               observation_matrix = self.C,
                                                               observation_noise = self.observation_noise, 
                                                               initial_state_prior = self.initial_prior, 
                                                               initial_step=0,
                                                               validate_args=False, 
                                                               allow_nan_stats=True,
                                                               name='LinearGaussianStateSpaceModel')      

    @tf.function
    def get_sequence(self, X, no_samples = 1, mask=None):
        self.length = X.shape[1]
        
        ll, mu_f, P_f, mu_p, P_p, obs_mu, obs_cov = self.lgssm.forward_filter(X, mask=mask)
        
        out_dist = tfp.distributions.MultivariateNormalTriL(obs_mu, get_cholesky(obs_cov))
        std = out_dist.stddev()
        mean = out_dist.mean()

        z_samples = self.lgssm.posterior_sample(X,
                                                mask = mask,
                                                sample_shape=(no_samples))[:,0,...]
        
        samples = tf.matmul(self.C, z_samples[..., None])[...,0]
        return samples, mean, std

    def call(self, inputs, online_learning=False):
        x = inputs[0]
        mask = inputs[1]

        if online_learning:
            ll, mu_f, P_f, mu_p, P_p, obs_mu, obs_P = self.lgssm.forward_filter(x, mask=mask)                
            return {'obs_filt': [obs_mu, obs_P], 'pred': [mu_p, P_p]}, ll
        
        # Initialize state-space model        
        self.length = x.shape[1]
        
        # Run the smoother and draw sample from i
        # p(z[t]|x[:T])  
        mu_s, P_s = self.lgssm.posterior_marginals(x, mask = mask)
        p_smooth = tfp.distributions.MultivariateNormalTriL(mu_s, get_cholesky(P_s))
        z = p_smooth.sample()        

        # Get p(z[t+1] | x[:t]) and p(z[t] | x[:t])
        ll, mu_f, P_f, mu_p, P_p, _, _ = self.lgssm.forward_filter(x, mask=mask)

        # p(z[t+1] | x[:t]) -> p(z[t] | z[t-1]) for t = 2,...T, z = [z_1, z_2, ..., z_T]
        p_pred = tfp.distributions.MultivariateNormalTriL(mu_p[:,:-1,:], get_cholesky(P_p[:,:-1,:]))
        
        # p(z[t] | x[:t]) -> p(x[t] | x[:t])
        mu_obsfilt, P_obsfilt = self.lgssm.latents_to_observations(mu_f, P_f)
        p_obs_filt = tfp.distributions.MultivariateNormalTriL(mu_obsfilt, get_cholesky(P_obsfilt))
        
        log_pred, log_filt, log_p_1, log_smooth = self._get_loss(x, z, p_smooth, p_obs_filt, p_pred)
        return log_pred, log_filt, log_p_1, log_smooth, ll        
    
    def _get_loss(self, x, z, p_smooth, p_obs_filt, p_pred):
        """
        Get log probability densitity functions for the kalman filter
        ```
        z_t ~ N(μ_{t|T}, ∑_{t|T}) for t = 1,...,T
        log p(z_t|z_{t-1}) = log N(z_t | A z_{t-1}, R) = log N(z_t - Az_{t-1} | 0, R) for t = 2,...,T 
        log p(x_t|z_t) = log N (x_t | Cz_t, Q) = log N(x_t - Cz_t | 0, Q) for t = 1,...,T
        log p(z_1) = log N(z_1 | μ_0, ∑_0)
        log p(z_t|x_{1:T}) = log N(z_t | μ_{t|T}, ∑_{t|T}) for t = 1,...,T
        ```
        
        Args:
            x_: encoder sample
            z: smooth sample
            p_smooth: smooth distribution p(z_t | z_{1:T})
            
        Returns:
            Prediction : log p(z[t] | z[:t-1]) for t = 2,...,T
            Filtering : log p(x[t] | z[t]) for t = 1,..., T
            Initial : log p(z[1] | x_M)
            Smoothing : log p(z[t] | x[1:T]) for t = 1,..., T
            Log-likelihood: log p(x[t] | x[:t-1])
        """
        
        # log p(z[t] | x[1:T])
        log_smooth = p_smooth.log_prob(z)
        
        # log p(z[1]),
        z_1 = z[:, 0, :]
        log_p_1 = self.lgssm.initial_state_prior.log_prob(z_1)
        
        # log p(z[t] | z[t-1]) for t = 2,...T, z = [z_1, z_2, ..., z_T]
        log_pred = p_pred.log_prob(z[:,1:,:])
        
        # log p(x[t] | x[:t]) for t = 1,...,T
        log_filt = p_obs_filt.log_prob(x)
        
        return log_pred, log_filt, log_p_1, log_smooth

    def get_obs_distributions(self, x, mask):        
        steps = x.shape[1]
        #Smooth
        p_smooth, p_obssmooth = self.get_smooth_dist(x, mask, steps = steps)

        # Filter        
        p_filt, p_obsfilt, p_obspred = self.get_filter_dist(x, mask, steps = steps, get_pred=True)   

        return {"smooth_mean": p_obssmooth.mean(),
                "smooth_cov": p_obssmooth.covariance(),
                "filt_mean": p_obsfilt.mean(),
                "filt_cov": p_obsfilt.covariance(),
                "pred_mean": p_obspred.mean(), 
                "pred_cov": p_obspred.covariance(),
                "x": x}
        
    def get_distribtions(self, x, mask):        
        self.length = x.shape[1]
        
        #Smooth
        mu, P = self.lgssm.posterior_marginals(x, mask = mask)
        mu, P = self.lgssm.latents_to_observations(mu, P)
        p_obssmooth = tfp.distributions.MultivariateNormalTriL(loc=mu, 
                                                               scale_tril=get_cholesky(P))
		
        # Filter        
        _, mu_f, P_f, _, _, mu_obsp, P_obsp = self.lgssm.forward_filter(x, mask=mask)
        # Obs filt dist
        mu_obsfilt, P_obsfilt = self.lgssm.latents_to_observations(mu_f, P_f)
        p_obsfilt = tfp.distributions.MultivariateNormalTriL(mu_obsfilt, get_cholesky(P_obsfilt))
        
		# Obs pred dist p(x[t] | x[:t-1])
        p_obspred = tfp.distributions.MultivariateNormalTriL(mu_obsp, get_cholesky(P_obsp))
        
        return p_obssmooth, p_obsfilt, p_obspred

    def get_smooth_dist(self, x, mask, steps = None):        
        self.length = x.shape[1]
        mu, P = self.lgssm.posterior_marginals(x, mask = mask)
        p_smooth = tfp.distributions.MultivariateNormalTriL(loc=mu,
                                                            scale_tril=get_cholesky(P))
        mu, P = self.lgssm.latents_to_observations(mu, P)
        p_obssmooth = tfp.distributions.MultivariateNormalTriL(loc=mu, 
                                                               scale_tril=get_cholesky(P))
        return p_smooth, p_obssmooth

    def get_filter_dist(self, x, mask, get_pred=False, steps = None):        
        self.length = x.shape[1]
        _, mu_f, P_f, _, _, mu_obsp, P_obsp = self.lgssm.forward_filter(x, mask=mask)
        # Filt dist
        p_filt = tfp.distributions.MultivariateNormalTriL(mu_f, get_cholesky(P_f))
        # Obs filt dist
        mu_obsfilt, P_obsfilt = self.lgssm.latents_to_observations(mu_f, P_f)
        p_obsfilt = tfp.distributions.MultivariateNormalTriL(mu_obsfilt, get_cholesky(P_obsfilt))
        
        if get_pred:
            # Obs pred dist p(x[t] | x[:t-1])
            p_obspred = tfp.distributions.MultivariateNormalTriL(mu_obsp, get_cholesky(P_obsp))
        
            return p_filt, p_obsfilt, p_obspred
        
        return p_filt, p_obsfilt
    
    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            out, ll = self(data, online_learning=True) 
            loss_value = -ll*tf.cast(data[1]==False, tf.float32)
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))    
        return out, loss_value