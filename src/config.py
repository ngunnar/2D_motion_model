from collections import namedtuple
from tqdm import tqdm
def get_config(
    #DS
    ds_path,
    dim_y = (112,112),
    length=50,    
    ds_size= None,
    use_aug = None,
    #VAE
    enc_filters = [16, 32, 64, 128],
    dec_filters = [16, 32, 64, 128],
    int_steps = 0,
    spatial_smoothing = 2.0,
    # LGSSM
    dim_x = 16,
    dim_z = 32,
    noise_emission =  0.1, # x noise
    noise_transition =  0.1, # z noise
    init_cov = 1.0,
    trainable_A = True,
    trainable_C = True,
    trainable_R = True,
    trainable_Q = True,
    trainable_mu = True,
    trainable_sigma0 = True,
    # Training
    losses = ['kvae_loss'],
    gpu = '0',
    num_epochs = 50,
    start_epoch = 0,
    model_path = None,
    batch_size = 4,
    init_lr = 1e-4,
    scale_reconstruction = 1.0,
    kl_latent_loss_weight = 1.0,
    kf_loss_weight = 1.0,
    kl_growth = 3.0,
    # Plotting
    plot_epoch = 1
):
    
    if ds_size is not None and ds_size < batch_size:
        tqdm.write("Changing batch_size from {0} to {1}".format(batch_size, ds_size))
        batch_size = ds_size
    
    config_dict = {
        # DS
        "dim_y":dim_y,
        "length":length,    
        "ds_path": ds_path,
        "ds_size": ds_size,
        "use_aug": use_aug, 
        #VAE
        'enc_filters':enc_filters,
        'dec_filters':dec_filters,
        'int_steps': int_steps,
        'spatial_smoothing':spatial_smoothing,    
        # LGSSM
        "dim_x": dim_x,
        "dim_z": dim_z,
        "noise_emission": noise_emission,
        "noise_transition": noise_transition,
        "init_cov": init_cov, #30.0
        "trainable_A":trainable_A,
        "trainable_C":trainable_C,
        "trainable_R":trainable_R,
        "trainable_Q":trainable_Q,
        "trainable_mu0":trainable_mu,
        "trainable_sigma0":trainable_sigma0,        
        # Training
        "losses": losses,
        "gpu": gpu,
        "num_epochs": num_epochs,
        "start_epoch": start_epoch,
        "model_path": model_path,
        "batch_size": batch_size,
        "init_lr": init_lr,
        "scale_reconstruction": scale_reconstruction,
        "kl_latent_loss_weight": kl_latent_loss_weight,
        "kf_loss_weight": kf_loss_weight,
        "kl_growth": kl_growth,
        # Plotting
        "plot_epoch": plot_epoch,
        }
    config = namedtuple("Config", config_dict.keys())(*config_dict.values())
    return config, config_dict