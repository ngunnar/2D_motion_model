
import os
import argparse

from src.config import get_config
from src.train import train
from src.fkvae import fKVAE
from src.data.acdcDataset import get_dataset

def main(ds_path,
         dim_y = (124,124),
         dim_x = 4,
         dim_z = 8, 
         gpu = '0',
         int_steps = 0,
         model_path = None, 
         start_epoch = 0, 
         prefix = None,          
         batch_size = 4):
    
    config, config_dict = get_config(ds_path = ds_path,
                                     dim_y = dim_y, 
                                     dim_x = dim_x,
                                     dim_z = dim_z,     
                                     length=35,                                
                                     int_steps = int_steps,
                                     gpu = gpu, 
                                     start_epoch = start_epoch,
                                     model_path = model_path, 
                                     init_cov = 1.0,
                                     enc_filters = [32, 32, 32, 16],
                                     dec_filters = [32, 32, 32, 16],
                                     init_lr = 5e-4,                                     
                                     num_epochs = 500,
                                     batch_size = batch_size,
                                     spatial_smoothing = 2.0,
                                     losses = ['kvae_loss'],
                                     plot_epoch = 5,
                                     use_aug = {
                                         'add_noise_at_zeros': False,
                                         'add_noise':False,
                                         'add_flip':True,
                                         'add_rot':True,
                                         'add_zoom':False,
                                         'add_trans':True},
                                     )
    
    os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu

    # Data
    train_dataset, test_dataset, len_train = get_dataset(config.ds_path)

    log_folder = 'ACDC'

    train(config, config_dict, fKVAE, train_dataset, test_dataset, prefix, len_train, log_folder, buffer_size=len_train)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('-y', '--dim_y', type=tuple, help='dimension of image variable (default %(default))', default=(128,128))
    parser.add_argument('-x', '--dim_x', type=int, help='dimension of latent variable (default %(default))', default=16)
    parser.add_argument('-z', '--dim_z', type=int, help='dimension of state space variable (default %(default))', default=32)
    parser.add_argument('-int_steps', '--int_steps', type=int, help='flow integration steps (default %(default))', default=4)
    parser.add_argument('-saved_model','--saved_model', help='model path if continue running model (default:%(default))', default=None)    
    parser.add_argument('-start_epoch','--start_epoch', type=int, help='start epoch', default=0)
    parser.add_argument('-gpu','--gpus', help='comma separated list of GPUs (default %(default))', default='-1')
    parser.add_argument('-prefix','--prefix', help='predix for log folder (default:%(default))', default=None)    
    # data set
    parser.add_argument('-ds_path','--ds_path', help='path to dataset (default:%(default))', default='./ACDC')    
    
    args = parser.parse_args()
    
    
    main(dim_y = args.dim_y,
         dim_x = args.dim_x,
         dim_z = args.dim_z, 
         gpu = args.gpus,
         int_steps = args.int_steps,
         model_path = args.saved_model, 
         start_epoch = args.start_epoch,
         prefix = args.prefix,
         ds_path = args.ds_path)