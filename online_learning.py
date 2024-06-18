import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
import os
import argparse

from src.fkvae import fKVAE
from src.utils import read_config, plot_latent_sequence, plot_A


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', '--path', type=str, help='path to model checkpoint')
    parser.add_argument('-ckpt', '--ckpt', type=str, help='name of checkpoint')
    parser.add_argument('-ds_path', '--ds_path', type=str, help='path to data of checkpoint')

    args = parser.parse_args()
    # Path to saved model and config file
    path = args.path # path to model checkpoint
    ckpt = args.ckpt # name of checkpoint
    ds_path = args.ds_path # path to dataset
    config, config_dict = read_config(path)

    path = os.path.join(ds_path, 'testing_data')
    test_dataset = tf.data.Dataset.load(path).batch(1)

    for d in test_dataset:
        break

    lr = 5e-4 # Set learning rate
    model = fKVAE(config, 1) # Set length to 1!
    model.load_weights(ckpt)
    model.lgssm.optimizer = tf.keras.optimizers.Adam(lr)

    plot_A(model)

    model.lgssm.optimizer = tf.keras.optimizers.Adam(lr)

    # Fix the initial prior
    model.lgssm.sigma_0 = tf.Variable(initial_value=model.lgssm.sigma_0, 
                                    trainable=False, dtype='float32', name="Sigma0")
    model.lgssm.mu_0 = tf.Variable(initial_value=model.lgssm.mu_0, 
                                trainable=False, dtype='float32', name="mu0")

    y, y0, mask = model.parse_inputs(d)
    s_feats = model.s_encoder(y0)

    seq_length = 20# Set moving horizon
    train_seq = 40# Set online traning horizon
    horizon = 10# Set forecast horizon

    x_prev = tf.zeros((1, seq_length, config.dim_x), dtype='float32')
    m_prev = tf.ones((1, seq_length), dtype='bool')

    x = None
    L = []
    for t in tqdm.tqdm(range(train_seq)):
        yt = y[:,t:t+1,...]
        # Save current parameters and restore those in case of NaN result
        preA = model.lgssm.A.numpy()
        preC = model.lgssm.C.numpy()
        preR = model.lgssm.R.numpy()
        preQ = model.lgssm.Q.numpy()
        
        model.lgssm.length = seq_length
        phi, out, loss_value, x_prev, m_prev = model.online_train_step(yt, y0, s_feats, x_prev[:,1:], m_prev[:,1:])
        if np.any(np.isnan(model.lgssm.A.numpy())):
            model.lgssm.A.assign(preA)
            model.lgssm.C.assign(preC)
            model.lgssm.R.assign(preR)
            model.lgssm.Q.assign(preQ)
            model.lgssm.optimizer.learning_rate.assign(model.lgssm.optimizer.learning_rate.numpy()*0.95)
        else:
            idx = np.argmax(m_prev==False)
            model.lgssm.mu_0.assign(out['pred'][0][0,idx,:])
            model.lgssm.sigma_0.assign(out['pred'][1][0,idx,...])
            
        loss = loss_value[0]*tf.cast(m_prev==False, 'float32')
        non_zero = tf.cast(loss != 0, tf.float32)
        loss = tf.reduce_sum(loss) / tf.reduce_sum(non_zero)
        L.append(loss.numpy())    
        if x is None:
            x = x_prev[:,-1,:][:,None,:]
        else:
            x = tf.concat([x, x_prev[:,-1,:][:,None,:]], axis=1)    


    plt.plot(L)
    plot_A(model)

    # Get true latent values
    for t in tqdm.tqdm(range(train_seq, train_seq + horizon)):
        yt = y[:,t:t+1,...]
        xt = model.encoder([yt, y0], False).sample()
        x = tf.concat([x, xt], axis=1)    

    # Evaluate
    mask = np.zeros((1, train_seq + horizon), dtype='bool')
    mask[:,-horizon:] = True
    x_masked = tf.concat([x[:,:-horizon,:], tf.zeros((1,horizon,x.shape[2]), dtype='float32')], axis=1)
    x_lgssm, mu_lgssm, std_lgssm = model.lgssm.get_sequence(x_masked, mask=mask)

    plot_latent_sequence(x, x_lgssm, mu_lgssm, std_lgssm)