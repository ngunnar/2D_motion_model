import os
import json
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from .config import get_config


def read_config(path):
    print('%s/%s' % (os.path.dirname(path), 'config.json'))
    print(os.path.dirname(path))
    _, config_dict = get_config(ds_path=None)
    
    with open('%s/%s' % (os.path.dirname(path), 'config.json')) as data_file:
        config_dict_base = json.load(data_file)
    
    #assert np.all([k in config_dict.keys() for k in config_dict_base.keys()]), "Loaded config file includes attributes that are not listed in default file"
    
    for k in config_dict:
        if k in config_dict_base:
            config_dict[k] = config_dict_base[k] 
    
    #config = namedtuple("Config", config_dict.keys())(*config_dict.values())
    config = namedtuple("Config", config_dict.keys())(*config_dict.values())
    return config, config_dict

def load_model(path, Model, file):
    config, config_dict = get_config(path)
    model = Model(config = config)
    model.load_weights(path + '/' + file).expect_partial()
    return model, config


def show_pred(true, pred, cmap='gray', save_name=None):
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    [ax.axis('off') for ax in axs]

    im1 = axs[0].imshow(true[0,...], cmap=cmap)
    im2 = axs[1].imshow(pred[0,...], cmap=cmap)
    def animate(i):
        im1.set_array(true[i,...])
        im2.set_array(pred[i,...])
        return im1, im2,

    anim = animation.FuncAnimation(fig, func=animate, frames=pred.shape[0], interval=20)
    if save_name is not None:        
        anim.save(f'./{save_name}.gif', writer='imagemagick', fps=30)
    return anim.to_jshtml()

def plot_A(model):
    fig, axs = plt.subplots(1,1, figsize=(4,4))

    circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
    axs.add_patch(circ)
    A = model.lgssm.A.numpy()
    eigenvalues = [e for e in np.linalg.eig(A)[0]]
    [axs.plot(e.real, e.imag, 'go--', linewidth=2, markersize=6) for e in eigenvalues]


    plt.tight_layout()
    plt.show()


def plot_latent_sequence(x_true, x_lgssm, mu_lgssm, std_lgssm, batch=0):
    fig, axs = plt.subplots(2,2, figsize=(15,5))
    axs = axs.flatten()

    t = np.arange(x_true.shape[1])
    for i in range(x_true.shape[-1]):
        axs[i].plot(t,x_true[batch,:,i], label='x', color='orange')
        
        axs[i].plot(t, x_lgssm[batch,:,i], color='blue')
        axs[i].plot(t, mu_lgssm[batch,:,i], '--', label='LGSSM', color='blue')
        axs[i].fill_between(t, mu_lgssm[batch,:,i] + 2*std_lgssm[batch,:,i], mu_lgssm[batch,:,i] - 2*std_lgssm[batch,:,i], alpha=0.3, color='blue')        

    plt.legend(loc='lower center')
    plt.show()
