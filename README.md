# 2D motion model

## Online learning in motion modeling for intra-interventional image sequences
Official implementation of **Online learning in motion modeling for intra-interventional image sequences**.

## Installation
Install the conda environment from the YML-file:

```
conda env create -f 2DMM.yml
```

Activate environment:

```
conda activate 2DMM
```

## Train model

### ACDC
For the ACDC experiment, we first preprocessed the data. After downloading the dataset, please run the './script/preprocess_acdc.py'. This will save the training and test dataset as two Tensorflow datasets.

```
python train_acdc.py --gpus '0' --ds_path 'path_to_acdc_dataset'
```

### Echonet

```
python train_acdc.py --gpus '0' --ds_path 'path_to_echonet_dataset'
```

### Arguments

Arguments

```
optional arguments:
  -h, --help            show this help message and exit
  -y DIM_Y, --dim_y DIM_Y
                        dimension of image variable (default:(128, 128))
  -x DIM_X, --dim_x DIM_X
                        dimension of latent variable (default:4)
  -z DIM_Z, --dim_z DIM_Z
                        dimension of state space variable (default:8)
  -length LENGTH, --length LENGTH
                        length of time sequence (default:53)
  -int_steps INT_STEPS, --int_steps INT_STEPS
                        flow integration steps (default:7)
  -saved_model SAVED_MODEL, --saved_model SAVED_MODEL
                        model path if continue running model (default:None)
  -start_epoch START_EPOCH, --start_epoch START_EPOCH
                        start epoch (default:0)
  -gpu GPUS, --gpus GPUS
                        comma separated list of GPUs (default:-1)
  -prefix PREFIX, --prefix PREFIX
                        predix for log folder (default:None)
  -ds_path DS_PATH, --ds_path DS_PATH
                        path to dataset (default:/data/Niklas/CineMRI/tf_data_128)
  -ds_size DS_SIZE, --ds_size DS_SIZE
                        Size of datasets (default:None)
```

## Test model
Here are some code-snippets how to run inference on the model. If you run it in e.g. a jupyter notebook, we also provide animation of the transformed image.

```python
### INLCUDE THIS IS ALL EXAMPLE MODULES BELOW ####
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import os
import tqdm

from src.fkvae import fKVAE
from src.utils import read_config, load_model, show_pred, plot_latent_sequence, plot_A

# Path to saved model and config file
path = # example './logs/sim_sag/pre-runned/'
ckpt = # example: path + '/' + 'end_model'
ds_path = # path to dataset, example: ./CineMRI/tf_data_128'
config, config_dict = read_config(path)

path = os.path.join(ds_path, 'testing_data')
test_dataset = tf.data.Dataset.load(path).batch(1)

for d in test_dataset:
    break

#################################################

# Load model
model = fKVAE(config, config.length)
model.load_weights(ckpt)

y, y0, mask = model.parse_inputs(d)
phis, xs, x_samples = model.reconstruct_phis(y, y0, mask, False)

x_lgssm, mu_samples, std_samples = model.lgssm.get_sequence(xs, mask=mask)
plot_latent_sequence(xs, x_lgssm, mu_samples, std_samples)

y_pred = model.warp([phis, y0])
HTML(show_pred(y[0,...], y_pred[0,...]))

```
## Sequential run

```python
# Load model
model = fKVAE(config, 1) # Set length to 1!
model.load_weights(ckpt)

path = os.path.join(config.ds_path, 'testing_data')
test_dataset = tf.data.Dataset.load(path).batch(1)

y, y0, mask = model.parse_inputs(d)

phis = None
xs = None
x_samples = None

for t in tqdm.tqdm(range(y.shape[1])):    
    phi, x, x_sample = model.reconstruct_phis(y[:,t:t+1], y0, mask[:,t:t+1], False)    
    if t == 0:
        phis = phi
        xs = x
        x_samples = x_sample
    else:
        phis = tf.concat([phis, phi], axis=1)
        xs = tf.concat([xs, x], axis=1)
        x_samples = tf.concat([x_samples, x_sample], axis=1)

y_pred = model.warp([phis, y0])
x_lgssm, mu_samples, std_samples = model.lgssm.get_sequence(xs, mask=mask)
plot_latent_sequence(xs, x_lgssm, mu_samples, std_samples)

HTML(show_pred(y[0,...], y_pred[0,...]))
```
## Online traning
This is only suitable for the EchoNet experiment.

```
python train_acdc.py --path 'path to model checkpoint' --ckpt 'name of checkpoint' --ds_path 'path to data of checkpoint'
```

## Citation
TBD