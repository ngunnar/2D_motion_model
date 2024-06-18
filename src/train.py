import tensorflow as tf
import numpy as np
import datetime
import json

from .callbacks import VisualizeResultCallback


def train(config, config_dict, modelObj, train_dataset, test_dataset, prefix, len_train, log_folder, buffer_size = None, visualize_callback = VisualizeResultCallback, test_batch=None):
    if buffer_size is None:
        buffer_size = len_train    

    if test_batch is None:
        test_batch = config.batch_size

    # Put it before shuffle to get same plot images every time
    plot_train = list(train_dataset.batch(1).take(1))[0]    
    plot_test = list(test_dataset.batch(1).take(1))[0]
    
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(config.batch_size, drop_remainder=False)#.prefetch(None)
    test_dataset = test_dataset.batch(test_batch, drop_remainder=False)
    # Logging and callbacks
    if prefix is not None:
        log_dir = 'logs/{0}/{1}'.format(log_folder, prefix)
    else:
        log_dir = 'logs/{0}/{1}'.format(log_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    checkpoint_filepath = log_dir + '/cp-{epoch:04d}.ckpt'
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, 
        verbose=1, 
        save_weights_only=True,
        save_freq=int(np.ceil(len_train/config.batch_size)*10))
    
    img_log_dir = log_dir + '/img'
    file_writer = tf.summary.create_file_writer(img_log_dir)

    visualizeresult_callback = visualize_callback(file_writer, 
                                                  train_data = plot_train, 
                                                  test_data = plot_test, 
                                                  log_interval=config.plot_epoch)
    
    # model
    model = modelObj(config, config.length)
    model.compile(num_batches = np.ceil(len_train/config.batch_size))

    if config.model_path is not None:
        model.load_weights(config.model_path)

    model.save_weights(checkpoint_filepath.format(epoch=0))
    with open(log_dir + '/config.json', 'w') as f:
        json.dump(config_dict, f)
        
    model.fit(train_dataset, 
              epochs = config.num_epochs, 
              initial_epoch = config.start_epoch,
              verbose = 1, 
              validation_data = test_dataset,
              callbacks=[tensorboard_callback, 
                         visualizeresult_callback,
                         model_checkpoint_callback])


    model.save_weights(log_dir + '/end_model') 