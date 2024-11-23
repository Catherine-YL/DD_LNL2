import sys

# sys.path.append("..")

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import fire
import ml_collections
from functools import partial

# from jax._src.config import config
# config.update("jax_enable_x64", True)

import jax
from absl import logging
import absl
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
from flax.training import train_state, checkpoints


from dataloader import get_dataset, configure_dataloader

# from lib.dataset.dataloader import get_dataset, configure_dataloader
# from lib.models.utils import create_model
# from lib.datadistillation.utils import save_dnfr_image, save_proto_np
# from lib.datadistillation.frepo import proto_train_and_evaluate, init_proto, ProtoHolder
# from lib.training.utils import create_train_state
# from lib.dataset.augmax import get_aug_by_name

from clu import metric_writers

from collections import namedtuple


from models import ResNet18, KIP_ConvNet, linear_net, Conv
from augmax import get_aug_by_name

import numpy as np
import jax.numpy as jnp
import algorithms
import optax
import time
import pickle
import contextlib
import warnings

import json
from jax._src.config import config as jax_config
# from jax.config import config as jax_config

import sys 
sys.path.append("../")
from dataSolu.utils_noise import noisify

def get_config():
    # Note that max_lr_factor and l2_regularization is found through grid search.
    config = ml_collections.ConfigDict()
    config.random_seed = 0
    config.train_log = 'train_log'
    config.train_img = 'train_img'
    config.mixed_precision = False
    config.resume = True

    config.img_size = None
    config.img_channels = None
    config.num_prototypes = None
    config.train_size = None

    config.dataset = ml_collections.ConfigDict()
    config.kernel = ml_collections.ConfigDict()
    config.online = ml_collections.ConfigDict()

    # Dataset
    config.dataset.name = 'cifar100'  # ['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'tiny_imagenet']
    config.dataset.data_path = 'data/tensorflow_datasets'
    config.dataset.zca_path = 'data/zca'
    config.dataset.zca_reg = 0.1

    # online
    config.online.img_size = None
    config.online.img_channels = None
    config.online.mixed_precision = config.mixed_precision
    config.online.optimizer = 'adam'
    config.online.learning_rate = 0.0003
    config.online.arch = 'dnfrnet'
    config.online.output = 'feat_fc'
    config.online.width = 128
    config.online.normalization = 'identity'

    # Kernel
    config.kernel.img_size = None
    config.kernel.img_channels = None
    config.kernel.num_prototypes = None
    config.kernel.train_size = None
    config.kernel.mixed_precision = config.mixed_precision
    config.kernel.resume = config.resume
    config.kernel.optimizer = 'lamb'
    config.kernel.learning_rate = 0.0003
    config.kernel.batch_size = 1024
    config.kernel.eval_batch_size = 1000

    return config

# noise_type:[clean,symmetric,asymmetric] , is_annot:[aggre, worst, rand1, rand2, rand3, clean100, noisy100]
def main(dataset_name = 'cifar10', data_path=None, zca_path=None, train_log=None, train_img=None, width=128, random_seed=0, message = 'Put your message here!', output_dir = None, n_images = 10, config_path = None, log_dir = None, max_steps = 10000, use_x64 = False, skip_tune = False, naive_loss = False, init_random_noise = False, noise_type = 'clean', noise_rate = 0.0, is_annot=False, is_coarse=False):
    # --------------------------------------
    # Setup
    # --------------------------------------

    
    print(f"ipc:{n_images}")
    if use_x64:
        jax_config.update("jax_enable_x64", True)

    logging.use_absl_handler()

    if log_dir is None and output_dir is not None:
        log_dir = output_dir
    elif log_dir is None:
        log_dir = './logs/'
    
    if not os.path.exists('./{}'.format(log_dir)):
        os.makedirs('./{}'.format(log_dir))

    logging.get_absl_handler().use_absl_log_file('{}, {}'.format(int(time.time()), message), './{}/'.format(log_dir))
    absl.flags.FLAGS.mark_as_parsed() 
    logging.set_verbosity('info')
    
    logging.info('\n\n\n{}\n\n\n'.format(message))
    
    config = get_config()
    config.random_seed = random_seed
    config.train_log = train_log if train_log else 'train_log'
    config.train_img = train_img if train_img else 'train_img'
    # --------------------------------------
    # Dataset
    # --------------------------------------
    print('load dataset')
    

    config.dataset.data_path = data_path if data_path else 'data/tensorflow_datasets'
    config.dataset.zca_path = zca_path if zca_path else 'data/zca'
    config.dataset.name = dataset_name

    (ds_train, ds_test), preprocess_op, rev_preprocess_op, proto_scale = get_dataset(config.dataset, is_coarse)
        
    #--------make noise for label-----------------
    print('noise type is', noise_type)
    print("add noise")
    if noise_type != 'clean':
        if dataset_name == 'cifar10':
            images_tmp, labels_tmp = [], []
            for image_tmp, label_tmp in ds_train:
                images_tmp.append(image_tmp.numpy())  
                labels_tmp.append(label_tmp.numpy())  
            images_np = np.array(images_tmp)
            labels_np = np.array(labels_tmp)
            
            if is_annot == False:
                labels_np = labels_np.reshape(-1, 1)
                train_noisy_labels, actual_noise_rate = noisify(dataset='cifar10', train_labels=labels_np, noise_type=noise_type, noise_rate=noise_rate, random_state=0, nb_classes=10)
                print('noise rate is', noise_rate)
                print('over all noise rate is ', actual_noise_rate)

            else:
                noise_file = np.load('../dataSolu/CIFAR-10_human_ordered.npy', allow_pickle=True)
                noise_type_map = {'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3'}
                noise_type = noise_type_map[noise_type]
                aggre_label = noise_file.item().get(noise_type)
                train_noisy_labels = np.asarray(aggre_label).reshape(-1,1)
                print(f'The overall noise rate is {1-np.mean(labels_np == aggre_label)}')
                
            ds_train_new = tf.data.Dataset.from_tensor_slices((images_np, train_noisy_labels.squeeze()))  
            ds_train = ds_train_new

        elif dataset_name == 'cifar100':
            images_tmp, labels_tmp = [], []
            for image_tmp, label_tmp in ds_train:
                images_tmp.append(image_tmp.numpy())  
                labels_tmp.append(label_tmp.numpy())  
            images_np = np.array(images_tmp)
            labels_np = np.array(labels_tmp)
            if is_annot == False:
                labels_np = labels_np.reshape(-1, 1)
                train_noisy_labels, actual_noise_rate = noisify(dataset='cifar100', train_labels=labels_np, noise_type=noise_type, noise_rate=noise_rate, random_state=0, nb_classes=100)
                print('noise rate is', noise_rate)
                print('over all noise rate is ', actual_noise_rate)
            else:
                # noise_file = np.load('./dataSolu/CIFAR-100_human_ordered.npy', allow_pickle=True)
                # noise_type_map = {'clean100': 'clean_label', 'noisy100': 'noise_label'}
                # noise_type = noise_type_map[noise_type]
                # aggre_label = noise_file.item().get(noise_type)
                # train_noisy_labels = np.asarray(aggre_label).reshape(-1,1)
                # print(f'The overall noise rate is {1-np.mean(labels_np == aggre_label)}')
                noise_file = np.load('../dataSolu/CIFAR-100_human_ordered_coarse.npy', allow_pickle=True)
                noise_data = noise_file.item()
                print("Available keys:", noise_data.keys())
                
                # noise_type should be "noisy_coarse_label"
                if is_coarse:
                    clean_label = noise_data['clean_coarse_label']
                    train_noisy_labels = np.asarray(noise_data['noisy_coarse_label']).reshape(-1,1)
                    print(f'The overall noise rate is {1-np.mean(np.array(clean_label) == train_noisy_labels.squeeze())}')
                # else "noise_label"    
                else:
                    clean_label = noise_data['clean_label']
                    train_noisy_labels = np.asarray(noise_data['noise_label']).reshape(-1,1)
                    print(f'The overall noise rate is {1-np.mean(labels_np == train_noisy_labels.squeeze())}')

            ds_train_new = tf.data.Dataset.from_tensor_slices((images_np, train_noisy_labels.squeeze()))  
            ds_train = ds_train_new
            
        elif dataset_name == 'tiny_imagenet':
            images_tmp, labels_tmp = [], []
            for image_tmp, label_tmp in ds_train:
                images_tmp.append(image_tmp.numpy())  
                labels_tmp.append(label_tmp.numpy())  
            images_np = np.array(images_tmp)
            labels_np = np.array(labels_tmp)
            
            labels_np = labels_np.reshape(-1, 1)
            train_noisy_labels, actual_noise_rate = noisify(dataset='tiny', train_labels=labels_np, noise_type=noise_type, noise_rate=noise_rate, random_state=0, nb_classes=200)
            print('noise rate is', noise_rate)
            print('over all noise rate is ', actual_noise_rate)
            
            ds_train_new = tf.data.Dataset.from_tensor_slices((images_np, train_noisy_labels.squeeze()))  
            ds_train = ds_train_new

    #---------------------------------------------
    
    print("init_proto")
    
    coreset_images, coreset_labels = algorithms.init_proto(ds_train, n_images, config.dataset.num_classes, seed = random_seed, random_noise = init_random_noise)

    num_prototypes = n_images * config.dataset.num_classes
    print()
    print(num_prototypes)
    print()
    config.kernel.num_prototypes = num_prototypes
    
    y_transform = lambda y: tf.one_hot(y, config.dataset.num_classes, on_value=1 - 1 / config.dataset.num_classes,
                                           off_value=-1 / config.dataset.num_classes)

    ds_train = configure_dataloader(ds_train, batch_size=config.kernel.batch_size, y_transform=y_transform,
                                        train=True, shuffle=True)
    ds_test = configure_dataloader(ds_test, batch_size=config.kernel.eval_batch_size, y_transform=y_transform,
                                   train=False, shuffle=False)

    
    num_classes = config.dataset.num_classes


    if config.dataset.img_shape[0] in [28, 32]:
        depth = 3
    elif config.dataset.img_shape[0] == 64:
        depth = 4
    elif config.dataset.img_shape[0] == 128:
        depth = 5
    else:
        raise Exception('Invalid resolution for the dataset')
    
    key = jax.random.PRNGKey(random_seed)
    
    alg_config = ml_collections.ConfigDict()


    if config_path is not None:
        print(f'loading config from {config_path}')
        logging.info(f'loading config from {config_path}')
        loaded_dict = json.loads(open('./{}'.format(config_path), 'rb').read())
        loaded_dict['direct_batch_sizes'] = tuple(loaded_dict['direct_batch_sizes'])
        alg_config = ml_collections.config_dict.ConfigDict(loaded_dict)

    alg_config.l2 = alg_config.l2_rate * config.kernel.num_prototypes

    alg_config.use_x64 = use_x64
    alg_config.naive_loss = naive_loss

    alg_config.output_dir = output_dir
    alg_config.max_steps = max_steps
    alg_config.model_depth = depth

    print(alg_config)

    logging.info('using config from ./{}'.format(config_path))
    logging.info(alg_config)

    if output_dir is not None:
        if not os.path.exists('./{}'.format(output_dir)):
            os.makedirs('./{}'.format(output_dir))

        with open('./{}/config.txt'.format(output_dir), 'a') as config_file:
            config_file.write(repr(alg_config))

    print("load model")

    model_for_train = Conv(use_softplus = (alg_config.softplus_temp != 0), beta = alg_config.softplus_temp, num_classes = num_classes, width = width, depth = depth, normalization = 'batch' if alg_config.has_bn else 'identity')

    

    

    #Tuning inner and hessian inverse learning rate

    print("Tuning learning rates -- this may take a few minutes")
    logging.info("Tuning learning rates -- this may take a few minutes")

    inner_learning_rate = 0.00001 #initialize them to be small, then gradually increase until unstable
    hvp_learning_rate = 0.00005

    start_time = time.time()

    if not skip_tune:
        with contextlib.redirect_stdout(None):
        # if True:
            inner_result = 1
            while inner_result == 1:

                inner_result, _ = algorithms.run_rcig(coreset_images, coreset_labels, model_for_train.init, model_for_train.apply, ds_train, alg_config, key, inner_learning_rate, hvp_learning_rate, lr_tune = True)
                inner_learning_rate *= 1.2

            inner_learning_rate *= 0.7

            hvp_result = 1
            while hvp_result == 1:

                _, hvp_result = algorithms.run_rcig(coreset_images, coreset_labels, model_for_train.init, model_for_train.apply, ds_train, alg_config, key, inner_learning_rate, hvp_learning_rate, lr_tune = True)
                hvp_learning_rate *= 1.2

            hvp_learning_rate *= 0.7

        
    print("Done tuning learning rates")
    print(f'inner_learning_rate: {inner_learning_rate} hvp learning_rate: {hvp_learning_rate}')
    logging.info("Done tuning learning rates")
    logging.info(f'inner_learning_rate: {inner_learning_rate} hvp learning_rate: {hvp_learning_rate}')

    logging.info(f'Completed LR tune in {time.time() - start_time}s')


    #Training


    logging.info('Begin training')

    start_time = time.time()

    
    coreset_train_state, key, pool, inner_learning_rate, hvp_learning_rate, best_acc = algorithms.run_rcig(coreset_images, coreset_labels, model_for_train.init, model_for_train.apply, ds_train, alg_config, key, inner_learning_rate, hvp_learning_rate, start_iter = 0)

    logging.info(f'Completed in {time.time() - start_time}s')

    logging.info(f'Saving final checkpoint')
    absolute_path = os.path.abspath('./{}/'.format(alg_config.output_dir))
    checkpoints.save_checkpoint(ckpt_dir = absolute_path, target = coreset_train_state, step = 'final', keep = 1e10)



    #Save version for visualizing (without ZCA transform)
    visualize_output_dict = {
        'coreset_images': np.array(rev_preprocess_op(coreset_train_state.ema_average['x_proto'])),
        'coreset_labels': np.array(coreset_train_state.ema_average['y_proto']),
        'dataset': config.dataset
    }

    if output_dir is not None:
        pickle.dump(visualize_output_dict, open('./{}/{}.pkl'.format(output_dir, 'distilled_dataset_vis'), 'wb'))

    print(f'new learning rates: {inner_learning_rate}, {hvp_learning_rate}')
    logging.info(f'new learning rates: {inner_learning_rate}, {hvp_learning_rate}')
    
    print(f'max_acc: {best_acc}')
    logging.info(f'new learning rates: {best_acc}')

    
if __name__ == '__main__':

    tf.config.experimental.set_visible_devices([], 'GPU')
    devices = jax.devices()
    print(devices)  # 列出所有设备
    gpus = tf.config.experimental.list_physical_devices('GPU')

    fire.Fire(main)
