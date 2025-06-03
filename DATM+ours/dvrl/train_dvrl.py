# coding=utf-8
"""
Training on DVRL class
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys 
sys.path.append("../")
import pickle
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

import torch.optim as optim

import dvrl.dvrl as dvrl
import train_utils.helper as helper
# from data_utils.FeatureDataset import FeatureDataset
# from models.predictor_model import Predictor

from utils.utils_gsam import get_dataset, get_network, get_daparam,\
    TensorDataset, ParamDiffAug
from models.predictor_model import Predictor



def train(args):
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.log_path = os.path.join(args.log_path, args.dataset, args.model, \
                                 args.noise_type + '_' + str(args.noise_rate))

    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)

    # load dataset
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv, dvrl_train, dvrl_val = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    # get network
    feature_model = get_network(args.model, channel, num_classes, im_size).to(args.device)
    feature_model = feature_model.to(args.device)

    saved_path = os.path.join('models', args.log_path, 'train_source')
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    initial_epoch, feature_model = helper.load_saved_model(saved_path, feature_model)
    if initial_epoch == 0:
        # 训练 feature_model 50 轮
        train_loader = torch.utils.data.DataLoader(dvrl_train, batch_size=args.inner_batch_size, shuffle=True)
        optimizer = torch.optim.Adam(feature_model.parameters(), lr=1e-3)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        criterion = torch.nn.CrossEntropyLoss()

        for e in range(50):
            exp_lr_scheduler.step(e)

            # used to check the training accuracy
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for i_batch, datum in enumerate(train_loader):
                feature_model.train()
                optimizer.zero_grad()

                image_batch = datum[0].float().to(args.device)
                label_batch = datum[1].long().to(args.device)
        
                outputs = feature_model(image_batch)
                loss = criterion(outputs, label_batch)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * image_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == label_batch).sum().item()
                total_samples += image_batch.size(0)
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            print(f"Epoch {e+1}/50, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
            # save model
            if (e + 1) % args.save_epoch == 0:
                    torch.save(feature_model.state_dict(), os.path.join(saved_path, 'net_epoch%d.pth' % (e + 1)))
        
    pred_model = Predictor(feature_model.num_feat, num_classes)
    # import pdb
    # pdb.set_trace()

    # DVRL Initialize DVRL
    print('Initialize DVRL class')
    dvrl_class = dvrl.Dvrl(dvrl_train, dvrl_val, feature_model, pred_model, args)
    
    # Train DVRL Value estimator
    dvrl_class.train_dvrl()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--data_path', type=str, default='../../dataset', help='dataset path')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    

    parser.add_argument("--noise_type", default='clean', type=str, help="[clean,symmetric,asymmetric] , is_annot:[aggre, worst, rand1, rand2, rand3, clean100, noisy100]")
    parser.add_argument("--noise_rate", default=0, type=float, help="")
    parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)
    parser.add_argument('--is_annot', action='store_true', default=False)
    parser.add_argument('--is_human', action='store_true', default=False)
    parser.add_argument('--is_coarse', action='store_true', default=False)

    parser.add_argument('--zca', action='store_true')

    # parser.add_argument('--load_fmodel', action='store_true', default=False)


    parser.add_argument('--train_dvrl', action='store_true', default=True)
    # dvrl related
    parser.add_argument('--inner_batch_size', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--inner_iteration', type=int, default=10)
    parser.add_argument('--inner_learning_rate', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--comb_dim', type=int, default=10)
    parser.add_argument('--outer_iterations', type=int, default=2000) 
    parser.add_argument('--layer_number', type=int, default=5) 
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=1e-2)

    parser.add_argument('--epsilon', type=float, default=1e-8, help='Adds to the log to avoid overflow')  
    parser.add_argument('--threshold', type=float, default=0.9, help='Encourages exploration') 


    parser.add_argument('--log_path', type = str, default='../logs/')
    parser.add_argument('--save_epoch', type = int, default=50)
    parser.add_argument('--ori_epoch', type = int, default=5)
    parser.add_argument('--val_epoch', type = int, default=10)


    args = parser.parse_args()
    train(args)

# CUDA_VISIBLE_DEVICES=1 nohup python -u train_dvrl.py --dataset CIFAR10 --batch_real 256 --noise_type symmetric --noise_rate 0.2 --zca > ../logs/CIFAR10/train_dvrl_symmetric_0.2.log 2>&1 &






