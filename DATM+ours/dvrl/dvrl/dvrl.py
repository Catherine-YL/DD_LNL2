# coding=utf-8
"""
The core class of DVRL(Data Valuation using Reinforcement Learning).
"""

import copy
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from sklearn import metrics
from tqdm import tqdm

import train_utils.helper as helper
from models.value_estimator import ValueEstimator
from dvrl.dvrl_pretrain import pretrain
from dvrl.dvrl_loss import DvrlLoss


class Dvrl(object):
    """
    Data Valuation using Reinforcement Learning (DVRL) class.
    """

    def __init__(self, train_set, val_set, feature_model, pred_model, args):
        """
        Construct class.
        :param train_loader: dataloader for source.
        :type train_loader:
        :param val_loader: dataloader for validation.
        :type val_loader:
        :param pred_model: predictor model to classify images based on the features.
        :type pred_model:
        :param parameters: user argparse
        :type parameters:
        """
        # Network parameters for data value estimator
        self.hidden_dim = args.hidden_dim
        self.comb_dim = args.comb_dim
        self.outer_iterations = args.outer_iterations
        self.layer_number = args.layer_number
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate

        # Basic parameters
        self.epsilon = args.epsilon  # Adds to the log to avoid overflow
        self.threshold = args.threshold  # Encourages exploration

        self.inner_batch_size = args.inner_batch_size
        self.inner_lr = args.inner_learning_rate
        self.inner_iteration = args.inner_iteration

        self.ori_epoch = args.ori_epoch
        self.val_epoch = args.val_epoch

        self.pred_model = copy.deepcopy(pred_model)
        self.final_model = copy.deepcopy(pred_model)
        self.feature_model = copy.deepcopy(feature_model)

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.inner_batch_size, shuffle=True, num_workers=0)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.log_path = args.log_path
        self.device = args.device
        current_path = os.path.dirname(__file__)
        current_path = os.path.join(current_path, args.log_path)

        source_dir = os.path.join(current_path, 'source_pretrain')
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)

        val_dir = os.path.join(current_path, 'val_pretrain')
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)

        # check if source model is already pretrained
        init_epoch, self.ori_model = helper.load_saved_model(source_dir, copy.deepcopy(pred_model))
        if init_epoch == 0:
            self.ori_model = pretrain(self.ori_model, self.train_loader, self.inner_lr, self.feature_model,
                                    cuda=args.device, epoch=args.ori_epoch)
            torch.save(self.ori_model.state_dict(),
                       os.path.join(source_dir, 'net_epoch%d.pth' % args.ori_epoch))

        # check if val model is already pretrained
        init_epoch, self.val_model = helper.load_saved_model(val_dir, copy.deepcopy(pred_model))
        if init_epoch == 0:
            self.val_model = pretrain(self.val_model, self.val_loader, self.inner_lr, self.feature_model,
                                      cuda=args.device, epoch=args.val_epoch)
            torch.save(self.val_model.state_dict(),
                       os.path.join(val_dir, 'net_epoch%d.pth' % args.val_epoch))

        if args.device == 'cuda':
            self.final_model.cuda()
            self.pred_model.cuda()
            self.val_model.cuda()
            self.ori_model.cuda()

    def train_dvrl(self):
        """
        Train value estimator
        :return:
        :rtype:
        """
        # selection network
        # self.value_estimator = ValueEstimator(self.hidden_dim, self.layer_number, self.comb_dim)
        self.value_estimator = ValueEstimator(self.feature_model.num_feat, self.feature_model.num_classes, self.hidden_dim, self.layer_number, self.comb_dim)
        self.value_estimator = self.value_estimator.cuda()

        current_path = os.path.dirname(__file__)
        current_path = os.path.join(current_path, self.log_path)

        if not os.path.exists(os.path.join(current_path, 'selection_network')):
            os.makedirs(os.path.join(current_path, 'selection_network'))
        
        init, self.value_estimator = helper.load_saved_model(os.path.join(current_path,
                                                                          'selection_network'),
                                                             self.value_estimator)
        # if init > 0:
        #     print('value estimator already trained and loaded')
        #     return

        # loss function
        dvrl_criterion = DvrlLoss(self.epsilon, self.threshold)
        # optimizer
        dvrl_optimizer = optim.Adam(self.value_estimator.parameters(), lr=self.learning_rate)
        # learning rate scheduler
        scheduler = lr_scheduler.ExponentialLR(dvrl_optimizer, gamma=0.999)

        # baseline performance
        pred_list = []
        label_list = []

        # for (i, batch_data) in enumerate(self.val_loader):
            # feature, label = batch_data['feature'], batch_data['label']
        for i_batch, datum in enumerate(self.val_loader): # change
            image, label = datum[0].float().to(self.device), datum[1].long().to(self.device)

            feature = self.feature_model(image, finaly_layer=False)
            # feature = feature.cuda()
            # label = label.cuda()

            output = self.ori_model(feature)
            _, pred = torch.max(output, 1)

            pred = pred.cpu().detach().numpy()
            label = label.cpu().detach().numpy()

            pred_list += pred.tolist()
            label_list += label.tolist()

        # valid_perf = metrics.f1_score(label_list, pred_list, average='binary')
        valid_perf = metrics.f1_score(label_list, pred_list, average='micro')
        # valid_perf = metrics.accuracy_score(label_list, pred_list)
        print('Origin model Performance F1: %f' % valid_perf)

        best_reward = 0
        for epoch in range(self.outer_iterations):
            scheduler.step()
            # clean up grads
            self.value_estimator.train()
            self.value_estimator.zero_grad()
            dvrl_optimizer.zero_grad()

            # train predictor from scratch everytime
            new_model = copy.deepcopy(self.pred_model)
            new_model.cuda()
            # predictor optimizer
            pre_optimizer = optim.Adam(new_model.parameters(), lr=self.learning_rate)

            # use a list save all s_input and data values
            data_value_list = []
            s_input = []
            for inner in range(self.inner_iteration):
                # for (i, batch_data) in enumerate(self.train_loader):
                for i_batch, datum in enumerate(self.train_loader):
                    new_model.train()
                    new_model.zero_grad()
                    pre_optimizer.zero_grad()

                    # feature, label = batch_data['feature'], batch_data['label']
                    # feature = feature.cuda()
                    # label = label.cuda()

                    image, label = datum[0].float().to(self.device), datum[1].long().to(self.device)
                    with torch.no_grad():
                        # extract feature
                        feature = self.feature_model(image, finaly_layer=False)

                    label_one_hot = F.one_hot(label, num_classes=self.feature_model.num_classes)

                    output = self.val_model(feature)
                    output_softmax = F.softmax(output, dim=1)
                    # y_pred_diff = torch.abs(label_one_hot - output_softmax)[:, 0]
                    y_pred_diff = torch.abs(label_one_hot - output_softmax)
            
                    # selection estimation
                    # est_dv_curr = self.value_estimator(feature, torch.unsqueeze(label, 1), torch.unsqueeze(y_pred_diff, 1))  # raw
                    est_dv_curr = self.value_estimator(feature, label_one_hot, torch.squeeze(y_pred_diff, 1))
                    data_value_list.append(est_dv_curr)

                    sel_prob_curr = np.random.binomial(1, est_dv_curr.cpu().detach().numpy(), est_dv_curr.shape)
                    sel_prob_curr = torch.Tensor(sel_prob_curr).cuda()
                    s_input.append(sel_prob_curr)

                    # train new model
                    output = new_model(feature)

                    # loss function
                    pre_creterion = nn.CrossEntropyLoss(reduction='none')
                    loss = pre_creterion(output, label)
                    # the samples that are not selected won't have impact on the gradient
                    loss = loss * torch.squeeze(sel_prob_curr, 1)

                    # back propagation
                    loss.mean().backward()
                    pre_optimizer.step()

            # dvrl performance
            pred_list = []
            label_list = []

            # test the performance of the new model
            new_model.eval()
            # for (i, batch_data) in enumerate(self.val_loader):
            #     feature, label = batch_data['feature'], batch_data['label']
                # feature = feature.cuda()
                # label = label.cuda()
            for i_batch, datum in enumerate(self.val_loader):
                image, label = datum[0].float().to(self.device), datum[1].long().to(self.device)
                feature = self.feature_model(image, finaly_layer=False)

                output = new_model(feature)
                _, pred = torch.max(output, 1)

                pred = pred.cpu().detach().numpy()
                label = label.cpu().detach().numpy()

                pred_list += pred.tolist()
                label_list += label.tolist()

            # dvrl_perf = metrics.f1_score(label_list, pred_list, average='binary')
            dvrl_perf = metrics.f1_score(label_list, pred_list, average='micro')
            # dvrl_perf = metrics.accuracy_score(label_list, pred_list)
            reward = dvrl_perf - valid_perf

            if reward > best_reward:
                best_reward = reward
                flag_save = True
            else:
                flag_save = False

            # update the selection network
            reward = torch.Tensor([reward])
            data_value_list = torch.cat(data_value_list, 0)
            s_input = torch.cat(s_input, 0)

            reward = reward.cuda()
            data_value_list = data_value_list.cuda()
            s_input = s_input.cuda()
            loss = dvrl_criterion(data_value_list, s_input, reward)

            print('At epoch %d, the reward is %f, the prob is %f' % (epoch, reward.cpu().detach().numpy()[0], \
                  np.max(data_value_list.cpu().detach().numpy())))

            loss.backward()
            dvrl_optimizer.step()

            if flag_save or epoch % 100 ==0:
                saved_path = os.path.join(current_path, 'selection_network')
                if not os.path.exists(saved_path):
                    os.makedirs(saved_path)
                torch.save(self.value_estimator.state_dict(), os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

    def dvrl_estimate(self, feature, label):
        """
        Estimate the given data value.
        :param feature: Data intermediate feature.
        :type feature: torch.Tensor
        :param label: Corresponding labels
        :type label:torch.Tensor
        :return:
        :rtype:
        """
        # first calculate the prection difference
        label_one_hot = F.one_hot(label, num_classes=self.feature_model.num_classes)
        output = self.val_model(feature)
        output_softmax = F.softmax(output, dim=1)
        # y_pred_diff = torch.abs(label_one_hot - output_softmax)[:, 0]  # raw
        y_pred_diff = torch.abs(label_one_hot - output_softmax)

        # predict the value
        # data_value = self.value_estimator(feature, torch.unsqueeze(label, 1), torch.unsqueeze(y_pred_diff, 1))  # raw
        data_value = self.value_estimator(feature, label_one_hot, torch.squeeze(y_pred_diff, 1))
        return data_value.cpu().detach().numpy()