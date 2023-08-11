# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0

@author: Junxiao Song
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                            kernel_size=3,
                                            padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(out_channels, out_channels,
                                            kernel_size=3,
                                            padding=1),
                                  nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = self.relu(out)
        return out

class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height

        self.n_residual_blocks = 40

        # single convo layer
        self.conv = nn.Sequential(nn.Conv2d(4, 256, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU())

        # 40 residual layers
        self.residual_layer = []
        for i in range(self.n_residual_blocks):
            self.residual_layer.append(ResidualBlock(256, 256))

        # action policy layers
        self.policy = nn.Sequential(nn.Conv2d(256, 2, kernel_size=1),
                                    nn.BatchNorm2d(2),
                                    nn.ReLU())
        self.policy_fc1 = nn.Linear(2*board_width*board_height,
                                    board_width*board_height)

        # state value layers
        self.value = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1),
                                   nn.BatchNorm2d(1),
                                   nn.ReLU())
        self.value_fc1 = nn.Linear(board_width*board_height, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, state_input):
        # common layers
        x = self.conv(state_input)
        for i in range(self.n_residual_blocks):
            x = self.residual_layer[i](x)

        # action
        x_act = self.policy(x) 
        x_act = x_act.view(-1, 2*self.board_width*self.board_height)
        x_act = F.log_softmax(self.policy_fc1(x_act), dim=-1) # add dim= -1

        # state value layers
        x_val = self.value(x)
        x_val = x_val.view(-1, self.board_width*self.board_height)
        x_val = F.relu(self.value_fc1(x_val))
#         x_val = torch.tanh(self.val_fc2(x_val)) # use torch.tanh
        x_val = self.value_fc2(x_val).tanh() # use torch.tanh

        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            if use_gpu == True:
                net_params = torch.load(model_file)
            else:
                net_params = torch.load(model_file, map_location=torch.device('cpu'))
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
#         return loss.data[0], entropy.data[0]
        #for pytorch version >= 0.5 please use the following line instead.
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
