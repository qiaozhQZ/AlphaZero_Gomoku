# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import os
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras
from multiprocessing import pool
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

use_gpu = False

class board_data(Dataset):
    def __init__(self, data_buffer):
        self.data_buffer = data_buffer

    def __len__(self):
        return len(self.data_buffer)

    def __getitem__(self, idx):
        return self.data_buffer[idx] 


def do_selfplay(args):
    game_num, model_checkpoint, board_width, board_height, n_in_row, alpha, c_puct, n_playout, temp=args

    board = Board(width=board_width,
                  height=board_height,
                  n_in_row=n_in_row)
    game = Game(board)

    policy = PolicyValueNet(board_width, board_height,
                            model_file=model_checkpoint, use_gpu=False)

    mcts_player = MCTSPlayer(policy.policy_value_fn,
                             alpha=alpha, c_puct=c_puct,
                             n_playout=n_playout, is_selfplay=1)

    return game.start_self_play(mcts_player, is_shown=0, temp=temp)

class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-4
        # self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 3.0
        self.alpha = 10 / (self.board_width * self.board_height)
        self.buffer_size = 10000
        self.batch_size = 1024  # mini-batch size for training, 512 as default
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 8
        self.epochs = 1  # num of train_steps for each update
        # self.kl_targ = 0.001
        self.check_freq = 25 # num of games in a batch
        self.game_batch_num = 3000 #maximun games played
        self.best_win_ratio = 0.0

        if not use_gpu:
            self.pool = Pool(processes=self.play_batch_size)

        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model,
                                                   use_gpu=use_gpu)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   use_gpu=use_gpu)

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      alpha=self.alpha,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)

                states = torch.tensor(equi_state, dtype=torch.float)
                probs = torch.tensor(np.flipud(equi_mcts_prob).flatten(), dtype=torch.float)
                winners = torch.tensor(winner, dtype=torch.float) 

                if use_gpu:
                    states = states.cuda()
                    probs = probs.cuda()
                    winners = winners.cuda()

                extend_data.append((states, probs, winners))

                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)

                states = torch.tensor(equi_state, dtype=torch.float)
                probs = torch.tensor(np.flipud(equi_mcts_prob).flatten(), dtype=torch.float)
                winners = torch.tensor(winner, dtype=torch.float) 

                if use_gpu:
                    states = states.cuda()
                    probs = probs.cuda()
                    winners = winners.cuda()

                extend_data.append((states, probs, winners))

        return extend_data



    def collect_selfplay_data(self, n_games=1):
        self.episode_len = []

        if not use_gpu:
            model_checkpoint = os.getcwd() + '/temp.model'
            self.policy_value_net.save_model(model_checkpoint)

            results = self.pool.map(do_selfplay, [(i, model_checkpoint, self.board_width,
                                                   self.board_height,
                                                   self.n_in_row, self.alpha,
                                                   self.c_puct, self.n_playout,
                                                   self.temp)
                                                  for i in range(n_games)])

            for winner, play_data in results:
                play_data = list(play_data)[:]
                self.episode_len.append(len(play_data))
                # augment the data
                play_data = self.get_equi_data(play_data)
                self.data_buffer.extend(play_data)

        else:
            for i in range(n_games):
                winner, play_data = self.game.start_self_play(self.mcts_player,
                                                              temp=self.temp)
                # winner, play_data = roll_out[i] # reading the data from roll_out
                play_data = list(play_data)[:]
                self.episode_len.append(len(play_data))
                # augment the data
                play_data = self.get_equi_data(play_data)
                self.data_buffer.extend(play_data)

    def policy_update(self):
        train_set = board_data(self.data_buffer)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        loss = self.policy_value_net.train_step(train_loader,
                                                lr=self.learn_rate,
                                                epochs=self.epochs)

        print(("loss:{:.5f},").format(loss))
        return loss

    def policy_evaluate(self, n_games=10): # set n_games to 0
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """

        # TODO why?
        if n_games == 0:
            return 0.0

        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)

        # make a new MCTS player
#         best_policy_value_net = PolicyValueNet(self.board_width,
#                                                self.board_height,
#                                                model_file='./best_policy.model',
#                                               use_gpu=True)

#         best_mcts_player = MCTSPlayer(best_policy_value_net.policy_value_fn,
#                                       c_puct=self.c_puct,
#                                       n_playout=self.n_playout)

        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          #                                           best_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def render_probs_empty(self, i, save=False):

        path = os.getcwd() + '/' + 'board_heatmap'
        if not os.path.exists(path):
            os.mkdir(path)
        empty_board = Board()
        empty_board.init_board()

        # pure_mcts_player = MCTS_Pure(c_puct=1, n_playout=2000)
        # move_probs = None

        # for _ in range(20):
        #     acts, mp = pure_mcts_player.get_action(empty_board, counts=True)
        #     if move_probs is None:
        #         move_probs = mp.copy()
        #     else:
        #         move_probs += mp

        acts, move_probs = self.mcts_player.get_action(empty_board, temp=1.0, return_prob=1)

        fig, ax = plt.subplots()
        im = ax.imshow(move_probs.reshape((8,8)))

        row_label = [0 + i*8 for i in range(8)]
        col_label = [56 + i for i in range(8)]
        ax.set_yticks(np.arange(len(row_label)), labels=row_label)
        ax.set_xticks(np.arange(len(col_label)), labels=col_label)

        for row in range(8):
            for col in range(8):
                text = ax.text(col, row, round(move_probs.reshape((8,8))[row, col], 3),
                               ha="center", va="center", color="w")

        ax.set_title("batch_{}".format(i))
        fig.tight_layout()
        # print("Move Probs Shape:", move_probs.shape)
        # print(move_probs.reshape((8,8)))
        if path:
            # print("current_batch", i)
            fig.savefig(path + "/batch_{}".format(i) + ".png")
        else:
            fig.show()
        plt.close(fig)

    def run(self):
        """run the training pipeline"""
        # make directory to save policy models
        # dir_path = os.getcwd() + '/' + 'batch{}_mini{}_model'.format(self.check_freq, self.batch_size)
        datestring = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        path = os.getcwd() + "/" + "testing_only_" + datestring

        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)

        with open('train_settings.yaml', 'w') as f:
            f.write(yaml.dump({
                'learn_rate': self.learn_rate,
                'batch_size': self.batch_size,
                'check_freq': self.check_freq,
                'epochs': self.epochs,
                'temp': self.temp,
                'alpha': self.alpha,
                'c_puct': self.c_puct,
                'n_playout': self.n_playout,
                'buffer_size': self.buffer_size
                }))

        try:
            i = 0 #start
            while True: #keep training

                with open('train_settings.yaml', 'r') as f:
                    config = yaml.safe_load(f)
                self.learn_rate = config['learn_rate']
                self.batch_size = config['batch_size']
                self.check_freq = config['check_freq']
                self.epochs = config['epochs']
                self.temp = config['temp']

                if config['buffer_size'] != self.buffer_size:
                    self.buffer_size = config['buffer_size']
                    new_data_buffer = deque(maxlen=self.buffer_size)
                    new_data_buffer.extend(self.data_buffer)
                    self.data_buffer = new_data_buffer

                if (config['alpha'] != self.alpha or config['c_puct'] !=
                    self.c_puct or config['n_playout'] != self.n_playout):
                    self.alpha = config['alpha']
                    self.c_puct = config['c_puct']
                    self.n_playout = config['n_playout']

                    self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                                  alpha=self.alpha,
                                                  c_puct=self.c_puct,
                                                  n_playout=self.n_playout,
                                                  is_selfplay=1)

                print(("lr:{:.7f}, batch_size:{}, epochs:{}, temp:{}, c_puct:{}, n_playout:{}, alpha: {}, buffer_size: {}/{}, check_freq: {}").format(self.learn_rate, self.batch_size, self.epochs, self.temp, self.c_puct, self.n_playout, self.alpha, len(self.data_buffer), self.buffer_size, self.check_freq))

                self.render_probs_empty(i, save=True)
#             for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                    i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model(os.getcwd() + '/' + '{}_current_policy.model'.format(i+1))
                    if win_ratio > 0.5 and win_ratio > self.best_win_ratio: # default is > self.best_win_ratio
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model(os.getcwd() + '/' + '{}_best_policy.model'.format(i+1))
                        if (self.best_win_ratio == 1.0 and
                            self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                i += 1 #increase batch number
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    # training_pipeline = TrainPipeline('./testing_only_2023-08-27_153627/temp.model')
    training_pipeline = TrainPipeline()
    training_pipeline.run()

    # training_pipeline.render_probs_empty(0, save=False)
