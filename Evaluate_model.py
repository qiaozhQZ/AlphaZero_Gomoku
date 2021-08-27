# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras


def run():
    n = 5
    width, height = 8, 8
    model_file_1 = 'best_policy_8_8_5.model'
    model_file_2 = 'best_policy_885_1.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        try:
            policy_param_1 = pickle.load(open(model_file_1, 'rb'))
        except:
            policy_param_1 = pickle.load(open(model_file_1, 'rb'),
                                       encoding='bytes')  # To support python3
        best_policy_1 = PolicyValueNetNumpy(width, height, policy_param_1)
        mcts_player_1 = MCTSPlayer(best_policy_1.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # set larger n_playout for better performance
        #second player and policy
#         try:
#             policy_param_2 = pickle.load(open(model_file_2, 'rb'))
#         except:
#             policy_param_2 = pickle.load(open(model_file_2, 'rb'),
#                                        encoding='bytes')  # To support python3
        best_policy_2 = PolicyValueNet(width, height, model_file_2, use_gpu=True)
        mcts_player_2 = MCTSPlayer(best_policy_2.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # set larger n_playout for better performance

        # set start_player=0 for human first
        for i in range(5):  
            winner = game.start_play(mcts_player_1, mcts_player_2, start_player=1, is_shown=0)
            print('winner is {}'.format(winner))
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
