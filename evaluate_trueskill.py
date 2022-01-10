# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import os
import random
import itertools
import math

import pickle
from trueskill import Rating, quality_1vs1, rate_1vs1, rate

from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
from policy_value_net_pytorch import PolicyValueNet  # Pytorch

def readModel():
    
    '''save model name and model file name to a dictionary'''
    
    model_path = os.getcwd() + '/PyTorch_models'
    model_list = [files for root, dirs, files in os.walk(model_path)][0] # have two lists, second is empty
    model_dict = {}
    for model in model_list:
        model_dict['model' + '_' + model.split('.')[0].split('_')[-1]] = model
    return model_dict
    
def compete(model_file_1, model_file_2):
    
    '''compare any two models'''
    
    # build the board
    n = 5
    width, height = 8, 8

    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)
        
        # for pyTorch
        best_policy_1 = PolicyValueNet(width, height, model_file_1, use_gpu=True)
        mcts_player_1 = MCTSPlayer(best_policy_1.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # set larger n_playout for better performance
        
        best_policy_2 = PolicyValueNet(width, height, model_file_2, use_gpu=True)
        mcts_player_2 = MCTSPlayer(best_policy_2.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # set larger n_playout for better performance

        winner = game.start_play(mcts_player_1, mcts_player_2, start_player=1, is_shown=0)
        return winner
#         print('winner is {}'.format(winner))
            
    except KeyboardInterrupt:
        print('\n\rquit')
        
    
def run():
    model_dict = readModel()
    test_models = [[model, Rating()] for model in model_dict.values()]
    
    for _ in range(2):
        model_1 = random.choice(test_models)
        model_2 = sorted([(quality_1vs1(model_1[1], m[1]), m) for m in test_models if m[0] != model_1[0]])[-1][1]
        
        model_file_1 = os.getcwd() + '/PyTorch_models/' + model_1[0]
        model_file_2 = os.getcwd() + '/PyTorch_models/' + model_2[0]
        
        winner = compete(model_file_1, model_file_2)
        model1_wins = winner == 1
        model2_wins = winner == 2
        tie = winner == -1

        if model1_wins:
            model_1[1], model_2[1] = rate_1vs1(model_1[1], model_2[1])
        if model2_wins:
            model_2[1], model_1[1] = rate_1vs1(model_2[1], model_1[1])
        if tie:
            model_1[1], model_2[1] = rate_1vs1(model_1[1], model_2[1], drawn=True)
        
        print('game {} completed'.format(_))
    
    print(test_models)
    return test_models

if __name__ == '__main__':
    run()
