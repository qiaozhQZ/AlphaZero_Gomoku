# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import os
import re
import sys
import random
import itertools
import math
import pandas as pd

import pickle
from trueskill import Rating, quality_1vs1, rate_1vs1, rate

from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
from policy_value_net_pytorch import PolicyValueNet  # Pytorch

def readModel():
    
    '''save model name and model file name to a dictionary'''
    
    model_dir = 'batch100_mini2048_model'
    model_path = os.getcwd() + '/' + model_dir
#     model_list = [files for root, dirs, files in os.walk(model_path)][0] # have two lists, second is empty
    model_list = [f for f in os.listdir(model_path) if re.match('\d*_best.*', f)] # only take the best policy models
    model_dict = {}
    for model in model_list:
#         model_dict['model' + '_' + model.split('.')[0].split('_')[-1]] = model # PyTorch_model
        model_dict['model' + '_' + model.split('_')[0]] = model # batch100_mini2048_model
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
        
def run_all():
    model_dict = readModel()
    game_rst = {k: [] for k in ['model_1', 'model_2', 'winner']}
    
    for i in range(10):
        game_cnt = 0
        for model_1 in model_dict.keys():
            for model_2 in model_dict.keys():
                if model_1 == model_2:
                    continue
                
                game_cnt += 1
                model_file_1 = model_path + '/' + model_dict[model_1]
                model_file_2 = model_path + '/' + model_dict[model_2]
                
                winner = compete(model_file_1, model_file_2)
                game_rst['model_1'].append(model_1)
                game_rst['model_2'].append(model_2)
                game_rst['winner'].append(winner)
                
                print('game {} of batch {} completed: {} vs {}, winner is {}'.format(game_cnt, i+1, model_1, model_2, winner))
        # save at the end of each loop
        game_df = pd.DataFrame(game_rst)
        game_df.to_csv(os.getcwd() + '/' + 'game_rst_{}.txt'.format(i+1), sep = '\t')
                
    game_df = pd.DataFrame(game_rst)
    game_df.to_csv(os.getcwd() + '/' + 'game_rst.txt', sep = '\t')

        
def run_epoch():
    model_dict = readModel()
    test_models = [[0, random.random(), model_name, Rating()] for model_name in model_dict.keys()]
    game_rst = {k: [] for k in ['model_1', 'model_2', 'winner']}
    
    for i in range(10):
        
        test_models = [[model[0], random.random(), model[2], model[3]] for model in test_models]
        random.shuffle(test_models)
        test_models = sorted(test_models)
        
        model_1 = test_models[0]
        model_2 = test_models[1]
        
        # randomly choose model_1, and choose model_2 that has the closest mu to model_2
#         model_1 = random.choice(test_models)
#         model_2 = sorted([(quality_1vs1(model_1[1], m[1]), m) for m in test_models if m[0] != model_1[0]])[-1][1]

        # count the number of games each model has played
        model_1[0] += 1
        model_2[0] += 1

        #find relevant model files
        model_file_1 = os.getcwd() + '/PyTorch_models/' + model_dict[model_1[2]]
        model_file_2 = os.getcwd() + '/PyTorch_models/' + model_dict[model_2[2]]
                
        # run the game and save the game information to a dictionary
        winner = compete(model_file_1, model_file_2)
        game_rst['model_1'].append(model_1[2])
        game_rst['model_2'].append(model_2[2])
        game_rst['winner'].append(winner)        
        
        # true skill update
        if winner == 1:
            model_1[-1], model_2[-1] = rate_1vs1(model_1[-1], model_2[-1])
        elif winner == 2:
            model_2[-1], model_1[-1] = rate_1vs1(model_2[-1], model_1[-1])
        elif winner == -1:
            model_1[-1], model_2[-1] = rate_1vs1(model_1[-1], model_2[-1], drawn=True)
        else:
            print("unknown winner type ", winner)

        print('game {} completed: {} vs {}, winner is {}'.format(i, model_1[2], model_2[2], winner))
        
    print(test_models)
    test_models_df = pd.DataFrame(sorted(test_models), columns = ['game_count', 'random_number', 'model_name', 'true_skill'])
    test_models_df.to_csv(os.getcwd() + '/' + 'test_models.txt', sep = '\t')
    game_df = pd.DataFrame(game_rst)
    game_df.to_csv(os.getcwd() + '/' + 'game_rst.txt', sep = '\t')


if __name__ == '__main__':
    run_all()
#     run_epoch()