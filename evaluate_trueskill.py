# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

import random
import itertools
import math

from __future__ import print_function
import pickle
from trueskill import Rating, quality_1vs1, rate_1vs1, rate

from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
from policy_value_net_pytorch import PolicyValueNet  # Pytorch


def run():
    n = 5
    width, height = 8, 8
    model_file_1 = 'best_policy_885_2_50.model'
    model_file_2 = 'best_policy_885_2_10500.model'
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

        # set start_player=0 for human first
        for i in range(5):  
            winner = game.start_play(mcts_player_1, mcts_player_2, start_player=1, is_shown=0)
            print('winner is {}'.format(winner))
            
            
        
    except KeyboardInterrupt:
        print('\n\rquit')
        
    def win_probability(team1, team2):
        delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
        sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
        size = len(team1) + len(team2)
        denom = math.sqrt(size * (1 * 1) + sum_sigma)
        ts = trueskill.global_env()
        return ts.cdf(delta_mu / denom)
    
        # models = [['model{} weights'.format(i) , Rating()] for i in range(15)]
        
        for _ in range(100):
        player1 = random.choice(models)
        player2 = sorted([(quality_1vs1(player1[1], m[1]), m) for m in models if m[0] != player1[0]])[-1][1]
        #print(player1[0], player2[0])
        player1_wins = random.random() < win_probability([player1[1]], [player2[1]])
        if player1_wins:
            player1[1], player2[1] = rate_1vs1(player1[1], player2[1])
        else:
            player2[1], player1[1] = rate_1vs1(player2[1], player1[1])


if __name__ == '__main__':
    run()
