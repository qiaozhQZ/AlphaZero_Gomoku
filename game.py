# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np
from scipy.signal import convolve2d


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def __hash__(self) -> int:
        return hash(frozenset(self.states.items()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return False
        return self.states == other.states

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((9, self.width, self.height))

        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            # 0 is self, 1 is opp, 2 is empty playable
            square_state[2][:, :] = 1.0
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[2][move_curr // self.width,
                            move_curr % self.height] = 0.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            square_state[2][move_oppo // self.width,
                            move_oppo % self.height] = 0.0
            # indicate the last move location
            square_state[3][self.last_move // self.width,
                            self.last_move % self.height] = 1.0

            # get all legal moves
            rows, cols = np.where(square_state[2] == 1)
            coordinates = list(zip(rows, cols))

            for x, y in list(zip(rows, cols)):
                own = square_state[0].copy()
                own[x,y] = 1.0

                if self.is_five_in_a_row(own):
                    # marks moves that win in 1
                    square_state[4][x,y] = 1.0

                opp = square_state[1].copy()
                opp[x,y] = 1.0
                if self.is_five_in_a_row(opp):
                    # marks moves that lose in 1
                    square_state[5][x,y] = 1.0

        if len(self.states) % 2 == 0:
            square_state[6][:, :] = 1.0  # indicate the colour to play

        # plane of ones, marks playable area, to distinguish from padded regions
        square_state[7][:, :] = 1.0

        # plane of zeros
        square_state[8][:, :] = 0.0

        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
                self.players[0] if self.current_player == self.players[1]
                else self.players[1]
                )
        self.last_move = move

    def is_five_in_a_row(self, matrix):
        # Defining kernels for each direction
       horizontal_kernel = np.array([[1, 1, 1, 1, 1]])
       vertical_kernel = np.transpose(horizontal_kernel)
       diag_tl_br_kernel = np.eye(5)
       diag_bl_tr_kernel = np.flipud(np.eye(5))

       instances = []

       for kernel in [horizontal_kernel, vertical_kernel, diag_tl_br_kernel,
                      diag_bl_tr_kernel]:
           conv_result = convolve2d(matrix, kernel, mode='valid')
           matches = np.column_stack(np.where(conv_result == 5))
           if len(matches) > 0:
               return True

       return False

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []

        # target_temp = 0.1
        # anneal_time = 8 # moves before we reach target temp

        # temp_slope = 0
        # if temp > 0.1:
        #     temp_slope = (temp - target_temp) / anneal_time

        while True:

            # print("moving with temp", temp)
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)

            # # anneal temp
            # if temp > 0.1:
            #     temp -= temp_slope
            #     temp = max(temp, 0.1)

            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
