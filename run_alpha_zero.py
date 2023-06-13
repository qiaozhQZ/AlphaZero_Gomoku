from matplotlib import pyplot as plt
import numpy as np
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet

if __name__ == "__main__":
    size = 8 # DON"T CHANGE
    empty_board = Board(width=size, height=size)
    empty_board.init_board()

    model_file = "C:/Users/qzhang490/Lab/AlphaZero_Gomoku/Models/Batch_30_models/210_current_policy.model"
    policy_value_net = PolicyValueNet(size,
                                    size,
                                    model_file=model_file,
                                    use_gpu=True)

    az_mcts_player = MCTSPlayer(
        policy_value_net.policy_value_fn,
        c_puct=8,
        n_playout=400)
    
    _, probs = az_mcts_player.get_action(empty_board, temp=1, return_prob=True)

    print(probs)

    fig, ax = plt.subplots()
    
    im = ax.imshow(probs.reshape((size, size)))

    row_label = [0 + i*size for i in range(size)]
    ax.set_yticks(np.arange(len(row_label)), labels=row_label)

    for row in range(size):
        for col in range(size):
            text = ax.text(col, row, round(probs.reshape((size, size))[row, col], 3),
                            ha="center", va="center", color="w")

    ax.set_title("AlphaZero MCTS Probs")
    fig.tight_layout()
    print("Move Probs Shape:", probs.shape)
    print(probs.reshape((size, size)))
    plt.show()