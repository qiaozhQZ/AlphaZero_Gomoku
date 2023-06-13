from matplotlib import pyplot as plt
import numpy as np
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure

if __name__ == "__main__":
    size = 6
    empty_board = Board(width=size, height=size)
    empty_board.init_board()

    pure_mcts_player = MCTS_Pure(
        c_puct=8,
        n_playout=20000)
    
    _, counts = pure_mcts_player.get_action(empty_board, counts=True)

    print(counts)

    fig, ax = plt.subplots()
    
    im = ax.imshow(counts.reshape((size, size)))

    row_label = [0 + i*size for i in range(size)]
    ax.set_yticks(np.arange(len(row_label)), labels=row_label)

    for row in range(size):
        for col in range(size):
            text = ax.text(col, row, round(counts.reshape((size, size))[row, col], 3),
                            ha="center", va="center", color="w")

    ax.set_title("Pure MCTS Counts")
    fig.tight_layout()
    print("Move Counts Shape:", counts.shape)
    print(counts.reshape((size, size)))
    plt.show()