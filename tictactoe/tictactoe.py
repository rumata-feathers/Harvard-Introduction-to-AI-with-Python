"""
Tic Tac Toe Player
"""
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """

    empt_cnt = 0  # Amount of empty cells on the board
    for row in board:  # Let's iterate and count all empty cells
        for cell in row:
            if cell == EMPTY:
                empt_cnt += 1

    if empt_cnt % 2 == 0:  # With every move amount of empty spaces reduces by 1
        return O
    return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    Every EMPTY cell is a possible move, so let's add all empty cells to the set
    """

    moves = set()  # Set of all possible moves

    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == EMPTY:
                moves.add((i, j))  # Iterators are coordinates

    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    if board[action[0]][action[1]] != EMPTY:  # Check if cell is empty
        raise NameError('Invalid move')

    board_copy = copy.deepcopy(board)  # Create board copy to place the piece there
    board_copy[action[0]][action[1]] = player(board)  # Place a piece on the board
    return board_copy


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    winner_piece = X  # possible winner piece
    if player(board) == X:
        winner_piece = O

    # Now let's check if there is a winner
    #   we'll check rows, columns and diagonals

    # rows
    for row in board:
        if row == [winner_piece, winner_piece, winner_piece]:
            return winner_piece

    # columns
    for j in range(len(board[0])):
        column = [board[i][j] for i in range(len(board))]
        if column == [winner_piece, winner_piece, winner_piece]:
            return winner_piece

    # diagonals
    major_dgnl = [board[i][i] for i in range(len(board))]
    minor_dgnl = [board[len(board) - 1 - i][i] for i in range(len(board))]
    if [winner_piece, winner_piece, winner_piece] in [major_dgnl, minor_dgnl]:
        return winner_piece

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board):
        return True

    for row in board:
        for cell in row:
            if cell == EMPTY:
                return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    goal = 0  # 1 if X is making a move, -1 if O
    if player(board) == X:
        goal = 1
    elif player(board) == O:
        goal = -1

    if terminal(board):  # That means game is over
        return None

    score, best_action = minimax_cnt(board, goal)

    return best_action


def minimax_cnt(board, goal):
    """
    Returns the board score of minimax
    """

    if terminal(board):  # That means game is over
        return utility(board), None

    best_score = 0  # Best score on the board
    best_action = None  # Best action possible on the board
    moves = actions(board)

    while len(moves) != 0:
        action = moves.pop()
        action_score, next_action = minimax_cnt(result(board, action), goal * (-1))

        if best_action is None or abs(action_score - goal) < abs(best_score - goal):
            best_action = action
            best_score = action_score

    return best_score, best_action
