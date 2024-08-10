import numpy as np

# fmt: off
puzzles = [
    (
        "0-1 We all start somewhere",
        np.array([[1, -1, 1]])
    ),
    (
        "0-2 Careful, getting fancy now",
        np.array([
            [-1,  1, -1],
            [-1, -1, -1],
            [ 1, -1,  1]
        ])
    ),
    (
        "0-3 Got them moves",
        np.array([
            [-1, -1, -1],
            [-1, -2, -1],
            [ 1, -2,  1]
        ])
    ),
    (
        "0-4 Careful...",
        np.array([
            [-1, -1, -1,  1],
            [-1, -2, -1, -1],
            [ 1, -2, 1, 1],
            [ 1, -2, -1, -1]
        ])
    ),
    # TODO: create a 0-5 which is a hard single color
    (
        "1-1 There's colors now?",
        np.array([
            [0, 0, 0],
            [0, 1, 0],
            [1, -1, 1]
        ])
    ),
    (
        "1-2 Can you figure out the right move?",
        np.array([
            [0, -1, 0],
            [0, 1,-1],
            [1, -1, -1]
        ])
    ),
    (
        "1-3 Twirly swirly",
        np.array([
            [1, 0, 0, -1],
            [1, 1, -1, 1],
            [-1, 0, -1, 1],
            [-1, 0, -1, -1]
        ])
    ),
    (
        "1-4 Is it just me,\nor are these getting harder?",
        np.array([
            [-1, -1, 0, -1],
            [-1, -1, 0, 1],
            [-1, 1, 1, 0],
            [1, 0, -1, -1]
        ])
    ),
    (
        "1-5 Nvm, its just me",
        np.array([[2, -1, 2]])
    )
]