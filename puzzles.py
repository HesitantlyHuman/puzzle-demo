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
    (
        "0-5 ...its getting crowded",
        np.array([
            [-1, -1, -1,  1, 1],
            [-1, -1, -2, -2, -2],
            [ 1, -1, -1, -1, -1],
            [-2, -1, 1, -2, 1],
            [-2, 1, -2, -2, 1]
        ])
    ),
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
        "1-4 Excuse me, pardon",
        np.array([
            [-1, -1, 0, -1],
            [-1, -1, 0, 1],
            [0, 1, 1, 0],
            [1, -1, -1, -1]
        ])
    ),
    (
        "1-5 Is it just me,\nor are these getting harder?",
        np.array([
            [1, -1, 0, 0, 1],
            [-1, -1, -1, -1, 1],
            [-1, 0, 0, -2, -1],
            [-1, 0, 1, -1, -1],
            [-1, -1, 1, -1, 0]
        ])
    ),
    (
        "2-0 Nvm, its just me",
        np.array([[2, -1, 2]])
    ),
    (
        "2-1 Is that supposed to be there?",
        np.array([
            [1, 1, 0, -1],
            [-1, 1, -1, 0],
            [-1, -1, -1, -2],
            [-1, 0, 2, -2]
        ])
    ),
    (
        "2-2 This seems familiar...",
        np.array([
            [1, 0, 0, -1],
            [1, 1, 2, 1],
            [-1, 0, -1, 1],
            [-1, 0, -1, 2]
        ])
    )
]