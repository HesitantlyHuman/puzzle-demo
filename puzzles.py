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
        "1-2 Can you figure out\nthe right move?",
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
    ),
    (
        "2-3 Peekaboo",
        np.array([
            [2, -2, -1, -1, -1, -1],
            [2, -1, -1, -1, -1, -1],
            [-1, -2, -1, -1, -1, -1],
            [-1, -1, -1, 1, 1, 0],
            [-1, -2, -2, 0, 1, 2]
        ])
    ),
    (
        "2-4 Ok... really!?\n(Peekaboo 2 Electric Boogaloo)",
        np.array([
            [2, -2, -1, -1, -1, -2],
            [2, -1, -1, -2, -1, -1],
            [-1, -2, -1, -1, -1, -1],
            [-1, -1, -1, 1, 1, 0],
            [-1, -2, -2, 0, 1, 2]
        ])
    ),
    (
        "2-5 They've really latched on\nto this, haven't they",
        np.array([
            [-1, -1, -1, -1, -2, -1],
            [-1, -2, -2, -1, -2, 2],
            [1, 1, -1, -1, 0, 0],
            [-1, -2, -1, -1, -2, -1],
            [-1, -2, 2, -2, -2, -1],
            [-1, -1, 2, -1, 0, 1]
        ])
    ),
    (
        "3-1 Are we sure this is safe?",
        np.array([
            [-1,  1, -1],
            [-1, -1, -1],
            [1, -3, 1]
        ])
    ),
    (
        "3-2 There's more of them?!",
        np.array([
            [1,  1, -1, -2],
            [ -1, -1, -1, -2],
            [-3, -1, -1, 1],
            [-3, -3, -2, -2]
        ])
    ),
    (
        "3-3 Threading the needle",
        np.array([
            [1, 1,  -1, -1, -1],
            [-2, -3, -1, -3, -2],
            [1, -3, -1, -1, 1],
            [-1, -2, 0, -1, 1],
            [-1, -1, -1, -1, 0]
        ])
    ),
    (
        "3-4 Locked and loaded",
        np.array([
            [-2, -2, -1,  -1, -1, -1, -1, -1, -2, -2],
            [-2, -2, -1,  -1, -1, 1, 1, -1, -2, -2],
            [1, 1, -1,  -1, -2, 1, -1, -1, 0, 0],
            [1, -1, -3,  -3, -1, -1, -1, -1, -2, 0],
            [-2, -2, -1,  -1, -1, 0, -1, -1, -2, -2],
            [-2, -2, -1,  -1, 0, 0, 0, 1, -2, -2],
        ])
    ),
    (
        "3-5 Bomb carpeting",
        np.array([
            [2, -1, 1,  -1, -1, 0,-1],
            [-1, -1, 1,  -1, -1, 0, -1],
            [0, -1, -2,  1, -1, 2, -1],
            [-2, -1, -1,  2, -1, -2, -1],
            [-1, -1, -1,  -2, -1, -1, -1],
            [-3, -3, -3,  -3, -2, -2, -2],
        ])
    ),
]
