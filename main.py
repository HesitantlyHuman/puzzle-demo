from typing import List, Tuple

import numpy as np


def parse_blocks(state: np.ndarray) -> List[Tuple[int, List[Tuple[int, int]]]]:
    blocks = []
    claimed = np.zeros(state.shape, dtype=np.int8)

    for x in range(state.shape[0]):
        for y in range(state.shape[1]):
            if state[x, y] == -1 or claimed[x, y] == 1:
                continue
            # We have found a new block
            # Flood fill to get all of the block tiles
            block_identity = state[x, y]
            block_tiles = [(x, y)]
            frontier = [(x, y)]
            claimed[x, y] = 1

            while len(frontier) > 0:
                x_frontier, y_frontier = frontier.pop()

                frontier_candidates = []
                # Up
                if y_frontier > 0:
                    frontier_candidates.append((x_frontier, y_frontier - 1))

                # Down
                if y_frontier < state.shape[1] - 1:
                    frontier_candidates.append((x_frontier, y_frontier + 1))

                # Left
                if x_frontier > 0:
                    frontier_candidates.append((x_frontier - 1, y_frontier))

                # Right
                if x_frontier < state.shape[0] - 1:
                    frontier_candidates.append((x_frontier + 1, y_frontier))

                for frontier_candidate in frontier_candidates:
                    if (
                        state[frontier_candidate] == block_identity
                        and claimed[frontier_candidate] == 0
                    ):
                        claimed[frontier_candidate] = 1
                        block_tiles.append(frontier_candidate)
                        frontier.append(frontier_candidate)

            blocks.append((block_identity, block_tiles))
    return blocks


def generate_state(blocks, size):
    state = np.zeros(size, dtype=np.int8) - 1
    block_map = np.zeros(size, dtype=np.int8) - 1
    for block_id, (block_identity, tiles) in enumerate(blocks):
        for tile in tiles:
            state[tile] = block_identity
            block_map[tile] = block_id
    return state, block_map


def down(blocks, size):
    blocks = blocks.copy()
    bombs = [block for block in blocks if block[0] == -3]
    blocks = [block for block in blocks if block[0] != -3]
    state, block_map = generate_state(blocks, size)
    bomb_tiles = np.zeros(state.shape, dtype=np.int8) - 1
    for bomb_index, (_, tiles) in enumerate(bombs):
        for tile in tiles:
            bomb_tiles[tile] = bomb_index
    steps = []
    while True:
        # First, check which blocks are grounded
        block_is_grounded = [False for _ in blocks]
        tile_is_grounded = np.zeros(state.shape, dtype=np.int8)
        current_ground_check_level = state.shape[1] - 1
        while current_ground_check_level >= 0:
            current_ground_check_x = 0
            while current_ground_check_x < state.shape[0]:
                if (
                    state[current_ground_check_x, current_ground_check_level] == -1
                    or tile_is_grounded[
                        current_ground_check_x, current_ground_check_level
                    ]
                    == 1
                ):
                    current_ground_check_x += 1
                    continue

                if state[current_ground_check_x, current_ground_check_level] == -2:
                    block = block_map[
                        current_ground_check_x, current_ground_check_level
                    ]
                    block_is_grounded[block] = True
                    bottommost_y = current_ground_check_level
                    for tile in blocks[block][1]:
                        bottommost_y = max(bottommost_y, tile[1])
                        tile_is_grounded[tile] = 1
                    if current_ground_check_level <= bottommost_y - 1:
                        current_ground_check_level = bottommost_y - 1
                        current_ground_check_x = 0
                    continue

                if (
                    current_ground_check_level == state.shape[1] - 1
                    or tile_is_grounded[
                        current_ground_check_x, current_ground_check_level + 1
                    ]
                    == 1
                ):
                    block = block_map[
                        current_ground_check_x, current_ground_check_level
                    ]
                    block_is_grounded[block] = True
                    bottommost_y = current_ground_check_level
                    for tile in blocks[block][1]:
                        bottommost_y = max(bottommost_y, tile[1])
                        tile_is_grounded[tile] = 1
                    if bottommost_y > current_ground_check_level:
                        current_ground_check_level = bottommost_y
                        current_ground_check_x = 0

                current_ground_check_x += 1
            current_ground_check_level -= 1

        # End condition
        if all(block_is_grounded):
            break

        bomb_has_gone_off = False
        # Now move all blocks that are not grounded down
        new_state = np.zeros(state.shape, dtype=np.int8) - 1
        new_block_map = np.zeros(state.shape, dtype=np.int8) - 1
        new_blocks = []
        for block_idx, (is_grounded, block) in enumerate(
            zip(block_is_grounded, blocks)
        ):
            identity, tiles = block
            if not is_grounded:
                tiles = [(tile_x, tile_y + 1) for tile_x, tile_y in tiles]
            for tile in tiles:
                if bomb_tiles[tile] >= 0:
                    # Change the bomb in question to a explosion
                    bomb = bombs[bomb_tiles[tile]]
                    bombs[bomb_tiles[tile]] = (-4, bomb[1])
                    bomb_has_gone_off = True
                new_state[tile] = identity
                new_block_map[tile] = block_idx
            new_blocks.append((identity, tiles))

        state = new_state
        blocks = new_blocks
        block_map = new_block_map
        blocks_and_bombs = blocks + bombs
        steps.append(blocks_and_bombs)
        if bomb_has_gone_off:
            return steps, True
    return steps, False


def up(blocks, size):
    # size is the same, flip the y
    max_y = size[1] - 1

    new_blocks = []
    for i, tiles in blocks:
        new_tiles = [(x, max_y - y) for x, y in tiles]
        new_blocks.append((i, new_tiles))

    steps, bombs_gone_off = down(new_blocks, size)

    new_steps = []
    for step in steps:
        new_step = []
        for i, tiles in step:
            new_tiles = [(x, max_y - y) for x, y in tiles]
            new_step.append((i, new_tiles))
        new_steps.append(new_step)
    return new_steps, bombs_gone_off


def right(blocks, size):
    new_size = (size[1], size[0])
    new_blocks = []
    for i, tiles in blocks:
        new_tiles = [(y, x) for x, y in tiles]
        new_blocks.append((i, new_tiles))

    steps, bombs_gone_off = down(new_blocks, new_size)

    new_steps = []
    for step in steps:
        new_step = []
        for i, tiles in step:
            new_tiles = [(y, x) for x, y in tiles]
            new_step.append((i, new_tiles))
        new_steps.append(new_step)
    return new_steps, bombs_gone_off


def left(blocks, size):
    magic_val = size[0] - 1
    new_blocks = []
    for i, tiles in blocks:
        new_tiles = [(y, magic_val - x) for x, y in tiles]
        new_blocks.append((i, new_tiles))
    new_size = (size[1], size[0])

    steps, bombs_gone_off = down(new_blocks, new_size)

    new_steps = []
    for step in steps:
        new_step = []
        for i, tiles in step:
            new_tiles = [(magic_val - y, x) for x, y in tiles]
            new_step.append((i, new_tiles))
        new_steps.append(new_step)

    return new_steps, bombs_gone_off


def has_won(blocks):
    block_identities = set()
    for identity, _ in blocks:
        if not identity in [-2, -3] and identity in block_identities:
            return False
        block_identities.add(identity)
    return True


import tkinter


def center(win):
    """
    centers a tkinter window
    :param win: the main window or Toplevel window to center
    """
    win.update_idletasks()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2
    win.geometry("{}x{}+{}+{}".format(width, height, x, y))
    win.deiconify()


root = tkinter.Tk()
root.geometry("720x770")
root.title("unnamed puzzle game")
root.resizable(False, False)
canvas = tkinter.Canvas(root, width=720, height=770)
canvas.pack()
center(root)
top_padding = 50
canvas_padding = 100
play_area = 720 - 2 * canvas_padding

current_puzzle_is_failed = False
puzzle_name = None
puzzle_size = None
puzzle_height = None
puzzle_width = None
blocks = None
tile_size = None
puzzle_x_offset = None
puzzle_y_offset = None
current_puzzle_index = 0
next_puzzle_available = False
prev_puzzle_available = False
next_level_button = None
prev_level_button = None
bomb_image = tkinter.PhotoImage(file="bomb.png").subsample(20, 20)
explosion_image = tkinter.PhotoImage(file="explosion.png").subsample(6, 6)

from puzzles import puzzles

puzzle_is_solved = [False for _ in puzzles]


def go_to_next_level():
    global current_puzzle_index
    global next_puzzle_available
    global prev_puzzle_available
    current_puzzle_index += 1
    current_puzzle_index = min(len(puzzles) - 1, current_puzzle_index)
    next_puzzle_available = (
        current_puzzle_index < len(puzzles) - 1
        and puzzle_is_solved[current_puzzle_index]
    )
    prev_puzzle_available = current_puzzle_index > 0
    setup_puzzle(puzzles[current_puzzle_index][0], puzzles[current_puzzle_index][1])


def go_to_prev_level():
    global current_puzzle_index
    global next_puzzle_available
    global prev_puzzle_available
    current_puzzle_index -= 1
    current_puzzle_index = max(0, current_puzzle_index)
    next_puzzle_available = (
        current_puzzle_index < len(puzzles) - 1
        and puzzle_is_solved[current_puzzle_index]
    )
    prev_puzzle_available = current_puzzle_index > 0
    setup_puzzle(puzzles[current_puzzle_index][0], puzzles[current_puzzle_index][1])


def setup_puzzle(name, state):
    state = state.T
    global current_puzzle_is_failed
    global puzzle_name
    global puzzle_size
    global puzzle_height
    global puzzle_width
    global blocks
    global tile_size
    global puzzle_x_offset
    global puzzle_y_offset
    global next_puzzle_available
    global prev_puzzle_available
    global next_level_button
    global prev_level_button
    puzzle_name = name
    puzzle_size = state.shape
    puzzle_height = puzzle_size[1]
    puzzle_width = puzzle_size[0]
    blocks = parse_blocks(state)
    tile_size = play_area / (max(puzzle_height, puzzle_width) + 2)
    puzzle_x_offset = (max(0, puzzle_height - puzzle_width) * tile_size) / 2
    puzzle_y_offset = (max(0, puzzle_width - puzzle_height) * tile_size) / 2
    canvas.delete("all")
    draw_level_text()
    draw_background()
    draw_blocks(blocks)
    prev_level_button = tkinter.Button(
        canvas, text="<", font=("Helvetica 20 bold"), command=go_to_prev_level
    )
    prev_level_button.config(height=1, width=1)
    prev_level_button.place(x=30, y=390)
    prev_level_button["state"] = "normal" if prev_puzzle_available else "disabled"
    next_level_button = tkinter.Button(
        canvas, text=">", font=("Helvetica 20 bold"), command=go_to_next_level
    )
    next_level_button.config(height=1, width=1)
    next_level_button.place(x=650, y=390)
    next_level_button["state"] = "normal" if next_puzzle_available else "disabled"
    retry_button = tkinter.Button(
        canvas,
        text="Reset",
        font=("Helvetica 20 bold"),
        command=lambda: setup_puzzle(name, state.T),
    )
    retry_button.config(height=1, width=6)
    retry_button.place(x=310, y=700)
    current_puzzle_is_failed = False


def draw_background():
    canvas.create_rectangle(
        canvas_padding,
        canvas_padding + top_padding,
        720 - canvas_padding,
        720 - canvas_padding + top_padding,
        fill="#6c757d",
        width=0,
        tag="blocks"
    )
    canvas.create_rectangle(
        canvas_padding + tile_size + puzzle_x_offset,
        canvas_padding + top_padding + tile_size + puzzle_y_offset,
        720 - canvas_padding - tile_size - puzzle_x_offset,
        720 - canvas_padding + top_padding - tile_size - puzzle_y_offset,
        fill="#edf2f4",
        width=0,
        tag="blocks"
    )


colors = {-1: "#edf2f4", -2: "#6c757d", 0: "#d90429", 1: "#3a5a40", 2: "#ffbf00"}


def draw_tile(x, y, type):
    x_start = (x + 1) * tile_size + canvas_padding + puzzle_x_offset
    y_start = (y + 1) * tile_size + canvas_padding + top_padding + puzzle_y_offset
    x_end = x_start + tile_size
    y_end = y_start + tile_size
    if type == -3:
        canvas.create_image(
            (x_start + x_end) / 2,
            (y_start + y_end) / 2,
            image=bomb_image,
            anchor="center",
            tag="blocks"
        )
    elif type == -4:
        canvas.create_image(
            (x_start + x_end) / 2,
            (y_start + y_end) / 2,
            image=explosion_image,
            anchor="center",
            tag="blocks"
        )
    else:
        canvas.create_rectangle(
            x_start, y_start, x_end, y_end, fill=colors[type], width=0, tag="blocks"
        )


def draw_level_text():
    lines = puzzle_name.split("\n")
    line_height = 30
    available_space = 150
    line_spacing = 10
    leftover_space = (
        available_space - (len(lines) * line_height) - ((len(lines) - 1) * line_spacing)
    )
    top_pad = leftover_space / 2
    for line in lines:
        text = canvas.create_text(
            0, top_pad, text=line, anchor="nw", font=("Helvetica 30")
        )
        coords = canvas.bbox(text)
        x_offset = (720 / 2) - ((coords[2] - coords[0]) / 2)
        canvas.move(text, x_offset, 0)
        top_pad += line_height + line_spacing


def draw_blocks(blocks):
    for block_identity, tiles in blocks:
        for tile in tiles:
            draw_tile(tile[0], tile[1], block_identity)


def draw_win():
    canvas.create_text(365, 400, text="Good Job!", font=("Helvetica 42 bold"))

def draw_lose():
    canvas.create_text(365, 400, text="Oops...", font=("Helvetica 42 bold"))
    
def draw_final_message():
    canvas.create_text(355, 400, text="Thanks for playing!", font=("Helvetica 42 bold"))


setup_puzzle(puzzles[current_puzzle_index][0], puzzles[current_puzzle_index][1])

animating = False
animation_frames = []

def handle_animation():
    global animation_frames
    global animating
    global puzzle_is_solved
    if len(animation_frames) == 0:
        if all(puzzle_is_solved):
            draw_final_message()
            animating = False
            return
        if puzzle_is_solved[current_puzzle_index]:
            draw_win()
        if current_puzzle_is_failed:
            draw_lose()
        animating = False
        return
    canvas.delete("blocks")
    to_draw = animation_frames.pop(0)
    draw_background()
    draw_blocks(to_draw)
    root.after(50, handle_animation)

def reset_animation():
    global animation_frames
    global animating
    animating = False
    animation_frames = []


def handle_input(event):
    global blocks
    global animating
    global next_level_button
    global prev_level_button
    global next_puzzle_available
    global prev_puzzle_available
    global current_puzzle_is_failed
    print(f"Pressed key code: {event.keycode}")
    if event.keycode in [111, 38] and not current_puzzle_is_failed:
        steps, failed = up(blocks, puzzle_size)
        current_puzzle_is_failed = failed
        animation_frames.extend(steps)
        if len(steps) > 0:
            blocks = steps[-1]
            state, _ = generate_state(blocks, puzzle_size)
            blocks = parse_blocks(state)

            if not animating:
                animating = True
                handle_animation()

                if not current_puzzle_is_failed and has_won(blocks):
                    puzzles[current_puzzle_index] = (puzzles[current_puzzle_index][0], state.T)
                    puzzle_is_solved[current_puzzle_index] = True
                    next_puzzle_available = current_puzzle_index < len(puzzles) - 1
                    prev_puzzle_available = current_puzzle_index > 0
                    prev_level_button["state"] = "normal" if prev_puzzle_available else "disabled"
                    next_level_button["state"] = "normal" if next_puzzle_available else "disabled"

    elif event.keycode in [116, 40] and not current_puzzle_is_failed:
        steps, failed = down(blocks, puzzle_size)
        current_puzzle_is_failed = failed
        animation_frames.extend(steps)
        if len(steps) > 0:
            blocks = steps[-1]
            state, _ = generate_state(blocks, puzzle_size)
            blocks = parse_blocks(state)

            if not animating:
                animating = True
                handle_animation()

                if not current_puzzle_is_failed and has_won(blocks):
                    puzzles[current_puzzle_index] = (puzzles[current_puzzle_index][0], state.T)
                    puzzle_is_solved[current_puzzle_index] = True
                    next_puzzle_available = current_puzzle_index < len(puzzles) - 1
                    prev_puzzle_available = current_puzzle_index > 0
                    prev_level_button["state"] = "normal" if prev_puzzle_available else "disabled"
                    next_level_button["state"] = "normal" if next_puzzle_available else "disabled"
    elif event.keycode in [113, 37] and not current_puzzle_is_failed:
        steps, failed = left(blocks, puzzle_size)
        current_puzzle_is_failed = failed
        animation_frames.extend(steps)
        if len(steps) > 0:
            blocks = steps[-1]
            state, _ = generate_state(blocks, puzzle_size)
            blocks = parse_blocks(state)

            if not animating:
                animating = True
                handle_animation()

                if not current_puzzle_is_failed and has_won(blocks):
                    puzzles[current_puzzle_index] = (puzzles[current_puzzle_index][0], state.T)
                    puzzle_is_solved[current_puzzle_index] = True
                    next_puzzle_available = current_puzzle_index < len(puzzles) - 1
                    prev_puzzle_available = current_puzzle_index > 0
                    prev_level_button["state"] = "normal" if prev_puzzle_available else "disabled"
                    next_level_button["state"] = "normal" if next_puzzle_available else "disabled"
    elif event.keycode in [114, 39] and not current_puzzle_is_failed:
        steps, failed = right(blocks, puzzle_size)
        current_puzzle_is_failed = failed
        animation_frames.extend(steps)
        if len(steps) > 0:
            blocks = steps[-1]
            state, _ = generate_state(blocks, puzzle_size)
            blocks = parse_blocks(state)

            if not animating:
                animating = True
                handle_animation()

                if not current_puzzle_is_failed and has_won(blocks):
                    puzzles[current_puzzle_index] = (puzzles[current_puzzle_index][0], state.T)
                    puzzle_is_solved[current_puzzle_index] = True
                    next_puzzle_available = current_puzzle_index < len(puzzles) - 1
                    prev_puzzle_available = current_puzzle_index > 0
                    prev_level_button["state"] = "normal" if prev_puzzle_available else "disabled"
                    next_level_button["state"] = "normal" if next_puzzle_available else "disabled"
    elif event.keycode in [24, 81]:
        root.destroy()


if __name__ == "__main__":
    root.bind("<Key>", handle_input)
    root.mainloop()
