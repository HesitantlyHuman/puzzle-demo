import numpy as np

# fmt: off
state = np.array([
    [-1, -1,  0, -1],
    [-1, -1,  0,  1],
    [-1,  1,  1,  0],
    [ 1,  0, -1, -1]
])
# fmt: on

state = state.T

blocks = []

claimed = np.zeros((4, 4))

for x in range(4):
    for y in range(4):
        if state[x, y] == -1 or claimed[x, y] == 1:
            continue
        # We have found a new block
        # Flood fill to get all of the block tiles
        block_identity = state[x, y]
        block_tiles = [(x, y)]
        to_search = [(x, y)]
        claimed[(x, y)] == 1

        while len(to_search) > 0:
            x_to_search, y_to_search = to_search.pop()

            # Up
            if (
                y_to_search > 0
                and state[x_to_search, y_to_search - 1] == block_identity
                and claimed[x_to_search, y_to_search - 1] == 0
            ):
                new_block_tile = (x_to_search, y_to_search - 1)
                claimed[new_block_tile] = 1
                block_tiles.append(new_block_tile)
                to_search.append(new_block_tile)

            # Down
            if (
                y_to_search < 3
                and state[x_to_search, y_to_search + 1] == block_identity
                and claimed[x_to_search, y_to_search + 1] == 0
            ):
                new_block_tile = (x_to_search, y_to_search + 1)
                claimed[new_block_tile] = 1
                block_tiles.append(new_block_tile)
                to_search.append(new_block_tile)

            # Left
            if (
                x_to_search > 0
                and state[x_to_search - 1, y_to_search] == block_identity
                and claimed[x_to_search - 1, y_to_search] == 0
            ):
                new_block_tile = (x_to_search - 1, y_to_search)
                claimed[new_block_tile] = 1
                block_tiles.append(new_block_tile)
                to_search.append(new_block_tile)

            # Right
            if (
                x_to_search < 3
                and state[x_to_search + 1, y_to_search] == block_identity
                and claimed[x_to_search + 1, y_to_search] == 0
            ):
                new_block_tile = (x_to_search + 1, y_to_search)
                claimed[new_block_tile] = 1
                block_tiles.append(new_block_tile)
                to_search.append(new_block_tile)

        blocks.append((block_identity, block_tiles))


print(blocks)


blocks = [
    ("red", [(2, 0), (2, 1)]),
    ("green", [(3, 1)]),
    ("green", [(1, 2), (2, 2)]),
    ("red", [(3, 2)]),
    ("green", [(0, 3)]),
    ("red", [(1, 3)]),
]

color_lookup = {"red": 0, "green": 1}

block_map = np.zeros((4, 4)) - 1
color_map = np.zeros((4, 4)) - 1

for block_idx, (color, tiles) in enumerate(blocks):
    color_idx = color_lookup[color]
    for tile in tiles:
        block_map[tile] = block_idx
        color_map[tile] = color_idx

print(block_map.T)
print(color_map.T)

block_locations = {}

# Run the falling step until every block is grounded
# Check if anything below you is grounded
# If for every tile a block, nothing below is grounded
# move that block down one in the next step
