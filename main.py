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

claimed = np.zeros(state.shape)

for x in range(4):
    for y in range(4):
        if state[x, y] == -1 or claimed[x, y] == 1:
            continue
        # We have found a new block
        # Flood fill to get all of the block tiles
        block_identity = state[x, y]
        block_tiles = [(x, y)]
        frontier = [(x, y)]
        claimed[(x, y)] = 1

        while len(frontier) > 0:
            x_frontier, y_frontier = frontier.pop()

            frontier_candidates = []
            # Up
            if y_frontier > 0:
                frontier_candidates.append((x_frontier, y_frontier - 1))

            # Down
            if y_frontier < state.shape[0] - 1:
                frontier_candidates.append((x_frontier, y_frontier + 1))

            # Left
            if x_frontier > 0:
                frontier_candidates.append((x_frontier - 1, y_frontier))

            # Right
            if x_frontier < state.shape[1] - 1:
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

print(blocks)

# For now, lets just fall down
while True:
    grounded = np.zeros(state.shape)
    for current_ground_check_level in range(state.shape[0])

# Run the falling step until every block is grounded
# Check if anything below you is grounded
# If for every tile a block, nothing below is grounded
# move that block down one in the next step
