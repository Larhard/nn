BLOCK_DIM_1D = (1024, 1, 1)
BLOCK_DIM_2D = (32, 32, 1)


def get_grid_dim(n, block_dim):
    return (n - 1) // block_dim + 1


def get_dims_1d(x):
    return BLOCK_DIM_1D, (get_grid_dim(x, BLOCK_DIM_1D[0]), 1, 1)


def get_dims_2d(x, y):
    return BLOCK_DIM_2D, (get_grid_dim(x, BLOCK_DIM_2D[0]), get_grid_dim(y, BLOCK_DIM_2D[1]), 1)
