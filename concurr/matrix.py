import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda import gpuarray


cuda_matrix = SourceModule("""
#include <stdio.h>

#define get2d(a,p,q,n) (a[(p)*(n)+(q)])

__global__
void matrix_multiply(double *dest, double *p, double *q, int x, int y, int z)
{
    const int idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int idx_y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (idx_y < x && idx_x < z) {
        double tmp = 0;
        for (int i = 0; i < y; ++i) {
            tmp += get2d(p, idx_y, i, y) * get2d(q, i, idx_x, z);
        }
        get2d(dest, idx_y, idx_x, z) = tmp;
    }
}

""")

cuda_matrix_multiply = cuda_matrix.get_function('matrix_multiply')


def matrix_multiply(p, q):
    """

         y         z           z
       +----     +----       +----
       |         |           |
     x |       y |       = x |
       |         |           |

    """

    assert p.dtype == np.float64
    assert q.dtype == np.float64
    x, y = p.shape
    yy, z = q.shape
    assert (y == yy)
    if not isinstance(p, gpuarray.GPUArray):
        p = gpuarray.to_gpu(np.ascontiguousarray(p))
    if not isinstance(q, gpuarray.GPUArray):
        q = gpuarray.to_gpu(np.ascontiguousarray(q))

    out = np.zeros((x, z), dtype=np.float64)

    block_x = 32
    block_y = 32
    grid_x = (z - 1) // block_x + 1
    grid_y = (x - 1) // block_y + 1
    # print("{} x {} x {} : {}x{} / {}x{}".format(x, y, z, block_x, block_y, grid_x, grid_y))
    cuda_matrix_multiply(drv.Out(out), p, q,
        np.int32(x), np.int32(y), np.int32(z),
        block=(block_x, block_y, 1), grid=(grid_x, grid_y, 1))
    return out
