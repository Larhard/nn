import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


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

    x, y = p.shape
    yy, z = q.shape
    assert (y == yy)

    out = np.zeros((x, z), dtype=np.float64)
    cuda_matrix_multiply(drv.Out(out), drv.In(p), drv.In(q),
        np.int32(x), np.int32(y), np.int32(z),
        block=(32, 32, 1), grid=(1, 1, 1))
    return out
