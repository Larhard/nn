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

__global__
void matrix_multiply_tn(double *dest, double *p, double *q, int x, int y, int z)
{
    const int idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int idx_y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (idx_y < x && idx_x < z) {
        double tmp = 0;
        for (int i = 0; i < y; ++i) {
            tmp += get2d(p, i, idx_y, x) * get2d(q, i, idx_x, z);
        }
        get2d(dest, idx_y, idx_x, z) = tmp;
    }
}

__global__
void matrix_add(double *dest, double *src, int x, int y, double* val)
{
    const int idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int idx_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    const double value = *val;

    if (idx_y < y && idx_x < x) {
        double tmp = get2d(src, idx_y, idx_x, x);
        tmp += value;
        get2d(dest, idx_y, idx_x, x) = 0;
        get2d(dest, idx_y, idx_x, x) = tmp;
    }
}

""")

cuda_matrix_multiply = cuda_matrix.get_function('matrix_multiply')
cuda_matrix_multiply_tn = cuda_matrix.get_function('matrix_multiply_tn')

cuda_matrix_add = cuda_matrix.get_function('matrix_add')


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


def matrix_multiply_tn(p, q):
    assert p.dtype == np.float64
    assert q.dtype == np.float64
    y, x = p.shape
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
    cuda_matrix_multiply_tn(drv.Out(out), p, q,
        np.int32(x), np.int32(y), np.int32(z),
        block=(block_x, block_y, 1), grid=(grid_x, grid_y, 1))
    return out


def matrix_add(matrix, value):
    if not isinstance(matrix, gpuarray.GPUArray):
        matrix = gpuarray.to_gpu(np.ascontiguousarray(matrix))

    y, x = matrix.shape

    block_x = 32
    block_y = 32
    grid_x = (x - 1) // block_x + 1
    grid_y = (y - 1) // block_y + 1

    out = gpuarray.GPUArray((y, x), dtype=np.float64)
    val = gpuarray.to_gpu(np.array(value))

    cuda_matrix_add(
        out, matrix,
        np.int32(x), np.int32(y),
        val,
        block=(block_x, block_y, 1), grid=(grid_x, grid_y, 1)
    )

    # cuda_matrix_add(out, matrix, np.float64(value), np.int32(x), np.int32(y),
    #     block=(block_x, block_y, 1), grid=(grid_x, grid_y, 1))
    return out
