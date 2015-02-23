import functools
import operator

import numpy as np
import numpy.testing

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda import gpuarray

import concurr.utils


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
void matrix_add(double *dest, double *src, double *val, int n)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const double value = *val;

    if (idx < n) {
        double tmp = src[idx];
        // printf("%lf + %lf = %lf\\n", tmp, value, tmp + value);
        tmp += value;
        dest[idx] = tmp;
    }
}

__global__
void matrix_sum(double *dest, double *src, double *val, int n)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx < n) {
        double p = src[idx];
        double q = val[idx];
        double tmp = p + q;
        dest[idx] = tmp;
    }
}

__global__
void matrix_mul(double *dest, double *src, double *val, int n)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const double value = *val;

    if (idx < n) {
        double tmp = src[idx];
        tmp *= value;
        dest[idx] = tmp;
    }
}

__global__
void matrix_transpose(double *odata, double *idata, int x, int y)
{
    const int idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int idx_y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (idx_x < x && idx_y < y) {
        double tmp = get2d(idata, idx_y, idx_x, x);
        get2d(odata, idx_x, idx_y, y) = tmp;
    }
}

__global__
void matrix_append_value(double *odata, double *idata, double *val, int n, int m)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    const double value = idx < n ? idata[idx] : *val;

    if (idx < m) {
        odata[idx] = value;
    }
}

__global__
void matrix_cart_mul_sum(double *odata, double *p, double *q, int x, int y, int z)
{
    const int idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int idx_y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (idx_x < x && idx_y < z) {
        double tmp = 0;
        double tp;
        double tq;
        for (int i = 0; i < y; ++i) {
            tp = get2d(p, idx_x, i, y);
            tq = get2d(q, idx_y, i, y);
            tmp += tp * tq;
        }
        get2d(odata, idx_y, idx_x, x) = tmp;
    }
}

""")

cuda_matrix_multiply = cuda_matrix.get_function('matrix_multiply')
cuda_matrix_multiply_tn = cuda_matrix.get_function('matrix_multiply_tn')
cuda_matrix_transpose = cuda_matrix.get_function('matrix_transpose')

cuda_matrix_add = cuda_matrix.get_function('matrix_add')
cuda_matrix_sum = cuda_matrix.get_function('matrix_sum')
cuda_matrix_mul = cuda_matrix.get_function('matrix_mul')
cuda_matrix_append_value = cuda_matrix.get_function('matrix_append_value')
cuda_cart_mul_sum = cuda_matrix.get_function('matrix_cart_mul_sum')

def multiply(p, q):
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

    out = gpuarray.GPUArray((x, z), dtype=np.float64)

    block_x = 32
    block_y = 32
    grid_x = (z - 1) // block_x + 1
    grid_y = (x - 1) // block_y + 1
    # print("{} x {} x {} : {}x{} / {}x{}".format(x, y, z, block_x, block_y, grid_x, grid_y))
    cuda_matrix_multiply(out, p, q,
        np.int32(x), np.int32(y), np.int32(z),
        block=(block_x, block_y, 1), grid=(grid_x, grid_y, 1))
    return out


def multiply_tn(p, q):
    assert p.dtype == np.float64
    assert q.dtype == np.float64
    y, x = p.shape
    yy, z = q.shape
    assert (y == yy)
    if not isinstance(p, gpuarray.GPUArray):
        p = gpuarray.to_gpu(np.ascontiguousarray(p))
    if not isinstance(q, gpuarray.GPUArray):
        q = gpuarray.to_gpu(np.ascontiguousarray(q))

    out = gpuarray.GPUArray((x, z), dtype=np.float64)

    block_x = 32
    block_y = 32
    grid_x = (z - 1) // block_x + 1
    grid_y = (x - 1) // block_y + 1
    cuda_matrix_multiply_tn(out, p, q,
        np.int32(x), np.int32(y), np.int32(z),
        block=(block_x, block_y, 1), grid=(grid_x, grid_y, 1))
    return out


def add(matrix, value):
    """
    add scalar to matrix
    """
    if not isinstance(matrix, gpuarray.GPUArray):
        matrix = gpuarray.to_gpu(np.ascontiguousarray(matrix))

    n = functools.reduce(operator.mul, matrix.shape)

    block_x = 1024
    grid_x = (n - 1) // block_x + 1

    out = gpuarray.GPUArray(matrix.shape, dtype=np.float64)
    val = gpuarray.to_gpu(np.array(value, dtype=np.float64))

    cuda_matrix_add(
        out, matrix,
        val,
        np.int32(n),
        block=(block_x, 1, 1), grid=(grid_x, 1, 1)
    )

    # cuda_matrix_add(out, matrix, np.float64(value), np.int32(x), np.int32(y),
    #     block=(block_x, block_y, 1), grid=(grid_x, grid_y, 1))
    return out


def sum(matrix, value):
    """
    sum two matrices
    """
    if not isinstance(matrix, gpuarray.GPUArray):
        matrix = gpuarray.to_gpu(np.ascontiguousarray(matrix))
    if not isinstance(value, gpuarray.GPUArray):
        value = gpuarray.to_gpu(np.ascontiguousarray(value))

    n = functools.reduce(operator.mul, matrix.shape)
    assert matrix.shape == value.shape

    block_x = 1024
    grid_x = (n - 1) // block_x + 1

    out = gpuarray.GPUArray(matrix.shape, dtype=np.float64)

    cuda_matrix_sum(
        out, matrix,
        value,
        np.int32(n),
        block=(block_x, 1, 1), grid=(grid_x, 1, 1)
    )

    return out


def mul(matrix, value):
    """
    add scalar to matrix
    """
    if not isinstance(matrix, gpuarray.GPUArray):
        matrix = gpuarray.to_gpu(np.ascontiguousarray(matrix))

    n = functools.reduce(operator.mul, matrix.shape)

    block, grid = concurr.utils.get_dims_1d(n)

    out = gpuarray.GPUArray(matrix.shape, dtype=np.float64)
    val = gpuarray.to_gpu(np.array(value, dtype=np.float64))

    cuda_matrix_mul(
        out, matrix,
        val,
        np.int32(n),
        block=block, grid=grid
    )

    return out


def transpose(idata):
    if not isinstance(idata, gpuarray.GPUArray):
        idata = gpuarray.to_gpu(np.ascontiguousarray(idata))

    y, x = idata.shape

    odata = gpuarray.GPUArray((x, y), dtype=np.float64)

    block_dim, grid_dim = concurr.utils.get_dims_2d(x, y)

    cuda_matrix_transpose(
        odata, idata,
        np.int32(x), np.int32(y),
        block=block_dim, grid=grid_dim
    )

    return odata


def append_value_line(idata, value):
    """
    appends row filled with value to the bottom of the matrix
    """
    if not isinstance(idata, gpuarray.GPUArray):
        idata = gpuarray.to_gpu(np.ascontiguousarray(idata))

    y, x = idata.shape

    odata = gpuarray.GPUArray((y + 1, x), dtype=np.float64)
    val = gpuarray.to_gpu(np.array(value, dtype=np.float64))

    n = x * y
    m = n + x

    block, grid = concurr.utils.get_dims_1d(m)

    cuda_matrix_append_value(
        odata, idata, val,
        np.int32(n), np.int32(m),
        block=block, grid=grid
    )

    return odata


def cart_mul_sum(p, q):
    if not isinstance(p, gpuarray.GPUArray):
        p = gpuarray.to_gpu(np.ascontiguousarray(p))
    if not isinstance(q, gpuarray.GPUArray):
        q = gpuarray.to_gpu(np.ascontiguousarray(q))
    x, y = p.shape
    z, yy = q.shape
    assert y == yy
    odata = gpuarray.GPUArray((z, x), dtype=np.float64)

    block, grid = concurr.utils.get_dims_2d(x, z)

    cuda_cart_mul_sum(
        odata, p, q,
        np.int32(x), np.int32(y), np.int32(z),
        block=block, grid=grid
    )
    return odata
