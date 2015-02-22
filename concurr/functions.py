import pycuda.autoinit

import numpy as np
import functools
import operator

from pycuda.compiler import SourceModule
from pycuda import gpuarray

import concurr.utils


module = SourceModule("""
__global__
void sgm(double *odata, double *idata, int n)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < n) {
        double tmp = idata[idx];
        tmp = 1 / (1 + exp(-tmp));
        odata[idx] = tmp;
    }
}

__global__
void sgm_d(double *odata, double *idata, int n)
{
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < n) {
        double tmp = idata[idx];
        tmp = 1 / (1 + exp(-tmp));
        tmp = tmp * (1 - tmp);
        odata[idx] = tmp;
    }
}
""")


cuda_sgm = module.get_function('sgm')
cuda_sgm_d = module.get_function('sgm_d')


def sgm(idata):
    if not isinstance(idata, gpuarray.GPUArray):
        idata = gpuarray.to_gpu(np.ascontiguousarray(idata))

    odata = gpuarray.GPUArray(idata.shape, dtype=np.float64)

    n = functools.reduce(operator.mul, idata.shape)
    block_dim, grid_dim = concurr.utils.get_dims_1d(n)

    cuda_sgm(odata, idata, np.int32(n),
        block=block_dim, grid=grid_dim)

    return odata


def sgm_d(idata):
    if not isinstance(idata, gpuarray.GPUArray):
        idata = gpuarray.to_gpu(np.ascontiguousarray(idata))

    odata = gpuarray.GPUArray(idata.shape, dtype=np.float64)

    n = functools.reduce(operator.mul, idata.shape)
    block_dim, grid_dim = concurr.utils.get_dims_1d(n)

    cuda_sgm_d(odata, idata, np.int32(n),
        block=block_dim, grid=grid_dim)

    return odata
