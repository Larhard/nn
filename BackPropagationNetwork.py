import pycuda.autoinit

import numpy as np
from pycuda import curandom
from pycuda import gpuarray

import concurr


class TransferFunctions:
    @staticmethod
    def sgm(x, derivative=False):
        if isinstance(x, gpuarray.GPUArray):
            x = x.get()
        if not derivative:
            return 1 / (1 + np.exp(-x))
        else:
            out = TransferFunctions.sgm(x)
            return out * (1 - out)

    @staticmethod
    def sgm2(x, derivative=False):
        if isinstance(x, gpuarray.GPUArray):
            x = x.get()
        if not derivative:
            return 1 / (1 + np.exp(5-x))
        else:
            out = TransferFunctions.sgm(x)
            return out * (1 - out)

    @staticmethod
    def linear(x, derivative=False):
        if isinstance(x, gpuarray.GPUArray):
            x = x.get()
        if not derivative:
            return x
        else:
            return 1.0

    @staticmethod
    def gaussian(x, derivative=False):
        if isinstance(x, gpuarray.GPUArray):
            x = x.get()
        if not derivative:
            return np.exp(-x**2)
        else:
            return -2 * x * np.exp(-x**2)

    @staticmethod
    def tanh(x, derivative=False):
        if isinstance(x, gpuarray.GPUArray):
            x = x.get()
        if not derivative:
            return np.tanh(x)
        else:
            return 1.0 - np.tanh(x)**2


class NeuralNetwork:
    """Back-propagation network"""

    layer_count = 0
    shape = None
    weights = []
    transfer_functions = []

    def __init__(self, layer_size, layer_functions=None):
        """Initialize the network"""

        # layer info
        self.layer_count = len(layer_size) - 1
        self.shape = layer_size

        if layer_functions is None:
            for i in range(self.layer_count):
                # if i == self.layer_count - 1:
                #     self.transfer_functions.append(linear)
                # else:
                #     self.transfer_functions.append(sgm)
                self.transfer_functions.append(TransferFunctions.sgm)

        else:
            if len(layer_size)-1 != len(layer_functions):
                raise ValueError("Incompatible list of transfer functions")
            else:
                self.transfer_functions = layer_functions

        # Input/Output data from last run
        self._layerInput = []
        self._layerOutput = []

        # Create weight arrays
        for (p, q) in zip(layer_size[:-1], layer_size[1:]):
            self.weights.append(curandom.rand((q, p + 1), dtype=np.float64))

    # Run method
    def run(self, input_data):
        """Run network based on the input data"""

        input_cases = input_data.shape[0]

        # Clear
        self._layerInput = []
        self._layerOutput = []

        # Run
        for index in range(self.layer_count):
            if index == 0:
                layer_input = concurr.matrix_multiply(self.weights[0],
                    np.vstack((input_data.T, np.ones((1, input_cases)))))
            else:
                layer_input = concurr.matrix_multiply(self.weights[index],
                    np.vstack((self._layerOutput[-1], np.ones((1, input_cases)))))
            self._layerInput.append(layer_input)
            self._layerOutput.append(self.transfer_functions[index](layer_input))

        return self._layerOutput[-1].T

    # TrainEpoch methods
    def train_epoch(self, input_data, target, training_rate=0.01):
        delta = []
        input_cases = input_data.shape[0]
        error = 0

        self.run(input_data)

        # Calculate deltas
        for index in reversed(range(self.layer_count)):
            if index == self.layer_count - 1:
                # Compare to expected result
                output_delta = self._layerOutput[index] - target.T
                error = np.sum(output_delta ** 2)
                delta.append(output_delta * self.transfer_functions[index](self._layerInput[index], True))
            else:
                # Compare to following layer's delta
                delta_pullback = concurr.matrix_multiply_tn(self.weights[index + 1], delta[-1])
                delta.append(delta_pullback[:-1, :] * self.transfer_functions[index](self._layerInput[index], True))

        # Compute weight deltas
        for index in range(self.layer_count):
            delta_index = self.layer_count - 1 - index

            if index == 0:
                layer_output = np.vstack([input_data.T, np.ones([1, input_cases])])
            else:
                layer_output = np.vstack(
                    [self._layerOutput[index - 1], np.ones([1, self._layerOutput[index - 1].shape[1]])])

            weight_delta = np.sum(
                layer_output[None, :, :].transpose(2, 0, 1) * delta[delta_index][None, :, :].transpose(2, 1, 0)
                , axis=0)
            self.weights[index] = concurr.matrix_sum(self.weights[index], -training_rate * weight_delta)

        return error


if __name__ == '__main__':
    pass
