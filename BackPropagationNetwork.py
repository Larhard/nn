import pycuda.autoinit

import numpy as np
from pycuda import curandom
from pycuda import gpuarray

import concurr.matrix
import concurr.functions


class TransferFunctions:
    @staticmethod
    def sgm(x, derivative=False):
        if not derivative:
            return concurr.functions.sgm(x)
        else:
            return concurr.functions.sgm_d(x)

    @staticmethod
    def linear(x, derivative=False):
        if not derivative:
            return x
        else:
            return 1.0

    @staticmethod
    def gaussian(x, derivative=False):
        if not derivative:
            return concurr.functions.gaussian(x)
        else:
            return concurr.functions.gaussian_d(x)


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
                layer_input = concurr.matrix.multiply(self.weights[0],
                    concurr.matrix.append_value_line(
                        concurr.matrix.transpose(input_data),
                        1
                    )
                )
            else:
                layer_input = concurr.matrix.multiply(self.weights[index],
                    concurr.matrix.append_value_line(
                        self._layerOutput[-1],
                        1
                    )
                )
            self._layerInput.append(layer_input)
            self._layerOutput.append(self.transfer_functions[index](layer_input))

        return concurr.matrix.transpose(self._layerOutput[-1])

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
                output_delta = concurr.matrix.sum(self._layerOutput[index], concurr.matrix.mul(concurr.matrix.transpose(target), -1))
                error = gpuarray.sum(output_delta * output_delta)
                error = float(error.get())
                delta.append((output_delta * self.transfer_functions[index](self._layerInput[index], True)))
            else:
                # Compare to following layer's delta
                delta_pullback = concurr.matrix.multiply_tn(self.weights[index + 1], delta[-1])
                delta.append((delta_pullback[:-1, :] * self.transfer_functions[index](self._layerInput[index], True)))

        # Compute weight deltas
        for index in range(self.layer_count):
            delta_index = self.layer_count - 1 - index

            if index == 0:
                layer_output = concurr.matrix.append_value_line(
                    concurr.matrix.transpose(input_data),
                    1
                )
            else:
                layer_output = concurr.matrix.append_value_line(
                    self._layerOutput[index - 1],
                    1
                )

            weight_delta = concurr.matrix.cart_mul_sum(layer_output, delta[delta_index])
            self.weights[index] = concurr.matrix.sum(self.weights[index], concurr.matrix.mul(weight_delta, -training_rate))

        return error


if __name__ == '__main__':
    pass
