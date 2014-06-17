from BackPropagationNetwork import NeuralNetwork
from BackPropagationNetwork import TransferFunctions as Tf
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import random as rnd
import json


def test003():
    # bpn = NeuralNetwork((2, 10, 1), layer_functions=(Tf.sgm, Tf.sgm))
    bpn = NeuralNetwork((2, 50, 1), layer_functions=(Tf.sgm, Tf.sgm))
    # bpn = NeuralNetwork((2, 10, 1), layer_functions=(TF.tanh, TF.linear))
    # bpn = NeuralNetwork((2, 20, 1))

    # try:
    # with open('test003_weights.json', 'r') as fd:
    #         bpn.weights = [np.array(k) for k in json.load(fd)]
    # except FileNotFoundError:
    #     pass

    # train_input = np.array(
    #     [[0, 0], [1, 1], [2, 2], [3, 3], [0, 1], [1, 2], [2, 3], [3, 1], [0, 2], [1, 3], [2, 1], [3, 2], ])
    # train_target = np.array(
    #     [[0.95], [0.95], [0.95], [0.95], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], [0.05], ])

    train_input = np.array(
        [[0, 0], [1, 1], [0, 1], [1, 0], [0.3, 0.3], [0.8, 0.8], [0.3, 0.8], [0.8, 0.3], [0.0, 0.3], [0.1, 0.13],
         [0.5, 0.55], ])
    train_target = np.array([[1], [1], [0], [0], [1], [1], [0], [0], [0], [0], [0], ])

    # condition = lambda k: 0.95 if 4 * abs(k[0])+1.2 - abs(k[1]-0.5)*10 > 0.7 else 0.05
    # condition = lambda k: 0.95 if 4 * abs(k[0]-0.5)+1.2 - abs(k[1]-0.5)*4 > 0.7 else 0.05
    # condition = lambda k: 0.05 if abs(k[0]-np.sin(k[1]*11))*2 - abs(k[1]-np.sin(k[1]*17))*2 > 0.1 else 0.95
    # condition = lambda k: 0.05 if abs(k[0]-k[1]) < 0.1 else 0.95
    # condition = lambda k: 0.05 if k[0] > k[1] * 0.5+np.sin(k[1] * 20)/8 + 0.25 else 0.95
    # condition = lambda k: 0.05 if 0.03 < abs(k[0]-0.5)**2 + abs(k[1]-0.5)**2 < 0.17 else 0.95
    condition = lambda k: 0.95 if abs(k[0] - 0.3) ** 2 + abs(k[1] - 0.75) ** 2 < 0.04 or \
                                  abs(k[0] - 0.3) ** 2 + abs(k[1] - 0.25) ** 2 < 0.04 or \
                                  abs(k[0] - 0.65) ** 2 + abs(k[1] - 0.5) ** 2 < 0.07 else 0.05
    # condition = lambda k: 0.95 if abs(k[0]-0.5)**2 + abs(k[1]-0.5)**2 < 0.1 else 0.05

    def generate_sample_data():
        # sample_input = np.array([[rnd.randint(0, 100)/100, rnd.randint(0, 100)/100] for _ in range(100)])
        sample_input = np.array([[rnd.randint(0, 100) / 100, rnd.randint(0, 100) / 100] for _ in range(300)])
        sample_target = np.array([[condition(k)] for k in sample_input])
        return sample_input, sample_target

    train_input, train_target = generate_sample_data()
    train_iterations = 4000000
    train_error = 1e-5

    test_input = np.array([[p / 100.0, q / 100.0] for (p, q) in it.product(range(100), repeat=2)])

    fig = plt.figure()
    plot = fig.add_subplot(121)
    plt.ion()
    test_output = np.array([[condition(k)] for k in test_input])
    plot.imshow(test_output.reshape(100, 100))

    # image.set_cmap('gray')
    plot = fig.add_subplot(122)
    image = plot.imshow(test_output.reshape(100, 100), animated=True)

    plt.draw()
    plt.pause(0.01)
    # plt.show()

    for i in range(train_iterations):
        err = bpn.train_epoch(train_input, train_target, training_rate=0.02)
        if i % 2500 == 0:
            print("Iteration: {}\tError: {:0.6f}".format(i, err))

            test_input = np.array([[p / 100.0, q / 100.0] for (p, q) in it.product(range(100), repeat=2)])
            test_output = bpn.run(test_input)
            for k in train_input:
                test_output[(k[0] * 100 - 1) * 100 + k[1] * 100 - 1] = 0.5
            image.set_data(test_output.reshape(100, 100))
            plt.draw()
            plt.pause(0.01)

        if err <= train_error:
            print("Minimum error reached at iteration {}".format(i))
            break

        if i % 10000 == 0:
            train_input, train_target = generate_sample_data()

    with open('test003_weights.json', 'w') as fd:
        json.dump([k.tolist() for k in bpn.weights], fd)


if __name__ == '__main__':
    test003()
