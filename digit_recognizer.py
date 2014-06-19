#!/usr/bin/python3
from mimetypes import init

import MNIST
import pickle
import BackPropagationNetwork as Bpn
import numpy as np
import functools
import random as rnd
from operator import mul
import sys
import bmp


def teacher(clear=False, config_file='digit_recognizer.pkl', iterations=1000000000):
    config = {}

    if not clear:
        try:
            with open(config_file, 'rb') as fd:
                config = pickle.load(fd)
                print("config file loaded")
        except FileNotFoundError:
            pass

    config['layers'] = config.get('layers', [28*28, 1000, 1000, 10])

    network = Bpn.NeuralNetwork(config['layers'])

    if 'weights' in config:
        network.weights = config['weights']
        print("weights restored")

    images, labels = MNIST.get_data()
    data_size = len(labels)

    def generate_training_data(size=100):
        t_set = [rnd.randint(0, data_size-1) for k in range(size)]
        t_images = np.array([
            images[k].reshape((functools.reduce(mul, images[k].shape, 1))) for k in t_set
        ], dtype=np.float)

        for i in range(size):
            for j in range(t_images[i].shape[0]):
                t_images[i][j] /= 256

        t_labels = np.zeros((size, 10))
        for i, k in enumerate(t_set):
            t_labels[i][labels[k]] = 1
        return t_images, t_labels

    train_images = None
    train_labels = None
    try:
        for i in range(iterations):
            if i % 400 == 0:
                train_images, train_labels = generate_training_data()
                print("New training data loaded")
            error = network.train_epoch(train_images, train_labels)
            if i % 1 == 0:
                print("Iteration: {:10} Error: {:10.6}".format(i, error))
    except KeyboardInterrupt:
        pass

    config['weights'] = network.weights
    with open(config_file, 'wb') as fd:
        pickle.dump(config, fd)
        print("config file saved")


def recognizer(paths, config_file='digit_recognizer.pkl'):
    with open(config_file, 'rb') as fd:
        config = pickle.load(fd)
        print("config file loaded")

    network = Bpn.NeuralNetwork(config['layers'])

    if 'weights' in config:
        network.weights = config['weights']
        print("weights restored")

    for path in paths:
        print("Path: {}".format(path))
        image = bmp.load(path)
        image = image.reshape((functools.reduce(mul, image.shape)))
        for i in range(len(image)):
            image[i] = (255 - image[i]) / 255
        # print(image)

        result = network.run(image[None, :])
        for i in range(len(result[0])):
            print("{} : {:.6f}  ".format(i, result[0][i]), end='')
        print()


if __name__ == '__main__':
    clear = '-clear' in sys.argv
    if '-train' in sys.argv:
        teacher(clear=clear)
    else:
        # recognizer(sys.argv[1:])
        recognizer(['test_image.bmp'])
