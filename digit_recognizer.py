#!/usr/bin/python3

TRAINING_RATE = 0.001
TRAINING_DATA_SIZE = 100
TRAINING_RESET = 100
HIDDEN_LAYERS = []
SAVE_FREQUENCY = 10000
from mimetypes import init

import MNIST
import pickle
import BackPropagationNetwork as Bpn
import numpy as np
import functools
import random as rnd
from operator import mul
import itertools as it
import sys
import bmp


def teacher(clear=False, config_file='digit_recognizer.pkl', iterations=1000000000, save=True, ignore_interrupt=True):
    config = {}

    if not clear:
        try:
            with open(config_file, 'rb') as fd:
                config = pickle.load(fd)
                print("config file loaded")
        except FileNotFoundError:
            pass

    config['layers'] = config.get('layers', list(it.chain((28*28, ), HIDDEN_LAYERS, (10, ))))
    print("layers : {}".format(config['layers']))

    network = Bpn.NeuralNetwork(config['layers'])

    if 'weights' in config:
        network.weights = config['weights']
        print("weights restored")

    images, labels = MNIST.get_data()
    data_size = len(labels)

    def generate_training_data(size=TRAINING_DATA_SIZE):
        t_set = [rnd.randint(0, data_size-1) for k in range(size)]
        t_images = np.array([
            images[k].reshape((functools.reduce(mul, images[k].shape, 1))) for k in t_set
        ], dtype=np.float)

        for i in range(size):
            t_images /= 255

        t_labels = np.zeros((size, 10))
        for i, k in enumerate(t_set):
            t_labels[i][labels[k]] = 1
        return t_images, t_labels

    train_images = None
    train_labels = None
    avg_error = None
    tested_labels = [0]*10

    def save_config(cfg):
        config['weights'] = network.weights
        with open(cfg, 'wb') as fd:
            pickle.dump(config, fd)
            print("config file saved ({})".format(cfg))
    try:
        for i in range(iterations):
            if i % TRAINING_RESET == 0:
                train_images, train_labels = generate_training_data()
                for label in train_labels:
                    for idx, k in enumerate(label):
                        if k:
                            tested_labels[idx] += 1
                print("New training data loaded: {}".format(tested_labels))
            error = network.train_epoch(train_images, train_labels, training_rate=TRAINING_RATE)
            if avg_error is None:
                avg_error = error / TRAINING_DATA_SIZE
            else:
                avg_error = (avg_error * (SAVE_FREQUENCY - 1) + error / TRAINING_DATA_SIZE) / SAVE_FREQUENCY
            if i % 1 == 0:
                print("Iteration: {:10} Error: {:10.6f} Average: {:10.10f}".format(i, error, avg_error))
            if i % SAVE_FREQUENCY == 0:
                save_config("backup_{}.pkl".format(avg_error))
    except KeyboardInterrupt as e:
        if not ignore_interrupt:
            raise e

    if save:
        save_config(config_file)


def recognizer(paths, config_file='digit_recognizer.pkl'):
    with open(config_file, 'rb') as fd:
        config = pickle.load(fd)
        print("config file loaded")
        print("shape : {}".format(config['layers']))

    network = Bpn.NeuralNetwork(config['layers'])

    if 'weights' in config:
        network.weights = config['weights']
        print("weights restored")

    for path in paths:
        if isinstance(path, (np.ndarray)):
            image = path
        else:
            print("Path: {}".format(path))
            image = bmp.load(path)
            image = np.flipud(1 - image / 255)
        image = image.reshape((functools.reduce(mul, image.shape)))
        # print(image)

        return network.run(image[None, :])


if __name__ == '__main__':
    clear = '-clear' in sys.argv
    if '-train' in sys.argv:
        teacher(clear=clear)
    else:
        # recognizer(sys.argv[1:])
        result = recognizer(['test_image.bmp'])
        for i in sorted(enumerate(result[0]), key=lambda k: k[1], reverse=True):
            print("{} : {:.6f}".format(i[0], i[1]))
        print()
