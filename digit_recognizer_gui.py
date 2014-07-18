#!/usr/bin/python3
import random

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import MNIST

import bmp
import digit_recognizer


class DigitRecognizerGui:
    def change_active(self, idx):
        if idx is not None:
            self.active.imshow(self.images[idx].reshape(28, 28), cmap='gray')
            result = digit_recognizer.recognizer([self.images[idx]])
            for i in sorted(enumerate(result[0]), key=lambda k: k[1], reverse=True):
                print("{} : {:.6f}".format(i[0], i[1]))
            self.figure.canvas.draw()

    def __init__(self, images=[]):
        self.images = images
        self.figure = plt.figure()
        self.miniatures = []
        self.mini_dict = {}

        mini_rows = int(len(self.images) / 20)+1
        mini_cols = min(20, len(self.images))

        for i in range(len(self.images)):
            self.miniatures.append(self.figure.add_subplot(mini_rows+1, mini_cols, i + 1))
            self.miniatures[-1].imshow(self.images[i].reshape(28, 28), cmap='gray')
            self.mini_dict[self.miniatures[-1]] = i
        self.active = self.figure.add_subplot(mini_rows+1, 1, mini_rows+1)
        self.figure.canvas.mpl_connect('button_press_event',
                                       lambda k: self.change_active(self.mini_dict.get(k.inaxes, None)))

    def show(self):
        #self.figure.show()
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*', help="list of files to recognize")
    parser.add_argument('--mnist', help="count of random MNIST images", type=int)
    args = parser.parse_args(sys.argv[1:])
    images = [np.flipud(1 - bmp.load(k, floatize=True)) for k in args.files]
    if args.mnist:
        m_images, m_labels = MNIST.get_data()
        random.shuffle(m_images)
        images.extend(m_images[:args.mnist])
    drg = DigitRecognizerGui(images=images)
    drg.show()

if __name__ == '__main__':
    main()
