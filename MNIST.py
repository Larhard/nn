import os
import struct
from array import array
import numpy as np


DIGITS_ALL = set(range(10))


def get_data(digits=DIGITS_ALL, dataset='training', directory='./MNIST_data'):
    """
    Get MNIST dataset
    @param dataset: 'training' or 'test'
    @return: [images], [labels]
    """

    if dataset == 'training':
        lbl_file = os.path.join(directory, 'train-labels-idx1-ubyte')
        img_file = os.path.join(directory, 'train-images-idx3-ubyte')
    else:
        lbl_file = os.path.join(directory, 't10k-labels-idx1-ubyte')
        img_file = os.path.join(directory, 't10k-images-idx3-ubyte')

    with open(lbl_file, 'rb') as fd:
        lbl_magic_number, lbl_count = struct.unpack('>II', fd.read(8))
        labels = array('B', fd.read())
    with open(img_file, 'rb') as fd:
        img_magic_number, img_count, img_cols, img_rows = struct.unpack('>IIII', fd.read(16))
        images = array('B', fd.read())

    assert lbl_count == img_count

    images = [
        np.array(images[img_cols*img_rows*i:img_cols*img_rows*(i+1)]).reshape((img_rows, img_cols))
        for i in range(img_count) if labels[i] in digits
    ]
    labels = [k for k in labels if k in digits]

    return images, labels
