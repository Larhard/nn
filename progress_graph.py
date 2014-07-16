#!/usr/bin/python3

import os
import sys
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

DIR = '/home/larhard/NeuralNetwork/'

def get_graph(show=True):
    figure = plt.figure()
    data = [(os.stat(backup).st_mtime, float(re.match('^backup_(\d+(\.\d+)?)\.pkl$', backup).group(1))) for backup in os.listdir(DIR) if re.match('^backup_(\d+(\.\d+)?)\.pkl$', backup)]
    data.sort(reverse=True)
    #print(data)
    #print(list(zip(*data)))
    plt.plot(*zip(*data))
    if show:
        plt.show()
    else:
        canvas = FigureCanvas(figure)
        buf = io.BytesIO()
        canvas.print_png(buf)
        sys.stdout.buffer.write(buf.getvalue())

if __name__ == '__main__':
    get_graph()
