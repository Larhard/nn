#!/usr/bin/python2

#try:
#except Exception as e:
#    print e

import os
import sys
import re
import datetime

import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()

import matplotlib
matplotlib.use('Agg')

import matplotlib.dates as md
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

DIR = '/home/larhard/nn/'

def get_graph(show=True):
    figure = plt.figure()
    data = [(os.stat(os.path.join(DIR, backup)).st_mtime, float(re.match('^backup_(\d+(\.\d+)?)\.pkl$', backup).group(1))) for backup in os.listdir(DIR) if re.match('^backup_(\d+(\.\d+)?)\.pkl$', backup)]
    data.sort(reverse=True)
    #print(data)
    #print(list(zip(*data)))
    x, y = zip(*data)
    x = [datetime.datetime.fromtimestamp(k) for k in x]
    plt.plot(x, y, color='blue')
    plt.plot(x, [k**.5 for k in y], color='green')

    figure.subplots_adjust(bottom=0.3)

    ax = plt.gca()
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=65, horizontalalignment='right')

    if show:
        plt.show()
    else:
        canvas = FigureCanvas(figure)
        buf = io.BytesIO()
        canvas.print_png(buf)
        return buf.getvalue()
        #sys.stdout.buffer.write(buf.getvalue())

if __name__ == '__main__':
    get_graph(show=True)
