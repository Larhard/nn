import struct
from array import array
import numpy as np


def main_bytes(args):
    result = 0
    for i in args:
        result += i
    return result/len(args)


def concat_bytes(args):
    result = 0
    for i in args:
        result += i
        result <<= 8
    return result


def load(path, grayscale=True, floatize=False):
    with open(path, 'rb') as fd:
        magic_id, file_size, reserved_1, reserved_2, offset = struct.unpack('<HIHHI', fd.read(14))
        h_size, width, height, planes, bits_per_px, compression, image_size, x_px_per_meter, y_px_per_meter,\
        colors_used, important_colors, color_rotation, bitmap_reversed = struct.unpack('<IIIHHIIIIIBBH', fd.read(40))
        fd.read(offset - 54)
        bitmap_array = array('B', fd.read())
    bytes_per_px = int(bits_per_px/8)
    merge_func = main_bytes if grayscale else concat_bytes
    bitmap = np.array([merge_func(bitmap_array[bytes_per_px*i:bytes_per_px*(i+1)]) for i in range(width*height)])
    bitmap = bitmap.reshape((height, width))
    if floatize:
        bitmap /= 255
    return bitmap
