"""Utilities for dealing with colors"""

import numpy as np
import random

# colors
def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (255, 255, 255)'''
    color = np.array(color)
    white = np.array([255, 255, 255])
    vector = white-color
    return color + vector * percent

def rgb_to_hex(rgb):
    r,g,b = rgb
    return '#%02x%02x%02x' % (int(r), int(g), int(b))

def hex_to_rgb(hx):
    """hx is a string, begins with #. ASSUME len(hx)=7."""
    if len(hx) != 7:
        raise ValueError("Hex must be #------")
    hx = hx[1:]  # omit the '#'
    r = int('0x'+hx[:2], 16)
    g = int('0x'+hx[2:4], 16)
    b = int('0x'+hx[4:6], 16)
    return (r,g,b)

def inverse_color_rgb(rgb):
    r,g,b = rgb
    return (255-r, 255-g, 255-b)

def inverse_color_hex(hx):
    """hx is a string, begins with #. ASSUME len(hx)=7."""
    return inverse_color_rgb(hex_to_rgb(hx))

def random_unique_color(colors, ctype=1):
    """
    ctype=1: completely random
    ctype=2: red random
    ctype=3: blue random
    ctype=4: green random
    ctype=5: yellow random
    """
    if ctype == 1:
        color = "#%06x" % random.randint(0x444444, 0x999999)
        while color in colors:
            color = "#%06x" % random.randint(0x444444, 0x999999)
    elif ctype == 2:
        color = "#%02x0000" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#%02x0000" % random.randint(0xAA, 0xFF)
    elif ctype == 4:  # green
        color = "#00%02x00" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#00%02x00" % random.randint(0xAA, 0xFF)
    elif ctype == 3:  # blue
        color = "#0000%02x" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#0000%02x" % random.randint(0xAA, 0xFF)
    elif ctype == 5:  # yellow
        h = random.randint(0xAA, 0xFF)
        color = "#%02x%02x00" % (h, h)
        while color in colors:
            h = random.randint(0xAA, 0xFF)
            color = "#%02x%02x00" % (h, h)
    else:
        raise ValueError("Unrecognized color type %s" % (str(ctype)))
    return color
