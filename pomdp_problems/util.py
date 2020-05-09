import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import matplotlib.lines as lines
import math

# Convenient color utilities
def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % (rgb[0], rgb[1], rgb[2])

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

def linear_color_gradient(rgb_start, rgb_end, n, normalize=False):
    colors = [rgb_start]
    for t in range(1, n):
        color = tuple(
            rgb_start[i] + float(t)/(n-1)*(rgb_end[i] - rgb_start[i])
            for i in range(3)
        )
        if normalize:
            color = tuple(color[i] / 255.0 for i in range(3))
        colors.append(color)
    return colors  

def rgb_to_grayscale(rgb):
    r,g,b = rgb
    return (0.2989*r, 0.5870*g, 0.1140*b)

# colors
def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (255, 255, 255)'''
    color = np.array(color)
    white = np.array([255, 255, 255])
    vector = white-color
    return color + vector * percent    


# Plot polygons with colors
def plot_polygons(verts, colors, ax=None, edgecolor=None):
    """
    `verts` is a sequence of ( verts0, verts1, ...) where verts_i is a sequence of
    xy tuples of vertices, or an equivalent numpy array of shape (nv, 2).
   
    `c` is a sequence of (color0, color1, ...) where color_i is a color,
    represented by a hex string (7 characters #xxxxxx).
   
    Creates a PolygonCollection object in the axis `ax`."""
    if ax is None:
        fig = plt.gcf()
        ax = fig.add_subplot(1,1,1)
    pc = PolyCollection(verts)
    pc.set_edgecolor(edgecolor)    
    pc.set_facecolor(colors)
    ax.add_collection(pc)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    
def plot_line(ax, p1, p2,
              linewidth=1, color='black', zorder=0, alpha=1.0, linestyle="-"):
    p1x, p1y = p1
    p2x, p2y = p2
    line = lines.Line2D([p1x, p2x], [p1y, p2y],
                        linewidth=linewidth, color=color, zorder=zorder,
                        alpha=alpha, linestyle=linestyle)
    ax.add_line(line)
    
def plot_circle(ax, center, radius, color="blue",
                fill=False, zorder=0, linewidth=0,
                edgecolor=None, label_text=None,
                alpha=1.0, text_color="white"):
    px, py = center
    circ = plt.Circle((px, py), radius, facecolor=color, fill=fill,
                      zorder=zorder, linewidth=linewidth, edgecolor=edgecolor, alpha=alpha)
    ax.add_artist(circ)
    if label_text:
        text = ax.text(px, py, label_text, color=text_color,
                        ha='center', va='center', size=7, weight='bold')
        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                               path_effects.Normal()])

    
# functional utilitiesx
def remap(oldval, oldmin, oldmax, newmin, newmax):
    return (((oldval - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin   

# Utility functions
def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

def to_rad(deg):
    return deg * math.pi / 180.0

def in_range(val, rang):
    # Returns True if val is in range (a,b); Inclusive.
    return val >= rang[0] and val <= rang[1]
