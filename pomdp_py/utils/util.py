# Assorted convenient functions
#
# Kaiyu Zheng
import numpy as np
import random

def remap(oldvalue, oldmin, oldmax, newmin, newmax):
    if oldmax - oldmin == 0:
        print("Warning in remap: the old range has size 0")
        oldmax = oldmin + oldvalue
    return (((oldvalue - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin

# Plotting
import matplotlib.pyplot as plt

def plot_points(xvals, yvals, color=None,
                size=1.5, label=None, connected=True, style="--", linewidth=1.5,
                xlabel='x', ylabel='f(x)', loc="lower right"):
    if not connected:
        plt.scatter(xvals, yvals, s=size, c=color, label=label)
    else:
        plt.plot(xvals, yvals, style, linewidth=linewidth, label=label)
    # plt.axhline(y=0, color='k')
    # plt.axvline(x=0, color='k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=loc)

def save_plot(path, bbox_inches='tight'):
    plt.savefig(path, bbox_inches=bbox_inches)
    plt.close()


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


# Printing
def json_safe(obj):
    if isinstance(obj, bool):
        return str(obj).lower()
    elif isinstance(obj, (list, tuple)):
        return [json_safe(item) for item in obj]
    elif isinstance(obj, dict):
        return {json_safe(key):json_safe(value) for key, value in obj.items()}
    else:
        return str(obj)
    return obj


# Math
def vec(p1, p2):
    """ vector from p1 to p2 """
    if type(p1) != np.ndarray:
        p1 = np.array(p1)
    if type(p2) != np.ndarray:
        p2 = np.array(p2)
    return p2 - p1

def proj(vec1, vec2, scalar=False):
    # Project vec1 onto vec2. Returns a vector in the direction of vec2.
    scale = np.dot(vec1, vec2) / np.linalg.norm(vec2)
    if scalar:
        return scale
    else:
        return vec2 * scale

def R_x(th):
    return np.array([
        1, 0, 0, 0,
        0, np.cos(th), -np.sin(th), 0,
        0, np.sin(th), np.cos(th), 0,
        0, 0, 0, 1
    ]).reshape(4,4)

def R_y(th):
    return np.array([
        np.cos(th), 0, np.sin(th), 0,
        0, 1, 0, 0,
        -np.sin(th), 0, np.cos(th), 0,
        0, 0, 0, 1
    ]).reshape(4,4)

def R_z(th):
    return np.array([
        np.cos(th), -np.sin(th), 0, 0,
        np.sin(th), np.cos(th), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]).reshape(4,4)

def T(dx, dy, dz):
    return np.array([
        1, 0, 0, dx,
        0, 1, 0, dy,
        0, 0, 1, dz,
        0, 0, 0, 1
    ]).reshape(4,4)

def to_radians(th):
    return th*np.pi / 180

def R_between(v1, v2):
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Only applicable to 3D vectors!")
    v = np.cross(v1, v2)
    c = np.dot(v1, v2)
    s = np.linalg.norm(v)
    I = np.identity(3)

    vX = np.array([
        0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0
    ]).reshape(3,3)
    R = I + vX + np.matmul(vX,vX) * ((1-c)/(s**2))
    return R

def approx_equal(v1, v2, epsilon=1e-6):
    if len(v1) != len(v2):
        return False
    for i in range(len(v1)):
        if abs(v1[i] - v2[i]) > epsilon:
            return False
    return True


# Others
def safe_slice(arr, start, end):
    true_start = max(0, min(len(arr)-1, start))
    true_end = max(0, min(len(arr)-1, end))
    return arr[true_start:true_end]


# Colors on terminal https://stackoverflow.com/a/287944/2893053
class bcolors:
    WHITE = '\033[97m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

    @staticmethod
    def disable():
        bcolors.WHITE   = ''
        bcolors.CYAN    = ''
        bcolors.MAGENTA = ''
        bcolors.BLUE    = ''
        bcolors.GREEN   = ''
        bcolors.YELLOW  = ''
        bcolors.RED     = ''
        bcolors.ENDC    = ''
        
    @staticmethod
    def s(color, content):
        """Returns a string with color when shown on terminal.
        `color` is a constant in `bcolors` class."""
        return color + content + bcolors.ENDC

def print_info(content):
    print(bcolors.s(bcolors.BLUE, content))

def print_note(content):
    print(bcolors.s(bcolors.YELLOW, content))

def print_error(content):
    print(bcolors.s(bcolors.RED, content))

def print_warning(content):
    print(bcolors.s(bcolors.YELLOW, content))

def print_success(content):
    print(bcolors.s(bcolors.GREEN, content))

def print_info_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.BLUE, content))

def print_note_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.GREEN, content))    

def print_error_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.RED, content))

def print_warning_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.YELLOW, content))

def print_success_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.GREEN, content))    
# For your convenience:
# from pomdp_py.util import print_info, print_error, print_warning, print_success, print_info_bold, print_error_bold, print_warning_bold, print_success_bold

