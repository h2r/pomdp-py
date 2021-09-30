"""
Utilties for typography, i.e. dealing with
strings for the purpose of displaying them.
"""

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

def info(content):
    return bcolors.s(bcolors.BLUE, content)

def note(content):
    return bcolors.s(bcolors.YELLOW, content)

def error(content):
    return bcolors.s(bcolors.GREEN, content)

def warning(content):
    return bcolors.s(bcolors.YELLOW, content)

def success(content):
    return bcolors.s(bcolors.GREEN, content)

def bold(content):
    return bcolors.s(bcolors.BOLD, content)

def white(content):
    return bcolors.s(bcolors.WHITE, content)

def green(content):
    return bcolors.s(bcolors.GREEN, content)

def cyan(content):
    return bcolors.s(bcolors.CYAN, content)

def magenta(content):
    return bcolors.s(bcolors.MAGENTA, content)

def blue(content):
    return bcolors.s(bcolors.BLUE, content)

def green(content):
    return bcolors.s(bcolors.GREEN, content)

def yellow(content):
    return bcolors.s(bcolors.YELLOW, content)

def red(content):
    return bcolors.s(bcolors.RED, content)

def white(content):
    return bcolors.s(bcolors.WHITE, content)

colors = {
    "white",
    "green",
    "cyan",
    "magenta",
    "blue",
    "green",
    "yellow",
    "red",
    "white",
}
