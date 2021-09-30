"""Misc Python utilities"""
from difflib import SequenceMatcher

def remap(oldvalue, oldmin, oldmax, newmin, newmax):
    if oldmax - oldmin == 0:
        print("Warning in remap: the old range has size 0")
        oldmax = oldmin + oldvalue
    return (((oldvalue - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin


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


def safe_slice(arr, start, end):
    true_start = max(0, min(len(arr)-1, start))
    true_end = max(0, min(len(arr)-1, end))
    return arr[true_start:true_end]

def similar(a, b):
    # Reference: https://stackoverflow.com/questions/17388213/find-the-similarity-metric-between-two-strings
    return SequenceMatcher(None, a, b).ratio()


class special_char:
    left = '\u2190'
    up = '\u2191'
    right = '\u2192'
    down = '\u2193'
    longleft = '\u27F5'
    longright = '\u27F6'

    hline = "─"
    vline = "│"

    bottomleft = "└"
    longbottomleft = "└─"
    topleft = "┌"
    longtopleft = "┌─"
    topright = "┐"
    longtopright = "─┐"
    bottomright = "┘"
    longbottomright = "─┘"

    intersect = "┼"
    topt = "┬"
    leftt = "├"
    rightt = "┤"
    bottomt = "┴"
    shadebar = "▒"

    SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
