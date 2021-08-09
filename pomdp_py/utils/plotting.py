# Plotting utilties
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
