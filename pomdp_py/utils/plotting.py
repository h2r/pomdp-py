"""Plotting utilties"""
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
