"""Plot the light dark environment"""
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import pomdp_problems.light_dark as ld
from pomdp_py.utils import plotting, colors
from pomdp_py.utils.misc import remap

class LightDarkViz:
    """This class deals with visualizing a light dark domain"""

    def __init__(self, env, x_range, y_range, res):
        """
        Args:
            env (LightDarkEnvironment): Environment for light dark domain.
            x_range (tuple): a tuple of floats (x_min, x_max).
            y_range (tuple): a tuple of floats (y_min, y_max).
            res (float): specifies the size of each rectangular strip to draw;
                As in the paper, the light is at a location on the x axis.
        """
        self._env = env
        self._res = res
        self._x_range = x_range
        self._y_range = y_range
        fig = plt.gcf()
        self._ax = fig.add_subplot(1,1,1)
        self._goal_pos = None
        self._m_0 = None  # initial belief pose

        # For tracking the path; list of robot position tuples
        self._log_paths = {}

    def log_position(self, position, path=0):
        if path not in self._log_paths:
            self._log_paths[path] = []
        self._log_paths[path].append(position)

    def set_goal(self, goal_pos):
        self._goal_pos = goal_pos

    def set_initial_belief_pos(self, m_0):
        self._m_0 = m_0

    def plot(self,
             path_colors={0: [(0,0,0), (0,0,254)]},
             path_styles={0: "--"},
             path_widths={0: 1}):
        self._plot_gradient()
        self._plot_path(path_colors, path_styles, path_widths)
        self._plot_robot()
        self._plot_goal()
        self._plot_initial_belief_pos()

    def _plot_robot(self):
        cur_pos = self._env.state.position
        plotting.plot_circle(self._ax, cur_pos,
                         0.25, # tentative
                         color="black", fill=False,
                         linewidth=1, edgecolor="black",
                         zorder=3)

    def _plot_initial_belief_pos(self):
        if self._m_0 is not None:
            plotting.plot_circle(self._ax, self._m_0,
                                 0.25, # tentative
                                 color="black", fill=False,
                                 linewidth=1, edgecolor="black",
                                 zorder=3)

    def _plot_goal(self):
        if self._goal_pos is not None:
            plotting.plot_circle(self._ax,
                                 self._goal_pos,
                                 0.25,  # tentative
                                 linewidth=1, edgecolor="blue",
                                 zorder=3)

    def _plot_path(self, colors, styles, linewidths):
        """Plot robot path"""
        # Plot line segments
        for path in self._log_paths:
            if path not in colors:
                path_color = [(0,0,0)] * len(self._log_paths[path])
            else:
                if len(colors[path]) == 2:
                    c1, c2 = colors[path]
                    path_color = colors.linear_color_gradient(c1, c2,
                                                              len(self._log_paths[path]),
                                                              normalize=True)
                else:
                    path_color = [colors[path]] * len(self._log_paths[path])

            if path not in styles:
                path_style = "--"
            else:
                path_style = styles[path]

            if path not in linewidths:
                path_width = 1
            else:
                path_width = linewidths[path]

            for i in range(1, len(self._log_paths[path])):
                p1 = self._log_paths[path][i-1]
                p2 = self._log_paths[path][i]
                try:
                    plotting.plot_line(self._ax, p1, p2, color=path_color[i],
                                   linestyle=path_style, zorder=2, linewidth=path_width)
                except Exception:
                    import pdb; pdb.set_trace()

    def _plot_gradient(self):
        """display the light dark domain."""
        xmin, xmax = self._x_range
        ymin, ymax = self._y_range
        # Note that higher brightness has lower brightness value
        hi_brightness = self._env.const
        lo_brightness = max(0.5 * (self._env.light - xmin)**2 + self._env.const,
                            0.5 * (self._env.light - xmax)**2 + self._env.const)
        # Plot a bunch of rectangular strips along the x axis
        # Check out: https://stackoverflow.com/questions/10550477
        x = xmin
        verts = []
        colors = []
        while x < xmax:
            x_next = x + self._res
            verts.append([(x, ymin), (x_next, ymin), (x_next, ymax), (x, ymax)])
            # compute brightness based on equation in the paper
            brightness = 0.5 * (self._env.light - x)**2 + self._env.const
            # map brightness to a grayscale color
            grayscale = int(round(remap(brightness, hi_brightness, lo_brightness, 255, 0)))
            grayscale_hex = colors.rgb_to_hex((grayscale, grayscale, grayscale))
            colors.append(grayscale_hex)
            x = x_next
        plotting.plot_polygons(verts, colors, ax=self._ax)
        self._ax.set_xlim(xmin, xmax)
        self._ax.set_ylim(ymin, ymax)

if __name__ == "__main__":
    env = ld.LightDarkEnvironment(ld.State((0.5, 2.5)),  # init state
                                  (1.5, -1),  # goal pose
                                  5,  # light
                                  1)  # const
    viz = LightDarkViz(env, (-1, 7), (-2, 4), 0.1)
    viz.log_position((5,2))
    viz.log_position((5,0))
    viz.log_position((4,-1))
    viz.plot()

    plt.show()
