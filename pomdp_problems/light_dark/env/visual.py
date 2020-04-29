"""Plot the light dark environment"""
import pomdp_problems.util as util
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import pomdp_problems.light_dark as ld

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

        # For tracking the path; list of robot position tuples
        self._log_path = []

    def log_position(self, position):
        self._log_path.append(position)

    def plot(self):
        self._plot_gradient()
        # self._plot_path()
        self._plot_robot()
        self._plot_goal()

    def _plot_robot(self):
        cur_pos = self._env.state.position
        util.plot_circle(self._ax, cur_pos,
                         0.25, # tentative
                         color="black", fill=True,
                         linewidth=2, edgecolor="black",
                         zorder=3)

    def _plot_goal(self):
        util.plot_circle(self._ax,
                         self._env.goal_pos,
                         0.5,  # tentative
                         linewidth=1, edgecolor="black",
                         zorder=3)
        
    def _plot_path(self):
        """Plot robot path"""
        # Plot line segments
        print("GG")
        for i in range(1, len(self._log_path)):
            print("kk")
            p1 = self._log_path[i-1]
            p2 = self._log_path[i]
            util.plot_line(self._ax, p1, p2, color="black", linestyle="--", zorder=2)

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
            grayscale = int(round(util.remap(brightness, hi_brightness, lo_brightness, 255, 0)))
            grayscale_hex = util.rgb_to_hex((grayscale, grayscale, grayscale))
            colors.append(grayscale_hex)
            x = x_next
        util.plot_polygons(verts, colors, ax=self._ax)
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
