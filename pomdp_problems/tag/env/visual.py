"""Largely based on MosViz, except this is not an OO-POMDP"""
import pygame
import cv2
import math
import numpy as np
import random
import pomdp_py.utils as util
from pomdp_problems.tag.env.env import *
from pomdp_problems.tag.domain.observation import *
from pomdp_problems.tag.domain.action import *
from pomdp_problems.tag.domain.state import *
from pomdp_problems.tag.example_worlds import *
from pomdp_problems.tag.models.observation_model import *

#### Visualization through pygame ####
class TagViz:

    def __init__(self, env, res=30, fps=30, controllable=False, observation_model=None):
        self._env = env

        self._res = res
        self._img = self._make_gridworld_image(res)
        self._last_observation = None
        self._last_action = None
        self._last_belief = None
        self._observation_model = observation_model

        self._controllable = controllable
        self._running = False
        self._fps = fps
        self._playtime = 0.0

        self._target_color = (200, 0, 50)

    def _make_gridworld_image(self, r):
        # Preparing 2d array
        w, l = self._env.width, self._env.length
        arr2d = np.full((self._env.width,
                         self._env.length), 0)  # free grids
        # Creating image
        img = np.full((w*r,l*r,3), 255, dtype=np.int32)
        for x in range(w):
            for y in range(l):
                if (x,y) not in self._env.grid_map.obstacle_poses:
                    arr2d[x,y] == 0    # free
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  (255, 255, 255), -1)
                else:
                    arr2d[x,y] == 1  # obstacle
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  (40, 31, 3), -1)
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              (0, 0, 0), 1, 8)
        return img

    @property
    def img_width(self):
        return self._img.shape[0]

    @property
    def img_height(self):
        return self._img.shape[1]

    @property
    def last_observation(self):
        return self._last_observation

    def update(self, action, observation, belief):
        """
        Update the visualization after there is new real action and observation
        and updated belief.
        """
        self._last_action = action
        self._last_observation = observation
        self._last_belief = belief

    @staticmethod
    def draw_robot(img, x, y, th, size, color=(255,12,12)):
        radius = int(round(size / 2))
        cv2.circle(img, (y+radius, x+radius), radius, color, thickness=6)
        # endpoint = (y+radius + int(round(radius*math.sin(th))),
        #             x+radius + int(round(radius*math.cos(th))))
        # cv2.line(img, (y+radius,x+radius), endpoint, color, 2)

    @staticmethod
    def draw_observation(img, z, rx, ry, rth, r, size, color=(12,12,255)):
        assert type(z) == TagObservation, "%s != TagObservation" % (str(type(z)))
        radius = int(round(r / 2))
        if z.target_position is not None:
            lx, ly = z.target_position
            cv2.circle(img, (ly*r+radius,
                             lx*r+radius), size, color, thickness=-1)


    # TODO! Deprecated.
    @staticmethod
    def draw_belief(img, belief, r, size, target_color):
        """belief (OOBelief)"""
        radius = int(round(r / 2))

        circle_drawn = {}  # map from pose to number of times drawn

        hist = belief.get_histogram()
        color = target_color

        last_val = -1
        count = 0
        for state in reversed(sorted(hist, key=hist.get)):
            if last_val != -1:
                color = util.lighter(color, 1-hist[state]/last_val)
            if np.mean(np.array(color) / np.array([255, 255, 255])) < 0.999:
                tx, ty = state.target_position
                if (tx,ty) not in circle_drawn:
                    circle_drawn[(tx,ty)] = 0
                circle_drawn[(tx,ty)] += 1

                cv2.circle(img, (ty*r+radius,
                                 tx*r+radius), size//circle_drawn[(tx,ty)], color, thickness=-1)
                last_val = hist[state]

                count +=1
                if last_val <= 0:
                    break

    # PyGame interface functions
    def on_init(self):
        """pygame init"""
        pygame.init()  # calls pygame.font.init()
        # init main screen and background
        self._display_surf = pygame.display.set_mode((self.img_width,
                                                      self.img_height),
                                                     pygame.HWSURFACE)
        self._background = pygame.Surface(self._display_surf.get_size()).convert()
        self._clock = pygame.time.Clock()

        # Font
        self._myfont = pygame.font.SysFont('Comic Sans MS', 30)
        self._running = True

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        # TODO! DEPRECATED!
        elif event.type == pygame.KEYDOWN:
            u = None  # control signal according to motion model
            action = None  # control input by user

            if event.key == pygame.K_LEFT:
                action = MoveWest2D
            elif event.key == pygame.K_RIGHT:
                action = MoveEast2D
            elif event.key == pygame.K_DOWN:
                action = MoveSouth2D
            elif event.key == pygame.K_UP:
                action = MoveNorth2D
            elif event.key == pygame.K_SPACE:
                action = TagAction()

            if action is None:
                return

            if self._controllable:
                reward = self._env.state_transition(action, execute=True)
                robot_pose = self._env.state.robot_position
                z = None
                if self._observation_model is not None:
                    z = self._observation_model.sample(self._env.state, action)
                    self._last_observation = z
                print("      state: %s" % str(self._env.state))
                print("     action: %s" % str(action.name))
                print("     observation: %s" % str(z))
                print("     reward: %s" % str(reward))
                print(" valid motions: %s" % str(self._env.grid_map.valid_motions(self._env.state.robot_position)))
                print("------------")
                if self._env.state.target_found:
                    self._running = False
            return action

    def on_loop(self):
        self._playtime += self._clock.tick(self._fps) / 1000.0

    def on_render(self):
        # self._display_surf.blit(self._background, (0, 0))
        self.render_env(self._display_surf)
        rx, ry = self._env.state.robot_position
        fps_text = "FPS: {0:.2f}".format(self._clock.get_fps())
        last_action = self._last_action
        last_action_str = "no_action" if last_action is None else str(last_action)
        pygame.display.set_caption("%s | Robot(%.2f,%.2f,%.2f) | %s | %s" %
                                   (last_action_str, rx, ry, 0,
                                    str(self._env.state.target_found),
                                    fps_text))
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while( self._running ):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()

    def render_env(self, display_surf):
        img = np.copy(self._img)
        r = self._res  # Not radius! It's resolution.

        # draw target
        tx, ty = self._env.state.target_position
        cv2.rectangle(img, (ty*r, tx*r), (ty*r+r, tx*r+r),
                      (255, 165, 0), -1)

        # draw robot
        rx, ry = self._env.state.robot_position
        r = self._res  # Not radius!
        # last_observation = self._last_observation.get(robot_id, None)
        # last_viz_observation = self._last_viz_observation.get(robot_id, None)
        # last_belief = self._last_belief.get(robot_id, None)
        if self._last_belief is not None:
            TagViz.draw_belief(img, self._last_belief, r, r//3, self._target_color)
        if self._last_observation is not None:
            TagViz.draw_observation(img, self._last_observation,
                                    rx, ry, 0, r, r//8, color=(20, 20, 180))

        TagViz.draw_robot(img, rx*r, ry*r, 0, r, color=(200, 12, 150))
        pygame.surfarray.blit_array(display_surf, img)

# TODO! DEPRECATED!
def unittest():
    worldmap, robot = world0
    env = TagEnvironment.from_str(worldmap)
    observation_model = TagObservationModel()
    viz = TagViz(env, controllable=True, observation_model=observation_model)
    viz.on_execute()

if __name__ == '__main__':
    unittest()
