"""The Environment"""

import pomdp_py
import copy
from pomdp_problems.multi_object_search.models.transition_model import *
from pomdp_problems.multi_object_search.models.reward_model import *
from pomdp_problems.multi_object_search.models.components.sensor import *
from pomdp_problems.multi_object_search.domain.state import *

class MosEnvironment(pomdp_py.Environment):
    """"""
    def __init__(self, dim, init_state, sensors, obstacles=set({})):
        """
        Args:
            sensors (dict): Map from robot_id to sensor (Sensor);
                            Sensors equipped on robots; Used to determine
                            which objects should be marked as found.
            obstacles (set): set of object ids that are obstacles;
                                The set difference of all object ids then
                                yields the target object ids."""
        self.width, self.length = dim
        self.sensors = sensors
        self.obstacles = obstacles
        transition_model = MosTransitionModel(dim,
                                              sensors,
                                              set(init_state.object_states.keys()))
        # Target objects, a set of ids, are not robot nor obstacles
        self.target_objects = \
            {objid
             for objid in set(init_state.object_states.keys()) - self.obstacles
             if not isinstance(init_state.object_states[objid], RobotState)}
        reward_model = GoalRewardModel(self.target_objects)
        super().__init__(init_state,
                         transition_model,
                         reward_model)
        
    @property
    def robot_ids(self):
        return set(self.sensors.keys())

    def state_transition(self, action, execute=True, robot_id=None):
        """state_transition(self, action, execute=True, **kwargs)

        Overriding parent class function.
        Simulates a state transition given `action`. If `execute` is set to True,
        then the resulting state will be the new current state of the environment.

        Args:
            action (Action): action that triggers the state transition
            execute (bool): If True, the resulting state of the transition will
                            become the current state.

        Returns:
            float or tuple: reward as a result of `action` and state
            transition, if `execute` is True (next_state, reward) if `execute`
            is False.

        """
        assert robot_id is not None, "state transition should happen for a specific robot"

        next_state = copy.deepcopy(self.state)
        next_state.object_states[robot_id] =\
            self.transition_model[robot_id].sample(self.state, action)
        
        reward = self.reward_model.sample(self.state, action, next_state,
                                          robot_id=robot_id)
        if execute:
            self.apply_transition(next_state)
            return reward
        else:
            return next_state, reward        

#### Interpret string as an initial world state ####
def interpret(worldstr):
    """
    Interprets a problem instance description in `worldstr`
    and returns the corresponding MosEnvironment.

    For example: This string
    
    .. code-block:: text

        rx...
        .x.xT
        .....
        ***
        r: laser fov=90 min_range=1 max_range=10
    
    describes a 3 by 5 world where x indicates obsticles and T indicates
    the "target object". T could be replaced by any upper-case letter A-Z
    which will serve as the object's id. Lower-case letters a-z (except for x)
    serve as id for robot(s).

    After the world, the :code:`***` signals description of the sensor for each robot.
    For example "r laser 90 1 10" means that robot `r` will have a Laser2Dsensor
    with fov 90, min_range 1.0, and max_range of 10.0.    

    Args:
        worldstr (str): a string that describes the initial state of the world.

    Returns:
        MosEnvironment: the corresponding environment for the world description.
            
    """
    worldlines = []
    sensorlines = []
    mode = "world"
    for line in worldstr.splitlines():
        line = line.strip()
        if len(line) > 0:
            if line == "***":
                mode = "sensor"
                continue
            if mode == "world":
                worldlines.append(line)
            if mode == "sensor":
                sensorlines.append(line)
    
    lines = [line for line in worldlines
             if len(line) > 0]
    w, l = len(worldlines[0]), len(worldlines)
    
    objects = {}    # objid -> object_state(pose)
    obstacles = set({})  # objid
    robots = {}  # robot_id -> robot_state(pose)
    sensors = {}  # robot_id -> Sensor

    # Parse world
    for y, line in enumerate(worldlines):
        if len(line) != w:
            raise ValueError("World size inconsistent."\
                             "Expected width: %d; Actual Width: %d"
                             % (w, len(line)))
        for x, c in enumerate(line):
            if c == "x":
                # obstacle
                objid = 1000 + len(obstacles)  # obstacle id
                objects[objid] = ObjectState(objid, "obstacle", (x,y))
                obstacles.add(objid)
                
            elif c.isupper():
                # target object
                objid = len(objects)
                objects[objid] = ObjectState(objid, "target", (x,y))
                
            elif c.islower():
                # robot
                robot_id = interpret_robot_id(c)
                robots[robot_id] = RobotState(robot_id, (x,y,0), (), None)

            else:
                assert c == ".", "Unrecognized character %s in worldstr" % c
    if len(robots) == 0:
        raise ValueError("No initial robot pose!")
    if len(objects) == 0:
        raise ValueError("No object!")

    # Parse sensors
    for line in sensorlines:
        if "," in line:
            raise ValueError("Wrong Fromat. SHould not have ','. Separate tokens with space.")
        robot_name = line.split(":")[0].strip()
        robot_id = interpret_robot_id(robot_name)
        assert robot_id in robots, "Sensor specified for unknown robot %s" % (robot_name)
        
        sensor_setting = line.split(":")[1].strip()
        sensor_type = sensor_setting.split(" ")[0].strip()
        sensor_params = {}
        for token in sensor_setting.split(" ")[1:]:
            param_name = token.split("=")[0].strip()
            param_value = eval(token.split("=")[1].strip())
            sensor_params[param_name] = param_value
        
        if sensor_type == "laser":
            sensor = Laser2DSensor(robot_id, **sensor_params)
        elif sensor_type == "proximity":
            sensor = ProximitySensor(robot_id, **sensor_params)
        else:
            raise ValueError("Unknown sensor type %s" % sensor_type)
        sensors[robot_id] = sensor

    return (w,l), robots, objects, obstacles, sensors

def interpret_robot_id(robot_name):
    return -ord(robot_name)


#### Utility functions for building the worldstr ####
def equip_sensors(worldmap, sensors):
    """
    Args:
        worldmap (str): a string that describes the initial state of the world.
        sensors (dict) a map from robot character representation (e.g. 'r') to a
    string that describes its sensor (e.g. 'laser fov=90 min_range=1 max_range=5
    angle_increment=5')

    Returns:
        str: A string that can be used as input to the `interpret` function
    """
    worldmap += "\n***\n"
    for robot_char in sensors:
        worldmap += "%s: %s\n" % (robot_char, sensors[robot_char])
    return worldmap

def make_laser_sensor(fov, dist_range, angle_increment, occlusion):
    """
    Returns string representation of the laser scanner configuration.
    For example:  "laser fov=90 min_range=1 max_range=10"

    Args:
        fov (int or float): angle between the start and end beams of one scan (degree).
        dist_range (tuple): (min_range, max_range)
        angle_increment (int or float): angular distance between measurements (rad).
        occlusion (bool): True if consider occlusion

    Returns:
        str: String representation of the laser scanner configuration.
    """
    fovstr = "fov=%s" % str(fov)
    rangestr = "min_range=%s max_range=%s" % (str(dist_range[0]), str(dist_range[1]))
    angicstr = "angle_increment=%s" % (str(angle_increment))
    occstr = "occlusion_enabled=%s" % str(occlusion)
    return "laser %s %s %s %s" % (fovstr, rangestr, angicstr, occstr)

def make_proximity_sensor(radius, occlusion):
    """
    Returns string representation of the proximity sensor configuration.
    For example: "proximity radius=5 occlusion_enabled=False"

    Args:
        radius (int or float)
        occlusion (bool): True if consider occlusion
    Returns:
        str: String representation of the proximity sensor configuration.
    """
    radiustr = "radius=%s" % str(radius)
    occstr = "occlusion_enabled=%s" % str(occlusion)
    return "proximity %s %s" % (radiustr, occstr)
