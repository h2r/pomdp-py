import pomdp_py
import cv2
from ..models.transition_model import *
from ..models.reward_model import *

#### Interpret string as an initial world state ####
def interpret(self, worldstr, sensor_type, sensor_params):
    """
    worldstr (str) a string that describes the initial state of the world.
        For example: This string

            rx...
            .x.xT
            .....
            ***
            r: laser fov-90 min_range-1 max_range-10

        describes a 3 by 5 world where x indicates obsticles and T indicates
        the "target object". T could be replaced by any upper-case letter A-Z
        which will serve as the object's id. Lower-case letters a-z (except for x)
        serve as id for robot(s).

        After the world, the "***" signals description of the sensor for each robot.
        For example "r laser 90 1 10" means that robot `r` will have a Laser2Dsensor
        with fov 90, min_range 1.0, and max_range of 10.0.

    sensor_type (str): "laser" or "proximity",
    sensor_params (dict): {"..." parameters}
    """
    worldlines = []
    sensorlines = []
    mode = "world"
    for line in worldstr.splitlines():
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
    
    objects = {}    # objid -> pose
    obstacles = set({})  # objid
    robots = {}  # robot_id -> pose
    sensors = {}  # robot_id -> Sensor

    # Parse world
    for y, line in enumerate(worldlines):
        if len(line) != w:
            raise ValueError("World size inconsistent."\
                             "Expected width: %d; Actual Width: %d"
                             % (w, len(l)))
        for x, c in enumerate(line):
            if c == "x":
                # obstacle
                objid = 1000 + len(obstacles)  # obstacle id
                objects[objid] = ObjectState(objid, "obstacle", (x,y))
                obstacles.add(objid)
                
            elif c.isupper():
                # target object
                objid = ord(c)
                objects[objid] = ObjectState(objid, "target", (x,y))
                
            elif c.islower():
                # robot
                robot_id = -ord(c)
                robots[robot_id] = RobotState(robot_id, (x,y,0), set({}), None)

            else:
                assert c == ".", "Unrecognized character %s in worldstr" % c
    if len(robots) == 0:
        raise ValueError("No initial robot pose!")
    if len(objects) == 0:
        raise ValueError("No object!")

    # Parse sensors
    for line in sensorlines:
        robot_name = line.split(":")[0].strip()
        robot_id = -ord(robot_name)
        assert robot_id in robots, "Sensor specified for unknown robot %s" % (robot_name)
        
        sensor_setting = line.split(":")[1].strip()
        sensor_type = sensor_setting.split(" ")[0].strip()
        sensor_params = {}
        for token in sensor_setting.split(" ")[1:]:
            param_name = token.split("-")[0].strip()
            param_value = eval(token.split("-")[1].strip())
            sensor_params[param_name] = param_value
        
        if sensor_type == "laser":
            sensor = Laser2DSensor(**sensor_params)
        elif sensor_type == "proximity":
            sensor = ProximitySensor(**sensor_params)
        else:
            raise ValueError("Unknown sensor type %s" % sensor_type)
        sensors[robot_id] = sensor

    # Make init state
    init_state = MosOOState({**objects, **robots})
    return init_state, sensors, obstacle, (w,l)


class MosEnvironment(pomdp_py.Environment):
    """"""
    def __init__(self, dim, init_state, obstacles=set({})):
        """
        sensors (dict): Map from robot_id to sensor (Sensor);
                        Sensors equipped on robots; Used to determine
                        which objects should be marked as found.
        obstacles (set): set of object ids that are obstacles;
                            The set difference of all object ids then
                            yields the target object ids."""
        self.width, self.length = dim
        transition_model = MosTransitionModel(dim, )
        super().__init__(init_state, 


#### Visualization through pygame ####
class MosViz:

    def __init__(self, env):
