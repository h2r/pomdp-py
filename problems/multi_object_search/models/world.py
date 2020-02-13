

#### Interpret string as a gridworld and initial world state ####
def interpret(self, worldstr, sensor_type, sensor_params):
    """
    worldstr (str) a string that describes the initial state of the world.
        For example: This string

            rx...
            .x.xT
            .....

        describes a 3 by 5 world where x indicates obsticles and T indicates
        the "target object". T could be replaced by any upper-case letter A-Z
        which will serve as the object's id. Lower-case letters a-z (except for x)
        serve as id for robot(s).

    sensor_type (str): "laser" or "proximity",
    sensor_params (dict): {"..." parameters}
    """
    lines = [line for line in worldstr.splitlines()
         if len(line) > 0]
    w, l = len(lines[0]), len(lines)
    
    objects = {}    # objid -> pose
    obstacles = set({})  # objid
    robots = {}  # robot_id -> pose
    
    for y, line in enumerate(lines):
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

    # Make gridworld
    if sensor_type == "laser":
        sensor = Laser2DSensor(**sensor_params)
    elif sensor_type == "proximity":
        sensor = ProximitySensor(**sensor_params)
    else:
        raise ValueError("Unknown sensor type %s" % sensor_type)
    gridworld = GridWorld(w, l, set(objects.keys()), sensor, obstacles)

    # Make init state
    init_state = MosOOState({**objects, **robots})
    return gridworld, init_state
