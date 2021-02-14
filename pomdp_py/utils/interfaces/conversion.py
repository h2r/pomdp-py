"""
Provides utility to convert a POMDP written in pomdp_py to
a POMDP file format (.pomdp or .pomdpx). Output to a file.
"""
import subprocess
import os
import pomdp_py
import numpy as np
import xml.etree.ElementTree as ET

def to_pomdp_file(agent, output_path=None,
                  discount_factor=0.95):
    """
    Pass in an Agent, and use its components to generate
    a .pomdp file to `output_path`.

    The .pomdp file format is specified at:
    http://www.pomdp.org/code/pomdp-file-spec.html

    Note:

    * It is assumed that the reward is independent of the observation.
    * The state, action, and observations of the agent must be
      explicitly enumerable.
    * The state, action and observations of the agent must be
      convertable to a string that does not contain any blank space.

    Args:
        agent (~pomdp_py.framework.basics.Agent): The agent
        output_path (str): The path of the output file to write in. Optional.
                           Default None.
        discount_factor (float): The discount factor
    Returns:
        (list, list, list): The list of states, actions, observations that
           are ordered in the same way as they are in the .pomdp file.
    """
    # Preamble
    try:
        all_states = list(agent.all_states)
        all_actions = list(agent.all_actions)
        all_observations = list(agent.all_observations)
    except NotImplementedError:
        raise ValueError("S, A, O must be enumerable for a given agent to convert to .pomdp format")

    content = "discount: %f\n" % discount_factor
    content += "values: reward\n" # We only consider reward, not cost.

    list_of_states = " ".join(str(s) for s in all_states)
    assert len(list_of_states.split(" ")) == len(all_states),\
        "states must be convertable to strings without blank spaces"
    content += "states: %s\n" % list_of_states

    list_of_actions = " ".join(str(a) for a in all_actions)
    assert len(list_of_actions.split(" ")) == len(all_actions),\
        "actions must be convertable to strings without blank spaces"
    content += "actions: %s\n" % list_of_actions

    list_of_observations = " ".join(str(o) for o in all_observations)
    assert len(list_of_observations.split(" ")) == len(all_observations),\
        "observations must be convertable to strings without blank spaces"
    content += "observations: %s\n" % list_of_observations

    # Starting belief state - they need to be normalized
    total_belief = sum(agent.belief[s] for s in all_states)
    content += "start: %s\n" % (" ".join(["%f" % (agent.belief[s]/total_belief)
                                          for s in all_states]))

    # State transition probabilities - they need to be normalized
    for s in all_states:
        for a in all_actions:
            probs = []
            for s_next in all_states:
                prob = agent.transition_model.probability(s_next, s, a)
                probs.append(prob)
            total_prob = sum(probs)
            for i, s_next in enumerate(all_states):
                prob_norm = probs[i] / total_prob
                content += 'T : %s : %s : %s %f\n' % (a, s, s_next, prob_norm)

    # Observation probabilities - they need to be normalized
    for s_next in all_states:
        for a in all_actions:
            probs = []
            for o in all_observations:
                prob = agent.observation_model.probability(o, s_next, a)
                probs.append(prob)
            total_prob = sum(probs)
            assert total_prob > 0.0,\
                "No observation is probable under state={} action={}"\
                .format(s_next, a)
            for i, o in enumerate(all_observations):
                prob_norm = probs[i] / total_prob
                content += 'O : %s : %s : %s %f\n' % (a, s_next, o, prob_norm)

    # Immediate rewards
    for s in all_states:
        for a in all_actions:
            for s_next in all_states:
                # We will take the argmax reward, which works for deterministic rewards.
                r = agent.reward_model.sample(s, a, s_next)
                content += 'R : %s : %s : %s : *  %f\n' % (a, s, s_next, r)

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(content)
    return all_states, all_actions, all_observations



def to_pomdpx_file(agent, pomdpconvert_path,
                   output_path=None,
                   discount_factor=0.95):
    """
    Converts an agent to a pomdpx file. This works by first converting the agent into a .pomdp file and then
    using the :code:`pomdpconvert` utility program to convert that file to a .pomdpx file. Check out
    :code:`pomdpconvert` at `github://AdaCompNUS/sarsop <https://github.com/AdaCompNUS/sarsop>`_

    Follow the instructions at https://github.com/AdaCompNUS/sarsop
    to download and build sarsop (I tested on Ubuntu 18.04, gcc version 7.5.0)

    See documentation for pomdpx at:
    https://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.PomdpXDocumentation

    First converts the agent into .pomdp, then convert it to pomdpx.

    Args:
        agent (~pomdp_py.framework.basics.Agent): The agent
        pomdpconvert_path (str): Path to the :code:`pomdpconvert` binary
        output_path (str): The path of the output file to write in. Optional.
                           Default None.
        discount_factor (float): The discount factor
    """
    pomdp_path = "./temp-pomdp.pomdp"
    to_pomdp_file(agent, pomdp_path,
                  discount_factor=discount_factor)
    proc = subprocess.Popen([pomdpconvert_path, pomdp_path])
    proc.wait()

    pomdpx_path = pomdp_path + "x"
    assert os.path.exists(pomdpx_path), "POMDPx conversion failed."

    with open(pomdpx_path, 'r') as f:
        content = f.read()

    if output_path is not None:
        os.rename(pomdpx_path, output_path)

    # Delete temporary files
    os.remove(pomdp_path)
    if os.path.exists(pomdpx_path):
        os.remove(pomdpx_path)


def parse_pomdp_solve_output(alpha_path, pg_path=None):
    """Parse the output of pomdp_solve, given
    by an .alpha file and a .pg file.

    Given a path to a .alpha file, read and interpret its contents.
    The file formats are specified at:
    https://www.pomdp.org/code/alpha-file-spec.html
    https://www.pomdp.org/code/pg-file-spec.html

    Note on policy graph (from the official website): To use this first requires
    knowing which of the policy graph states to start in. This can be achieved
    by finding the alpha vector with the maximal dot product with the initial
    starting state.

    Note: Parsing the .alpha file is required. The .pg path is optional (this
        is because I noticed some errors in the .pg file produced)

    Returns:
        alphas: [(alpha_vector, action_number) ...]
        policy_graph: a mapping from node number to (action_number, edges)
    """
    alphas = []  # (alpha_vector, action_number) tuples
    with open(alpha_path, 'r') as f:
        action_number = None
        alpha_vector = None
        mode = "action"
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                continue

            if mode == "action":
                action_number = int(line)
                mode = "alpha"
            elif mode == "alpha":
                alpha_vector = tuple(map(float, line.split(" ")))
                alphas.append((alpha_vector, action_number))
                mode = "action"
                action_number = None
                alpha_vector = None

    policy_graph = {}  # a mapping from node number to (action_number, edges)
    if pg_path is None:
        return alphas
    else:
        with open(pg_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                if len(line) == 0:
                    continue
                parts = list(map(int, line.split()))  # Splits on whitespace
                assert parts[0] not in policy_graph,\
                    "The node id %d already exists. Something wrong" % parts[0]
                policy_graph[parts[0]] = (parts[1], parts[2:])
        return alphas, policy_graph



class AlphaVectorPolicy(pomdp_py.Planner):
    """
    An offline POMDP policy is specified by a collection
    of alpha vectors, each associated with an action. When
    planning is needed, the dot product of these alpha vectors
    and the agent's belief vector is computed and the alpha
    vector leading to the maximum is the 'dominating' alpha
    vector and we return its corresponding action.

    An offline policy can be optionally represented as
    a policy graph. In this case, the agent can plan without
    actively maintaining a belief because the policy graph
    is a finite state machine that transitions by observations.

    This can be constructed using .policy file created by sarsop.
    """
    def __init__(self, alphas, states):
        """
        Args:
            alphas (list): A list of (alpha_vector, action) tuples.
                An alpha_vector is a list of floats :code:`[V1, V2, ... VN]`.
            states (list): List of states, ordered as in .pomdp file
        """
        self.alphas = alphas
        self.states = states

    def plan(self, agent):
        """Returns an action that is mapped by the agent belief, under this policy"""
        b = [agent.belief[s] for s in self.states]
        _, action = max(self.alphas,
                 key=lambda va: np.dot(b, va[0]))
        return action

    def value(self, belief):
        """
        Returns the value V(b) under this alpha vector policy.

        :math:`V(b) = max_{a\in\Gamma} {a} \cdot b`
        """
        b = [belief[s] for s in self.states]
        alpha_vector, _ = max(self.alphas,
                              key=lambda va: np.dot(b, va[0]))
        return np.dot(b, alpha_vector)

    @classmethod
    def construct(cls, policy_path,
                  states, actions, solver="pomdpsol"):
        """
        Returns an AlphaVectorPolicy, given `alphas`,
        which are the output of parse_appl_policy_file.

        Args:
            policy_path (str): Path to the generated .policy file
                (for sarsop) or .alpha file (for pomdp-solve)
            states (list): A list of States, in the same order as in the .pomdp file
            actions (list): A list of Actions, in the same order as in the .pomdp file
        Returns:
            AlphaVectorPolicy: The policy stored in the given policy file.
        """
        if solver == "pomdp-solve" or solver == "vi":
            return cls.construct_from_pomdp_solve(policy_path, states, actions)
        elif solver == "pomdpsol" or solver == "sarsop":
            alphas = []
            root = ET.parse(policy_path).getroot()
            for vector in root.find("AlphaVector").iter("Vector"):
                action = actions[int(vector.attrib["action"])]
                alpha_vector = tuple(map(float, vector.text.split()))
                alphas.append((alpha_vector, action))
            return AlphaVectorPolicy(alphas,
                                     states)

    @classmethod
    def construct_from_pomdp_solve(cls, alpha_path,
                                   states, actions):
        alphas_with_action_numbers = parse_pomdp_solve_output(alpha_path)
        alphas = [(alpha_vector, actions[action_number])
                  for alpha_vector, action_number in alphas_with_action_numbers]
        return AlphaVectorPolicy(alphas, states)


class PGNode:
    """A node on the policy graph"""
    def __init__(self, node_id, alpha_vector, action):
        self.node_id = node_id
        self.alpha_vector = alpha_vector
        self.action = action
    def __eq__(self, other):
        if isinstance(other, PolicyNode):
            return self.node_id == other.node_id
        return False
    def __hash__(self):
        return hash(self.node_id)
    def __str__(self):
        return repr(self)
    def __repr__(self):
        return "NodeID(%d)::AlphaVector(%s)::Action(%s)\n"\
            % (self.node_id, str(self.alpha_vector), self.action)


class PolicyGraph(pomdp_py.Planner):
    """A PolicyGraph encodes a POMDP plan. It
    can be constructed from the alphas and policy graph
    format output by Cassandra's pomdp-solver."""

    def __init__(self, nodes, edges, states):
        """
        Initializes a PolicyGraph.
        Args:
            nodes (list): A list of PGNodes
            edges (dict): Mapping from node_id to a dictionary {observation -> node_id}
            states (list): List of states, ordered as in .pomdp file
        """
        self.nodes = {n.node_id:n for n in nodes}
        self.edges = edges
        self.states = states
        self._current_node = None

    @classmethod
    def construct(cls, alpha_path, pg_path,
                  states, actions, observations):
        """
        See parse_pomdp_solve_output for detailed definitions of
        alphas and pg.

        Args:
            alpha_path (str): Path to .alpha file
            pg_path (str): Path to .pg file
            states (list): List of states, ordered as in .pomdp file
            actions (list): List of actions, ordered as in .pomdp file
            observations (list): List of observations, ordered as in .pomdp file
        """
        # alphas (list): List of ( [V1, V2, ... VN], A ) tuples
        # pg (dict): {node_id -> (A, edges)}
        alphas, pg = parse_pomdp_solve_output(alpha_path, pg_path)

        nodes = []
        for node_id, (alpha_vector, action_number) in enumerate(alphas):
            node = PGNode(node_id, alpha_vector, actions[action_number])
            nodes.append(node)

        edges = {}
        for node_id in pg:
            assert 0 <= node_id < len(nodes), "Invalid node id in policy graph"

            action_number, o_links = pg[node_id]
            assert actions[action_number] == nodes[node_id].action,\
                "Inconsistent action mapping"

            edges[node_id] = {}

            for o_index, next_node_id in enumerate(o_links):
                observation = observations[o_index]
                edges[node_id][observation] = next_node_id
        return PolicyGraph(nodes, edges, states)

    def plan(self, agent):
        """Returns an action that is mapped by the agent belief, under this policy"""
        if self._current_node is None:
            self._current_node = self._find_node(agent)
        return self._current_node.action

    def _find_node(self, agent):
        """Locate the node in the policy graph corresponding to the agent's current
        belief state. """
        b = [agent.belief[s] for s in self.states]
        nid = max(self.nodes,
                   key=lambda nid: np.dot(b, self.nodes[nid].alpha_vector))
        return self.nodes[nid]

    def update(self, agent, action, observation):
        """
        Updates the planner based on real action and observation.
        Basically sets the current node pointer based on the incoming
        observation."""
        if self._current_node is None:
            # Find out the node number using agent current belief
            self._current_node = self._find_node(agent)
        # Transition the current node following the graph
        self._current_node = self.nodes[self.edges[self._current_node.node_id][observation]]
