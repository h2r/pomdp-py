################################################################################
# WARNING: The library does not require the installation of networkx and
# pygraphviz any more. But this code will be kept. If you would like to plot the
# MCTS search tree, you should install networkx and pygraphviz yourself.
################################################################################

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from pomdp_py import QNode, VNode, RootVNode, util

# ---- POUCT Visualization ---- #

def _build_graph(G, root, parent, conn, depth,
                 max_depth=None, visit_threshold=0,
                 max_value=None, min_value=None,
                 relabel_actions={},
                 relabel_observations={}):
    if max_depth is not None and depth >= max_depth:
        return
    G.add_node(root)
    if parent is not None:
        if isinstance(root, VNode):
            if len(relabel_observations) > 0:
                conn = relabel_observations[conn]
        else:
            if len(relabel_actions) > 0:
                conn = relabel_actions[conn]
        if max_value - min_value > 0.0:
            weight = util.remap(root.value,
                                min_value, max_value,
                                0.1, 3.0)
        else:
            weight = 0.1
        G.add_edge(parent, root, **{"label":conn,
                                    "weight": weight})
    min_value = float('inf')
    max_value = float('-inf')
    for c in root.children:
        max_value = max(max_value, root[c].value)
        min_value = min(min_value, root[c].value)
    for c in root.children:
        if root[c].num_visits > visit_threshold:
            _build_graph(G, root[c], root, c, depth+1,
                         max_value=max_value, min_value=min_value,
                         max_depth=max_depth, visit_threshold=visit_threshold,
                         relabel_actions=relabel_actions, relabel_observations=relabel_observations)

def _build_relabel_dict(root, conn, depth, actions, observations, max_depth=None, visit_threshold=0):
    """Traverse the tree and collect unique actions and observations,
    store them in dictionaries `actions` and `observations`. Requires
    all observations and actions to be hashable."""
    if max_depth is not None and depth >= max_depth:
        return
    if conn is not None:
        if isinstance(root, VNode):
            if conn not in observations:
                observations[conn] = "o" + str(len(observations))
        else:
            if conn not in actions:
                actions[conn] = "a" + str(len(actions))
    for c in root.children:
        if root[c].num_visits > visit_threshold:
            _build_relabel_dict(root[c], c,  depth+1, actions, observations, max_depth=max_depth, visit_threshold=visit_threshold)

def visualize_pouct_search_tree(root, max_depth=1,
                                visit_threshold=1, anonymize=False,
                                anonymize_actions=False, anonymize_observations=False,
                                output_file=None, use_dot=False, ax=None):
    """
    Visualize the given tree up to depth `max_depth`. Display nodes
    with number of visits >= `visit_threshold`.

    Caveat: This only works well if the search tree depth is shallow.
    For larger trees, please use a combination of tree debugger (utils.debugging.TreeDebugger)
    and visualizer (custom in heritance of visual.visualizer.Visualizer).

    If anonymize is True, will only display actions as a1,a2,... and observations
    as o1,o2,... .
    """
    relabel_actions = {}
    relabel_observations = {}
    if anonymize:
        anonymize_actions = True
        anonymize_observations = True
    if anonymize_actions or anonymize_observations:
        _build_relabel_dict(root, None, 0, relabel_actions, relabel_observations,
                            max_depth=max_depth, visit_threshold=visit_threshold)

        print("---- Action labels ----")
        action_map = {relabel_actions[action]:action for action in relabel_actions}
        for label in sorted(action_map):
            print("%s : %s" % (label, action_map[label]))
        print("---- Observation labels ----")
        observation_map = {relabel_observations[ob]:ob for ob in relabel_observations}
        for label in sorted(observation_map):
            print("%s : %s" % (label, observation_map[label]))

    if not anonymize_actions:
        relabel_actions = {}
    if not anonymize_observations:
        relabel_observations = {}

    # Build a networkx graph.
    G = nx.DiGraph()
    _build_graph(G, root, None, None, 0, max_depth=max_depth, visit_threshold=visit_threshold,
                 relabel_actions=relabel_actions, relabel_observations=relabel_observations)
    if use_dot:
        if output_file is None:
            raise TypeError("Please provide output .dot file.")
        nx.nx_agraph.write_dot(G, output_file)
        print("Dot file saved at %s" % output_file)
        print("Please run `dot -Tpng %s > %s.png" % (output_file, output_file))
    else:
        node_labels = {}
        color_map = []
        for node in G.nodes():
            belief_str = ""
            if hasattr(node, "belief"):
                belief_str = " | %d" % len(node.belief)
            if isinstance(node, RootVNode):
                color_map.append("cyan")
                node_labels[node] = "R(%d | %.2f%s)" % (node.num_visits, node.value, belief_str)
            elif isinstance(node, VNode):
                color_map.append("yellow")
                node_labels[node] = "V(%d | %.2f%s)" % (node.num_visits, node.value, belief_str)
            else:
                color_map.append("orange")
                node_labels[node] = "Q(%d | %.2f)" % (node.num_visits, node.value)
        edge_labels = {(edge[0],edge[1]): edge[2]["label"] for edge in G.edges(data=True)}
        edge_widths = [edge[2]["weight"] for edge in G.edges(data=True)]

        pos = graphviz_layout(G, prog='dot')
        nx.draw_networkx(G, pos,
                         node_color=color_map, labels=node_labels,
                         width=edge_widths,
                         font_size=7, ax=ax)
        nx.draw_networkx_edge_labels(G, pos,
                                     edge_labels=edge_labels, ax=ax)
        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)
