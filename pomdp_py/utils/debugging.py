"""
Utility functions making it easier to debug POMDP planning.
"""
import sys
from pomdp_py.algorithms.po_uct import TreeNode, QNode, VNode, RootVNode
from pomdp_py.utils import typ

def sorted_nodes(nodes):
    return sorted(nodes, key=lambda n: str(n))


class _QNodePP(QNode):
    """QNode for better printing"""
    def __init__(self, qnode, parent_edge=None):
        super().__init__(qnode.num_visits, qnode.value)
        self.parent_edge = parent_edge
        self.children = qnode.children

    def __str__(self):
        return TreeDebugger.single_node_str(self,
                                            parent_edge=self.parent_edge)

class _VNodePP(VNode):
    """VNode for better printing"""
    def __init__(self, vnode, parent_edge=None):
        super().__init__(vnode.num_visits)
        self.parent_edge = parent_edge
        self.children = vnode.children

    def __str__(self):
        return TreeDebugger.single_node_str(self,
                                            parent_edge=self.parent_edge)


class TreeDebugger:
    """
    Helps you debug the search tree; A search tree is a tree
    that contains a subset of future histories, organized into
    QNodes (value represents Q(b,a); children are observations) and
    VNodes (value represents V(b); children are actions).
    """
    def __init__(self, tree):
        """
        Args:
            tree (VNode): the root node of a search tree. For example,
                the tree built by POUCT after planning an action,
                which can be accessed by agent.tree.
        """
        if not isinstance(tree, TreeNode):
            raise ValueError("Expecting tree to be a TreeNode, but got {}".format(type(tree)))

        self.tree = TreeDebugger._node_pp(tree)
        self.current = self.tree   # points to the node the user is interacting with
        self.parent_edge = None    # stores the edge that leads to current
        self._stats_cache = {}

    def __str__(self):
        return str(self.current)

    def __repr__(self):
        nodestr = TreeDebugger.single_node_str(self.current, parent_edge=self.parent_edge)
        return "TreeDebugger@\n{}".format(nodestr)

    def _get_stats(self):
        if id(self.current) in self._stats_cache:
            stats = self._stats_cache
        else:
            stats = TreeDebugger.tree_stats(self.current)
            self._stats_cache[id(self.current)] = stats
        return stats

    def num_nodes(self, kind='all'):
        """
        Returns the total number of nodes in the tree rooted at "current"
        """
        stats = self._get_stats()
        res = {
            'all': stats['total_vnodes'] + stats['total_qnodes'],
            'q': stats['total_qnodes'],
            'v': stats['total_vnodes']
        }
        if kind in res:
            return res[kind]
        else:
            raise ValueError("Invalid value for kind={}; Valid values are {}"\
                             .format(kind, list(res.keys())))

    @property
    def nn(self):
        return self.num_nodes(kind='all')

    @property
    def nq(self):
        return self.num_nodes(kind='q')

    @property
    def nv(self):
        return self.num_nodes(kind='v')

    def layer(self, depth, as_debuggers=False):
        """
        Returns a list of nodes at the given depth.

        Args:
            depth (int): Depth of the tree
            as_debuggers (bool): True if return a list of TreeDebugger objects,
                one for each tree on the layer.
        """
        pass

    @property
    def root(self):
        return self.tree

    @property
    def r(self):
        """For convenience during debugging"""
        return self.root

    @property
    def c(self):
        return self.current

    @staticmethod
    def _node_pp(node):
        # We want to return the node, but we don't want to print it on pdb with
        # its default string. But instead, we want to print it with our own
        # string formatting.
        if isinstance(node, VNode):
            return _VNodePP(node)
        else:
            return _QNodePP(node)


    @staticmethod
    def single_node_str(node, parent_edge=None, indent=1, include_children=True):
        """
        Returns a string for printing given a single vnode.
        """
        if isinstance(node, VNode):
            color = typ.green
        else:
            assert isinstance(node, QNode)
            color = typ.red

        output = ""
        if parent_edge is not None:
            output += "- {} -".format(typ.white(str(parent_edge)))

        output += color(str(node.__class__.__name__))\
            + "(n={}, v={:.3f})".format(node.num_visits,
                                    node.value)
        if include_children:
            output += "\n"
            for i, action in enumerate(sorted_nodes(node.children)):
                child = node.children[action]
                child_info = TreeDebugger.single_node_str(child, include_children=False)

                spaces = "    " * indent
                output += "{}- [{}] {}: {}".format(spaces, i,
                                                   typ.white(str(action)),
                                                   child_info)
                if i < len(node.children) - 1:
                    output += "\n"
        return output

    @staticmethod
    def tree_stats(root, max_depth=None):
        stats = {
            'total_vnodes': 0,
            'total_qnodes': 0,
            'total_vnodes_children': 0,
            'total_qnodes_children': 0,
            'max_vnodes_children': 0,
            'max_qnodes_children': 0
        }
        TreeDebugger._tree_stats_helper(root, 0, stats, max_depth=max_depth)
        stats['num_visits'] = root.num_visits
        stats['value'] = root.value
        return stats

    @staticmethod
    def _tree_stats_helper(root, depth, stats, max_depth=None):
        if max_depth is not None and depth >= max_depth:
            return
        else:
            if isinstance(root, VNode):
                stats['total_vnodes'] += 1
                stats['total_vnodes_children'] += len(root.children)
                stats['max_vnodes_children'] = max(stats['max_vnodes_children'], len(root.children))
            else:
                stats['total_qnodes'] += 1
                stats['total_qnodes_children'] += len(root.children)
                stats['max_qnodes_children'] = max(stats['max_qnodes_children'], len(root.children))

            for c in root.children:
                TreeDebugger._tree_stats_helper(root[c], depth+1, stats, max_depth=max_depth)



def print_tree(tree, max_depth=None, complete=False):
    """
    Prints out the POUCT tree.
    """
    _print_tree_helper(tree, "", 0, max_depth=max_depth, complete=complete)

def _print_tree_helper(root, parent_edge, depth, max_depth=None, complete=False):
    if max_depth is not None and depth >= max_depth:
        return
    print("%s%s" % ("    "*depth, str(parent_edge)))
    print("%s-%s" % ("    "*depth, str(root)))
    if root is None:
        return
    for c in root.children:
        if complete or (root[c].num_visits > 1):
            if isinstance(root[c], QNode):
                _print_tree_helper(root[c], c, depth+1, max_depth=max_depth, complete=complete)
            else:
                _print_tree_helper(root[c], c, depth, max_depth=max_depth, complete=complete)

def print_preferred_actions(tree, max_depth=None):
    """
    Print out the currently preferred actions up to given `max_depth`
    """
    _print_preferred_actions_helper(tree, 0, max_depth=max_depth)

def _print_preferred_actions_helper(root, depth, max_depth=None):
    if max_depth is not None and depth >= max_depth:
        return
    best_child = None
    best_value = float('-inf')
    if root is None:
        return
    for c in root.children:
        if root[c].value > best_value:
            best_child = c
            best_value = root[c].value
    equally_good = []
    if isinstance(root, VNode):
        for c in root.children:
            if not(c == best_child) and root[c].value == best_value:
                equally_good.append(c)

    if best_child is not None and root[best_child] is not None:
        if isinstance(root[best_child], QNode):
            print("  %s  %s" % (str(best_child), str(equally_good)))
        _print_preferred_actions_helper(root[best_child], depth+1, max_depth=max_depth)
