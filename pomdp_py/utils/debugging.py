"""This module contains utility functions making it easier to debug POMDP
planning.

TreeDebugger
************

The core debugging functionality for POMCP/POUCT search trees is incorporated
into the TreeDebugger.  It is designed for ease of use during a :code:`pdb` or
:code:`ipdb` debugging session. Here is a minimal example usage:

.. code-block:: python

   from pomdp_py.utils import TreeDebugger
   from pomdp_problems.tiger import TigerProblem

   # pomdp_py.Agent
   agent = TigerProblem.create("tiger-left", 0.5, 0.15).agent

   # suppose pouct is a pomdp_py.POUCT object (POMCP works too)
   pouct = pomdp_py.POUCT(max_depth=4, discount_factor=0.95,
                          num_sims=4096, exploration_const=200,
                          rollout_policy=tiger_problem.agent.policy_model)

   action = pouct.plan(agent)
   dd = TreeDebugger(agent.tree)
   import pdb; pdb.set_trace()

When the program executes, you enter the pdb debugger, and you can:

.. code-block:: text

    (Pdb) dd.pp
    _VNodePP(n=4095, v=-19.529)(depth=0)
    ├─── ₀listen⟶_QNodePP(n=4059, v=-19.529)
    │    ├─── ₀tiger-left⟶_VNodePP(n=2013, v=-16.586)(depth=1)
    │    │    ├─── ₀listen⟶_QNodePP(n=1883, v=-16.586)
    │    │    │    ├─── ₀tiger-left⟶_VNodePP(n=1441, v=-8.300)(depth=2)
    ... # prints out the entire tree; Colored in terminal.

    (Pdb) dd.p(1)
    _VNodePP(n=4095, v=-19.529)(depth=0)
    ├─── ₀listen⟶_QNodePP(n=4059, v=-19.529)
    │    ├─── ₀tiger-left⟶_VNodePP(n=2013, v=-16.586)(depth=1)
    │    │    ├─── ₀listen⟶_QNodePP(n=1883, v=-16.586)
    │    │    ├─── ₁open-left⟶_QNodePP(n=18, v=-139.847)
    │    │    └─── ₂open-right⟶_QNodePP(n=112, v=-57.191)
    ... # prints up to depth 1

Note that the printed texts are colored in the terminal.

You can retrieve the subtree through indexing:

.. code-block:: text

    (Pdb) dd[0]
    listen⟶_QNodePP(n=4059, v=-19.529)
        - [0] tiger-left: VNode(n=2013, v=-16.586)
        - [1] tiger-right: VNode(n=2044, v=-16.160)

    (Pdb) dd[0][1][2]
    open-right⟶_QNodePP(n=15, v=-148.634)
        - [0] tiger-left: VNode(n=7, v=-20.237)
        - [1] tiger-right: VNode(n=6, v=8.500)

You can obtain the currently preferred action sequence by:

.. code-block:: text

    (Pdb) dd.mbp
       listen  []
       listen  []
       listen  []
       listen  []
       open-left  []
     _VNodePP(n=4095, v=-19.529)(depth=0)
     ├─── ₀listen⟶_QNodePP(n=4059, v=-19.529)
     │    └─── ₁tiger-right⟶_VNodePP(n=2044, v=-16.160)(depth=1)
     │         ├─── ₀listen⟶_QNodePP(n=1955, v=-16.160)
     │         │    └─── ₁tiger-right⟶_VNodePP(n=1441, v=-8.300)(depth=2)
     │         │         ├─── ₀listen⟶_QNodePP(n=947, v=-8.300)
     │         │         │    └─── ₁tiger-right⟶_VNodePP(n=768, v=0.022)(depth=3)
     │         │         │         ├─── ₀listen⟶_QNodePP(n=462, v=0.022)
     │         │         │         │    └─── ₁tiger-right⟶_VNodePP(n=395, v=10.000)(depth=4)
     │         │         │         │         ├─── ₁open-left⟶_QNodePP(n=247, v=10.000)

:code:`mbp` stands for "mark best plan".

To explore more features, browse the list of methods in the documentation.
"""
import sys
from pomdp_py.algorithms.po_uct import TreeNode, QNode, VNode, RootVNode
from pomdp_py.utils import typ, similar, special_char

SIMILAR_THRESH = 0.6
DEFAULT_MARK_COLOR = "blue"
MARKED = {}  # tracks marked nodes on tree

def _node_pp(node, e=None, p=None, o=None):
    # We want to return the node, but we don't want to print it on pdb with
    # its default string. But instead, we want to print it with our own
    # string formatting.
    if isinstance(node, VNode):
        return _VNodePP(node, parent_edge=e, parent=p, original=o)
    else:
        return _QNodePP(node, parent_edge=e, parent=p, original=o)

class _NodePP:

    def __init__(self, node, parent_edge=None, parent=None, original=None):
        """node: either VNode or QNode (the actual node on the tree) """
        self.parent_edge = parent_edge
        self.parent = parent
        self.children = node.children
        self.print_children = True
        if original is None:
            self.original = node
        else:
            self.original = original

    def __hash__(self):
        return id(self.original)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return id(self.original) == id(other.original)
        else:
            return False

    @property
    def marked(self):
        return id(self.original) in MARKED

    def to_edge(self, key):
        if key in self.children:
            return key
        elif type(key) == int:
            edges = list(sorted_by_str(self.children.keys()))
            return edges[key]
        elif type(key) == str:
            chosen = max(self.children.keys(),
                         key=lambda edge: similar(str(edge), key))
            if similar(str(chosen), key) >= SIMILAR_THRESH:
                return chosen
        raise ValueError("Cannot access children with key {}".format(key))

    def __getitem__(self, key):
        """
        When debugging, you can access the child of a node by the key
        of the following types:
        - the key is an action or observation object that points to a child;
          that is, key in self.children is True.
        - the key is an integer corresponding to the list of children shown
          when printing the node in the debugger
        - the key is a string that is similar to the string
          version of any of the action or observation edges;
          the most similar one will be chosen; The threshold
          of similarity is SIMILAR_THRESH
        """
        edge = self.to_edge(key)
        c = self.children[edge]
        if isinstance(c, _NodePP):
            original = c.original
        else:
            original = None
        return _node_pp(c, e=edge, p=self, o=original)

    def __contains__(self, key):
        try:
            self.to_edge(key)
            return True
        except ValueError:
            return False

    @staticmethod
    def interpret_print_type(opt):
        if opt.startswith("b") or opt.startswith("m"):
            opt = "marked-only"
        elif opt.startswith("s"):
            opt = "summary"
        elif opt.startswith("c"):
            opt = "complete"
        else:
            raise ValueError("Cannot understand print type: {}".format(opt))
        return opt

    def p(self, opt=None, **kwargs):
        if opt is None:
            max_depth = None
            print_type_opt = kwargs.get('t', "summary")
        elif type(opt) == int:
            max_depth = opt
            print_type_opt = kwargs.get('t', "summary")
        elif type(opt) == str:
            print_type_opt = opt
            max_depth = kwargs.get('d', None)
        else:
            raise ValueError("Cannot deal with opt of type {}".format(type(opt)))
        self.print_tree(max_depth=max_depth,
                        print_type=_NodePP.interpret_print_type(print_type_opt))

    @property
    def pp(self):
        self.print_tree(max_depth=None)

    def print_tree(self, **options):
        """Prints the tree, rooted at self"""
        _NodePP._print_tree_helper(self, 0, "", [None], -1, **options)

    @staticmethod
    def _print_tree_helper(root,
                           depth,   # depth of root
                           parent_edge,
                           branch_positions,  # list of 'first', 'middle', 'last' for each level prior to root
                           child_index,  # Index of the root as a child of parent
                           max_depth=None,
                           print_type="summary"):
        """
        pos_among_children is either 'first', 'middle', or 'last'
        """
        if max_depth is not None and depth > max_depth:
            return
        if root is None:
            return

        # Print the tree branches for all levels up to current root
        branches = ""
        preceding_positions = branch_positions[:-1]  # all positions except for current root
        for pos in preceding_positions:
            if pos is None:
                continue
            elif pos == "first" or pos == "middle":
                branches += "│    "
            else:  # "last"
                branches += "     "

        last_position = branch_positions[-1]
        if last_position is None:
            pass
        elif last_position == "first" or last_position == "middle":
            branches += "├─── "
        else:  # last
            branches += "└─── "

        root.print_children = False
        if child_index >= 0:
            line = branches + str(child_index).translate(special_char.SUBSCRIPT) + str(root)
        else:
            line = branches + str(root)
        if isinstance(root, VNode):
            line += typ.cyan("(depth="+str(depth)+")")

        print(line)

        for i, c in enumerate(sorted_by_str(root.children)):

            skip = True
            if root[c].marked:
                skip = False
            elif print_type == "complete":
                skip = False
            elif (root[c].num_visits > 1):
                skip = False
            if print_type == "marked-only" and not root[c].marked:
                skip = True

            if not skip:
                if isinstance(root[c], QNode):
                    next_depth = depth
                else:
                    next_depth = depth + 1

                if i == len(root.children) - 1:
                    next_pos = "last"
                elif i == 0:
                    next_pos = "first"
                else:
                    next_pos = "middle"

                _NodePP._print_tree_helper(root[c],
                                           next_depth,
                                           c,
                                           branch_positions + [next_pos],
                                           i,
                                           max_depth=max_depth,
                                           print_type=print_type)


class _QNodePP(_NodePP, QNode):
    """QNode for better printing"""
    def __init__(self, qnode, **kwargs):
        QNode.__init__(self, qnode.num_visits, qnode.value)
        _NodePP.__init__(self, qnode, **kwargs)

    def __str__(self):
        return TreeDebugger.single_node_str(self,
                                            parent_edge=self.parent_edge,
                                            include_children=self.print_children)

class _VNodePP(_NodePP, VNode):
    """VNode for better printing"""
    def __init__(self, vnode, **kwargs):
        VNode.__init__(self, vnode.num_visits)
        _NodePP.__init__(self, vnode, **kwargs)

    def __str__(self):
        return TreeDebugger.single_node_str(self,
                                            parent_edge=self.parent_edge,
                                            include_children=self.print_children)


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

        self.tree = _node_pp(tree)
        self.current = self.tree   # points to the node the user is interacting with
        self._stats_cache = {}

    def __str__(self):
        return str(self.current)

    def __repr__(self):
        nodestr = TreeDebugger.single_node_str(self.current,
                                               parent_edge=self.current.parent_edge)
        return "TreeDebugger@\n{}".format(nodestr)

    def __getitem__(self, key):
        if type(key) == tuple:
            n = self.current
            for k in key:
                n = n[k]
            return n
        else:
            return self.current[key]

    def _get_stats(self):
        if id(self.current) in self._stats_cache:
            stats = self._stats_cache[id(self.current)]
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
    def depth(self):
        """Tree depth starts from 0 (root node only).
        It is the largest number of edges on a path from root to leaf."""
        stats = self._get_stats()
        return stats['max_depth']

    @property
    def d(self):
        """alias for depth"""
        return self.depth

    @property
    def num_layers(self):
        """Returns the number of layers;
        It is the number of layers of nodes, which equals to depth + 1"""
        return self.depth + 1

    @property
    def nl(self):
        """alias for num_layers"""
        return self.num_layers

    @property
    def nn(self):
        """Returns the total number of nodes in the tree"""
        return self.num_nodes(kind='all')

    @property
    def nq(self):
        """Returns the total number of QNodes in the tree"""
        return self.num_nodes(kind='q')

    @property
    def nv(self):
        """Returns the total number of VNodes in the tree"""
        return self.num_nodes(kind='v')

    def l(self, depth, as_debuggers=True):
        """alias for layer"""
        return self.layer(depth, as_debuggers=as_debuggers)

    def layer(self, depth, as_debuggers=True):
        """
        Returns a list of nodes at the given depth. Will only return VNodes.
        Warning: If depth is high, there will likely be a huge number of nodes.

        Args:
            depth (int): Depth of the tree
            as_debuggers (bool): True if return a list of TreeDebugger objects,
                one for each tree on the layer.
        """
        if depth < 0 or depth > self.depth:
            raise ValueError("Depth {} is out of range (0-{})".format(depth, self.depth))
        nodes = []
        self._layer_helper(self.current, 0, depth, nodes)
        return nodes

    def _layer_helper(self, root, current_depth, target_depth, nodes, as_debuggers=True):
        if current_depth == target_depth:
            if isinstance(root, VNode):
                if as_debuggers:
                    nodes.append(TreeDebugger(root.original))
                else:
                    nodes.append(root)
        else:
            for c in sorted_by_str(root.children):
                if isinstance(root[c], QNode):
                    next_depth = current_depth
                else:
                    next_depth = current_depth + 1
                self._layer_helper(root[c],
                                   next_depth,
                                   target_depth,
                                   nodes)

    @property
    def leaf(self):
        worklist = [self.current]
        seen = set({self.current})
        leafs = []
        while len(worklist) > 0:
            node = worklist.pop()
            if len(node.children) == 0:
                leafs.append(node)
            else:
                for c in node.children:
                    if node[c] not in seen:
                        worklist.append(node[c])
                        seen.add(node[c])
        return leafs

    def step(self, key):
        """Updates current interaction node to follow the
        edge along key"""
        edge = self.current.to_edge(key)
        self.current = self[edge]
        print("step: " + str(edge))


    def s(self, key):
        """alias for step"""
        return self.step(key)

    def back(self):
        """move current node of interaction back to parent"""
        self.current = self.current.parent

    @property
    def b(self):
        """alias for back"""
        self.back()

    @property
    def root(self):
        """The root node when first creating this TreeDebugger"""
        return self.tree

    @property
    def r(self):
        """alias for root"""
        return self.root

    @property
    def c(self):
        """Current node of interaction"""
        return self.current

    def p(self, *args, **kwargs):
        """print tree"""
        return self.current.p(*args, **kwargs)

    @property
    def pp(self):
        """print tree, with preset options"""
        return self.current.pp

    @property
    def mbp(self):
        """Mark Best and Print.
        Mark the best sequence, and then print with only the marked nodes"""
        self.mark(self.bestseq, color="yellow")
        self.p("marked-only")

    @property
    def pm(self):
        """Print marked only"""
        self.p("marked-only")

    def mark_sequence(self, seq, color=DEFAULT_MARK_COLOR):
        """
        Given a list of keys (understandable by __getitem__ in _NodePP),
        mark nodes (both QNode and VNode) along the path in the tree.
        Note this sequence starts from self.current; So self.current will
        also be marked.
        """
        node = self.current
        MARKED[id(node.original)] = interpret_color(color)
        for key in seq:
            MARKED[id(node[key].original)] = interpret_color(color)
            node = node[key]

    def mark(self, seq, **kwargs):
        """alias for mark_sequence"""
        return self.mark_sequence(seq, **kwargs)

    def mark_path(self, dest, **kwargs):
        """paths the path to dest node"""
        return self.mark(self.path_to(dest), **kwargs)

    def markp(self, dest, **kwargs):
        """alias to mark_path"""
        return self.mark_path(dest, **kwargs)

    @property
    def clear(self):
        """Clear the marks"""
        global MARKED
        MARKED = {}

    @property
    def bestseq(self):
        """Returns a list of actions, observation sequence
        that have the highest value for each step. Such
        a sequence is "preferred".

        Also, prints out the list of preferred actions for each step
        into the future"""
        return self.preferred_actions(self.current, max_depth=None)

    def bestseqd(self, max_depth):
        """
        alias for bestseq except with
        """
        return self.preferred_actions(self.current, max_depth=max_depth)

    @staticmethod
    def single_node_str(node, parent_edge=None, indent=1, include_children=True):
        """
        Returns a string for printing given a single vnode.
        """
        if hasattr(node, "marked") and node.marked:
            color_fn = MARKED[id(node.original)]
            opposite_color = color = lambda s: typ.bold(color_fn(s))
        elif isinstance(node, VNode):
            color = typ.green
            opposite_color = typ.red
        else:
            assert isinstance(node, QNode)
            color = typ.red
            opposite_color = typ.green

        output = ""
        if parent_edge is not None:
            output += opposite_color(str(parent_edge)) + "⟶"

        output += color(str(node.__class__.__name__))\
            + "(n={}, v={:.3f})".format(node.num_visits,
                                    node.value)
        if include_children:
            output += "\n"
            for i, action in enumerate(sorted_by_str(node.children)):
                child = node.children[action]
                child_info = TreeDebugger.single_node_str(child, include_children=False)

                spaces = "    " * indent
                output += "{}- [{}] {}: {}".format(spaces, i,
                                                   typ.white(str(action)),
                                                   child_info)
                if i < len(node.children) - 1:
                    output += "\n"
            output += "\n"
        return output

    @staticmethod
    def preferred_actions(root, max_depth=None):
        """
        Print out the currently preferred actions up to given `max_depth`
        """
        seq = []
        TreeDebugger._preferred_actions_helper(root, 0, seq, max_depth=max_depth)
        return seq

    @staticmethod
    def _preferred_actions_helper(root, depth, seq, max_depth=None):
        # don't care about last layer action because it's outside of planning
        # horizon and only has initial value.
        if max_depth is not None and depth > max_depth:
            return
        if root is None or len(root.children) == 0:
            return
        best_child = root.to_edge(0)
        best_value = root[0].value
        for c in root.children:
            if root[c].value > best_value:
                best_child = c
                best_value = root[c].value
        seq.append(best_child)
        equally_good = []
        if isinstance(root, VNode):
            for c in root.children:
                if not(c == best_child) and root[c].value == best_value:
                    equally_good.append(c)

        if best_child is not None and root[best_child] is not None:
            if isinstance(root[best_child], QNode):
                print("  %s  %s" % (typ.yellow(str(best_child)), str(equally_good)))
                next_depth = depth
            else:
                next_depth = depth + 1

            TreeDebugger._preferred_actions_helper(root[best_child], next_depth, seq,
                                                   max_depth=max_depth)

    def path(self, dest):
        """alias for path_to;
        Example usage:

        marking path from root to the first node on the second layer:

            dd.mark(dd.path(dd.layer(2)[0]))
        """
        return self.path_to(dest)

    def path_to(self, dest):
        """Returns a list of keys (actions / observations) that represents the path from
        self.current to the given node `dest`. Returns None if the path does not
        exist.  Uses DFS. Can be useful for marking path to a node to a specific
        layer. Note that the returned path is a list of keys (i.e. edges), not nodes.
        """
        # dest may be in the returned list of layer() which could be a TreeDebugger.
        if isinstance(dest, TreeDebugger):
            dest = dest.current
        worklist = [self.current]
        seen = set({self.current})
        parent = {self.current: None}
        while len(worklist) > 0:
            node = worklist.pop()
            if node == dest:
                return self._get_path(self.current, dest, parent)
            for c in node.children:
                if node[c] not in seen:
                    worklist.append(node[c])
                    seen.add(node[c])
                    parent[node[c]] = (node, c)
        return None

    def _get_path(self, start, dest, parent):
        """Helper method for path_to"""
        v = dest
        path = []
        while v != start:
            v, edge = parent[v]
            path.append(edge)
        return list(reversed(path))

    @staticmethod
    def tree_stats(root, max_depth=None):
        """Gether statistics about the tree"""
        stats = {
            'total_vnodes': 0,
            'total_qnodes': 0,
            'total_vnodes_children': 0,
            'total_qnodes_children': 0,
            'max_vnodes_children': 0,
            'max_qnodes_children': 0,
            'max_depth': 0
        }
        TreeDebugger._tree_stats_helper(root, 0, stats, max_depth=max_depth)
        stats['num_visits'] = root.num_visits
        stats['value'] = root.value
        return stats

    @staticmethod
    def _tree_stats_helper(root, depth, stats, max_depth=None):
        if max_depth is not None and depth > max_depth:
            return
        else:
            if isinstance(root, VNode):
                stats['total_vnodes'] += 1
                stats['total_vnodes_children'] += len(root.children)
                stats['max_vnodes_children'] = max(stats['max_vnodes_children'], len(root.children))
                stats['max_depth'] = max(stats['max_depth'], depth)
            else:
                stats['total_qnodes'] += 1
                stats['total_qnodes_children'] += len(root.children)
                stats['max_qnodes_children'] = max(stats['max_qnodes_children'], len(root.children))

            for c in root.children:
                if isinstance(root[c], QNode):
                    next_depth = depth
                else:
                    next_depth = depth + 1
                TreeDebugger._tree_stats_helper(root[c], next_depth, stats, max_depth=max_depth)

def sorted_by_str(enumerable):
    return sorted(enumerable, key=lambda n: str(n))

def interpret_color(colorstr):
    if colorstr.lower() in typ.colors:
        return eval("typ.{}".format(colorstr))
    else:
        raise ValueError("Invalid color: {};\n"
                         "The available ones are {}".format(colorstr, typ.colors))
