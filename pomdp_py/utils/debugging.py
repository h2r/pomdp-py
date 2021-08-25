"""
Utility functions making it easier to debug POMDP planning.
"""
from pomdp_py.algorithms.po_uct import QNode, VNode, RootVNode

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

def tree_stats(root, max_depth=None):
    stats = {
        'total_vnodes': 0,
        'total_qnodes': 0,
        'total_vnodes_children': 0,
        'total_qnodes_children': 0,
        'max_vnodes_children': 0,
        'max_qnodes_children': 0
    }
    _tree_stats_helper(root, 0, stats, max_depth=max_depth)
    stats['num_visits'] = root.num_visits
    stats['value'] = root.value
    return stats

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
            tree_stats_helper(root[c], depth+1, stats, max_depth=max_depth)
