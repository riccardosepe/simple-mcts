"""
This file contains all the code relative to a tree data structure
"""

class Node:
    def __init__(self, parent_node, _id, max_children):
        self._id = _id
        self._children = []
        self._parent_node = parent_node
        self._max_children = max_children

    def add_child(self, child):
        self._children.append(child)

    @property
    def id(self):
        return self._id

    @property
    def parent(self):
        return self._parent_node

    @property
    def children(self):
        return self._children

    @property
    def is_leaf(self):
        return len(self._children) == 0

    @property
    def is_fully_expanded(self):
        return len(self._children) == self._max_children


class Tree:
    """
    NB: this tree is thought (for the moment) to support only environments with a maximum branching factor
    """
    def __init__(self, branching_factor=4):
        self._root = Node(None, 0, branching_factor)
        self._nodes = []
        # TODO: this is going to be problematic for the scenarios where some moves are not allowed
        self._branching_factor = branching_factor

    def insert_node(self, parent_id):
        parent = self._nodes[parent_id]
        new_node = Node(parent, len(self._nodes), self._branching_factor)
        parent.add_child(new_node)

    @property
    def root(self):
        return self._root



