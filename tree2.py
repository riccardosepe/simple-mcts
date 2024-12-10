"""
This file contains all the code relative to a tree data structure
"""
import random


class Tree:
    class Node:
        def __init__(self, parent_node, _id, legal_actions):
            self._id = _id
            self._children = dict.fromkeys(legal_actions, None)
            self._parent_node = parent_node
            self._available_moves = legal_actions[:]
            self._visits = 0
            self._score = 0

        def add_child(self, child, move):
            # NB: this method is only meant to be used within the Tree class
            self._children[move] = child

        def visit(self):
            self._visits += 1

        def increase_score(self, score):
            self._score += score

        def random_move(self, exclude=False):
            move = random.choice(self._available_moves)
            if exclude:
                self._available_moves.remove(move)
            return move

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
            return len(self._available_moves) == 0

        @property
        def available_moves(self):
            return self._available_moves

        @property
        def is_root(self):
            return self._parent_node is None

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



