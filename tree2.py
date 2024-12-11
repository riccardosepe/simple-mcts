"""
This file contains all the code relative to a tree data structure
"""
import random


class Tree:
    class Node:
        def __init__(self, parent_node, _id, legal_actions, game_data):
            self._id = _id
            self._children = dict.fromkeys(legal_actions, None)
            self._parent_node = parent_node
            self._available_actions = legal_actions[:]
            self._visits = 0
            self._score = 0
            self._game_data = game_data

        def __repr__(self):
            return f"Node({self._id})"

        def add_child(self, child, action):
            # NB: this method is only meant to be used within the Tree class
            self._children[action] = child

        def visit(self):
            self._visits += 1

        def increase_score(self, score):
            self._score += score

        def random_action(self, exclude=False):
            action = random.choice(self._available_actions)
            if exclude:
                self._available_actions.remove(action)
            return action

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
            return len(self._available_actions) == 0

        @property
        def available_actions(self):
            return self._available_actions

        @property
        def is_root(self):
            return self._parent_node is None

        @property
        def score(self):
            return self._score

        @property
        def visits(self):
            return self._visits

    """
    NB: this tree is thought (for the moment) to support only environments with a maximum branching factor
    """
    def __init__(self, root_legal_actions, root_data):
        self._root = Tree.Node(None, 0, root_legal_actions, root_data)
        self._nodes = [self._root]

    def insert_node(self, parent_id, action, legal_actions, node_data):
        parent = self._nodes[parent_id]
        new_node = Tree.Node(parent, len(self._nodes), legal_actions, node_data)
        parent.add_child(new_node, action)
        self._nodes.append(new_node)
        return new_node

    @property
    def root(self):
        return self._root



