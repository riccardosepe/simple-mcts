"""
This file contains all the code relative to a tree data structure
"""
import random
from functools import cmp_to_key


class Node:
    def __init__(self, parent_node, _id, legal_actions, game_data, action):
        self._id = _id
        self._children = dict.fromkeys(legal_actions, None)
        self._parent_node = parent_node
        self._available_actions = legal_actions[:]
        self._visits = 0
        self._score = 0
        self._game_data = game_data
        self._action = action

    def __repr__(self):
        return f"{self.player}(id={self._id}, visits={self._visits}, score={self._score}, action={self._action})"

    def add_child(self, child):
        # NB: this method is only meant to be used within the Tree class
        self._children[child.action] = child
        self._available_actions.remove(child.action)

    def visit(self, n=1):
        self._visits += n

    def update_score(self, score):
        self._score += score

    def random_action(self):
        action = random.choice(self._available_actions)
        return action

    def set_root(self):
        assert self._parent_node is not None
        self._parent_node = None

    def ply(self, action):
        assert self.is_root
        self._available_actions.remove(action)
        del self._children[action]

    @property
    def _best_child(self):
        children_list = list(self._children.values())
        return sorted(children_list, key=cmp_to_key(Node.node_cmp))[0]

    @staticmethod
    def node_cmp(this, other):
        if this.visits > other.visits:
            return -1
        elif this.visits < other.visits:
            return 1
        else:
            if this.score > other.score:
                return -1
            elif this.score < other.score:
                return 1
            else:
                # break ties randomly
                return random.choice([-1, 1])

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
        return all(map(lambda x: x is None, self._children.values()))

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

    @property
    def is_terminal(self):
        return self._game_data['done']

    @property
    def game_reward(self):
        return self._game_data['reward']

    @property
    def game_state(self):
        return self._game_data['state']

    @property
    def time(self):
        return self._game_data['t']

    @property
    def action(self):
        return self._action

    @property
    def player(self):
        return self._game_data['player']

class Tree:

    @staticmethod
    def create_root(root_legal_actions, root_data):
        return Node(None, 0, root_legal_actions, root_data, None)

    def __init__(self, root_legal_actions, root_data):
        self._root = self.create_root(root_legal_actions, root_data)
        self._nodes = {0: self._root}
        self._last_id = 0

    def __repr__(self):
        s = ', '.join(map(str, self._nodes))
        return f"Tree({s})"

    def __getitem__(self, index):
        return self._nodes[index]

    def insert_node(self, parent_id, action, legal_actions, node_data, **kwargs):
        parent = self._nodes[parent_id]
        new_id = self._last_id + 1
        self._last_id = new_id
        new_node = Node(parent, new_id, legal_actions, node_data, action)
        parent.add_child(new_node)
        self._nodes[new_id] = new_node
        return new_node

    def delete_subtree(self, node, parent):
        """
        This method deletes a subtree that starts from `node` (included). It is only used within the method
        `keep_subtree`.
        """
        self._delete_subtree(node)
        assert node in parent.children.values()
        del parent.children[node.action]
        del self._nodes[node.id]

    def _delete_subtree(self, node):
        """
        This method recursively delete a subtree that starts from `node` (excluded)
        """
        if node.is_leaf:
            return
        for child_id in list(node.children):
            n = node.children[child_id]
            if n is None:
                continue
            self._delete_subtree(n)
            del node.children[child_id]
            try:
                del self._nodes[n.id]
            except KeyError:
                # TODO: this node was deleted following a different subtree
                pass

    def keep_subtree(self, node):
        assert node in self._root.children.values()

        # Delete the subtree relative to all the other children
        for action in list(self._root.children):
            n = self._root.children[action]
            if n is node:
                continue

            self.delete_subtree(n, self._root)

        # At this point, I still have the root with only the selected child (`node`)
        del self._nodes[self._root.id]
        del self._root
        self._root = node
        self._root.set_root()

        assert self._root is self._nodes[self._root.id]


    def visualize(self, node_id=None, level=None, mode='repr'):
        if level is None:
            level = self._last_id
        if node_id is None:
            node = self._root
        else:
            node = self._nodes[node_id]
        self._visualize(node, 0, level, mode)

    def _visualize(self, node, depth, level, mode):
        """
        Recursively prints the structure of the tree starting from the given node.

        :param node: The starting node for printing (usually the root node).
        :param depth: The current depth of the node, used for indentation.
        """
        indent = "  " * depth

        if mode == 'repr':
            s = node.__repr__()
        else:
            s = node.__str__()

        print(f"{indent}{s}")

        for action, child in node.children.items():
            if child is not None and level > 0:
                self._visualize(child, depth + 1, level-1, mode)

    @property
    def root(self):
        return self._root



