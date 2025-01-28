from absl.app import UsageError

from src.tree.tree import Tree, Node


class ChanceNode(Node):
    """
        Analogies with superclass:
        - always has only one parent
        Differences from superclass:
        - doesn't keep statistics
        - doesn't use UCT formula during selection

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self._score
        del self._game_data

    def __repr__(self):
        return f"Chance(visits={self.visits}, score={self.score}, action={self._action})"

    def __str__(self):
        return f"Chance(visits={self.visits}, score=n.a., action={self._action})"

    def update_score(self, score):
        raise RuntimeError

    @property
    def score(self):
        assert sum(map(lambda n: n.visits, filter(lambda n: n is not None, self.children.values()))) == self.visits
        return sum(map(lambda n: n.score * n.visits, filter(lambda n: n is not None, self.children.values()))) / self.visits


class ChoiceNode(Node):
    """
        Analogies with superclass:
        - keeps the statistics
        - uses UCT formula during selection
        Differences from superclass:
        - has multiple parents (not for now, need to implement state hashing)

    """
    def __init__(self, parent_node, *args, **kwargs):
        super().__init__(parent_node, *args, **kwargs)
        del self._parent_node
        self._parent_nodes = {parent_node.id: parent_node} if parent_node is not None else {}

    def __repr__(self):
        return f"Choice(visits={self.visits}, score={self.score}, state={self._action})"

    def __str__(self):
        return f"Choice(visits={self.visits}, score=n.a., state={self._action})"

    def set_root(self):
        assert self._parent_nodes is not None
        assert not all(map(lambda n: n is None, self._parent_nodes.values()))
        self._parent_nodes = None

    def add_parent(self, parent):
        self._parent_nodes[parent.id] = parent

    @property
    def parent(self):
        raise RuntimeError

    @property
    def parents(self):
        return self._parent_nodes

    @property
    def is_root(self):
        return self._parent_nodes is None or all(map(lambda n: n is None, self._parent_nodes.values()))

    @staticmethod
    def generate_node_hash(node_data):
        state = node_data['state']
        t = node_data['t']
        return state, t

class ChanceTree(Tree):
    """
        This class is a special case of a tree, used only for non-adversarial, single-agent settings. In this
        implementation, two things are mandatory:
        - nodes must be alternating between Chance and Choice
        - an episode must necessarily start with a Choice node and end with a Chance node
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._choice_nodes = dict()

    @staticmethod
    def create_root(root_legal_actions, root_data):
        return ChoiceNode(None, 0, root_legal_actions, root_data, None)

    def insert_node(self, parent_id, action, legal_actions, node_data, chance=None):
        parent = self._nodes[parent_id]
        new_id = self._last_id + 1
        self._last_id = new_id

        assert chance is not None
        if chance:
            new_node = ChanceNode(parent, new_id, legal_actions, node_data, action)
            parent.add_child(new_node)
            self._nodes[new_id] = new_node
        else:
            new_node = ChoiceNode(parent, new_id, legal_actions, node_data, action)
            parent.add_child(new_node)
            self._nodes[new_id] = new_node
            node_hash = ChoiceNode.generate_node_hash(node_data)
            self._choice_nodes[node_hash] = new_node

        return new_node