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
        del self._visits
        del self._score
        del self._game_data
        del self.visit
        del self.update_score
        del self.set_root






class ChoiceNode(Node):
    """
        Analogies with superclass:
        - keeps the statistics
        - uses UCT formula during selection
        Differences from superclass:
        - has multiple parents (not for now, need to implement state hashing)

    """



class ChanceTree(Tree):
    """
        This class is a special case of a tree, used only for non-adversarial, single-agent settings. In this
        implementation, two things are mandatory:
        - nodes must be alternating between Chance and Choice
        - an episode must necessarily start with a Choice node and end with a Chance node
    """

    @staticmethod
    def create_root(root_legal_actions, root_data):
        return ChanceNode(root_legal_actions, root_data)

