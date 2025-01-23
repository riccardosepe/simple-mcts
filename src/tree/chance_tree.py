from src.tree.tree import Tree, Node


class ChanceNode(Node):
    """
        Analogies with superclass:
        - always has only one parent
        Differences from superclass:
        - doesn't keep statistics
        - doesn't use UCT formula during selection

    """



class ChoiceNode(Node):
    """
        Analogies with superclass:
        - keeps the statistics
        - uses UCT formula during selection
        Differences from superclass:
        - has multiple parents

    """



class ChanceTree(Tree):
    """
        This class is a special case of a tree, used only for non-adversarial, single-agent settings. In thiis
        implementation, two things are mandatory:
        - nodes must be alternating between Chance and Choice
        - an episode must necessarily start with a Choice node and end with a Chance node
    """