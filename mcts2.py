import numpy as np
from tree2 import Tree


class MCTS:
    def __init__(self):
        self.tree = Tree()

    def _select(self):
        node = self.tree.root
        while not node.is_leaf and node.is_fully_expanded:
            node = self._ucb_child(node)

        return node

    def _expand(self, node):
        pass

    def _simulate(self, node):
        pass

    def _backpropagate(self, node):
        pass

    @staticmethod
    def ucb(node, parent, c=0.1):
        """
        Calculates the Upper Confidence Bound for an MCTS.
        :param node: the node for which it calculates the UCB
        :param parent: the parent node of `node`
        :param c: the coefficient of the formula
        """

        exploitation = node.data.value / node.data.simulations
        if parent.data.simulations == 0:
            exploration = 0
        else:
            exploration = np.sqrt(
                np.log(parent.data.simulations) / node.data.simulations
            )
        return exploitation + c * exploration

    @staticmethod
    def _ucb_child(parent):
        scores = [MCTS.ucb(node, parent) for node in parent.children]
        return parent.children[np.argmax(scores)]
