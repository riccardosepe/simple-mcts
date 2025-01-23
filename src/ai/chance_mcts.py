from src import MCTS
from src.tree.chance_tree import ChanceTree


class ChanceMCTS(MCTS):
    """
    Differences from superclass:
    - the Tree is necessarily a ChanceTree
    - the selection depends on the nature of the Node
    - the backpropagation skips the chance nodes
    """

    def __init__(self, *args, **kwargs):
        kwargs['adversarial'] = False
        super().__init__(*args, **kwargs)

    def _build_tree(self):
        return ChanceTree(self.transition_model.legal_actions, self.transition_model.backup())

    def _select(self):
        node = self.tree.root
        while not node.is_leaf and node.is_fully_expanded:
            chance_node = self.select_ucb(node)
            s, _, _, _, _ = self.transition_model.step(chance_node.action)
            # TODO: HASHING. FOR THE MOMENT (FROZEN LAKE) THE STATE IS JUST AN INTEGER
            if s in chance_node.children:
                node = chance_node.children[s]
            else:
                node = self.tree.insert_node(chance_node.id,
                                             )