from src import MCTS
from src.tree.chance_tree import ChanceTree, ChoiceNode, ChanceNode


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
            if chance_node.children[s] is not None:
                node = chance_node.children[s]
            else:
                node = self.tree.insert_node(chance_node.id,
                                             action=s,  # TODO: "action" is actually a state -> HASH
                                             legal_actions=self.transition_model.legal_actions,
                                             node_data=self.transition_model.backup(),
                                             chance=False)
            self.t += 1
        return node

    def _expand(self, node):
        random_action = node.random_action()
        support_random_action = self.transition_model.next_states(random_action)
        s, _, _, _, _ = self.transition_model.step(random_action)

        new_chance_node = self.tree.insert_node(node.id,
                                                action=random_action,
                                                legal_actions=support_random_action,  # TODO: is this ok?
                                                node_data=None,
                                                chance=True)  # TODO: is this ok?

        new_choice_node = self.tree.insert_node(new_chance_node.id,
                                                action=s,
                                                legal_actions=self.transition_model.legal_actions,
                                                node_data=self.transition_model.backup(),
                                                chance=False)
        self.t += 1

        return new_choice_node

    def _backpropagate(self, node, score):
        if node is None:
            return

        if isinstance(node, ChoiceNode):
            node.update_score(score)

        node.visit()  # count the visits also for the chance nodes
        self._backpropagate(node.parent, score * self.gamma)

    def determinize_chance_node(self, state):
        new_root = self.tree.root.children[state]
        self.tree.keep_subtree(new_root)

    def plan(self, *args, **kwargs):
        if isinstance(self.tree.root, ChanceNode):
            raise RuntimeError
        return super().plan(*args, **kwargs)