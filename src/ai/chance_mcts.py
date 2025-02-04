from src import MCTS
from src.evaluators.frozen_lake_evaluator import FrozenLakeEvaluator
from src.tree.chance_tree import ChanceTree, ChoiceNode, ChanceNode


class ChanceMCTS(MCTS):
    """
    Differences from superclass:
    - the Tree is necessarily a ChanceTree
    - the selection depends on the nature of the Node
    - the backpropagation skips the chance nodes
    """

    def __init__(self, *args, alpha=None, **kwargs):
        kwargs['adversarial'] = False
        super().__init__(*args, **kwargs)
        if alpha is None:
            self._evaluator = None
        else:
            self._evaluator = FrozenLakeEvaluator(self.transition_model.desc,
                                                  max_episode_length=self.transition_model.max_episode_length,
                                                  alpha=alpha)

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
                # insert a chance node
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

        # insert first a chance node
        new_chance_node = self.tree.insert_node(node.id,
                                                action=random_action,
                                                legal_actions=support_random_action,
                                                node_data=None,
                                                chance=True)

        # then insert a choice node
        # if there is already a hashed choice node, use that one instead
        node_data = self.transition_model.backup()
        node_hash = ChoiceNode.generate_node_hash(node_data)
        hashed_node = self.tree.get_choice_node_if_existing(node_hash)
        if hashed_node is None:
            new_choice_node = self.tree.insert_node(new_chance_node.id,
                                                    action=s,
                                                    legal_actions=self.transition_model.legal_actions,
                                                    node_data=self.transition_model.backup(),
                                                    chance=False)
        else:
            new_choice_node = hashed_node
            # IMPORTANT: when a hashed node is detected, it's important to first backpropagate its statistics to the root
            # and then go on with the simulation. At that point, backpropagate the new results along all paths
            self._backpropagate(new_chance_node, hashed_node.score, hashed_node.visits)

            new_chance_node.add_child(hashed_node)
            hashed_node.add_parent(new_chance_node)
        self.t += 1

        return new_choice_node

    def _evaluate(self, leaf_node):
        if self._evaluator is None:
            return super()._evaluate(leaf_node)
        else:
            node_state = leaf_node.game_state
            node_time = leaf_node.time
            value, features =  self._evaluator.evaluate(node_state, node_time, with_features=True)
            leaf_node.features = features
            return value

    def _backpropagate(self, node, score, visits=1):
        if node is None:
            return

        if isinstance(node, ChoiceNode):
            node.update_score(score)

        node.visit(visits)  # count the visits also for the chance nodes

        if isinstance(node, ChanceNode):
            self._backpropagate(node.parent, score * self.gamma, visits)
        else:
            for parent in node.parents.values():
                self._backpropagate(parent, score * self.gamma, visits)

    def determinize_chance_node(self, state):
        new_root = self.tree.root.children[state]
        self.tree.keep_subtree(new_root)

    def plan(self, *args, **kwargs):
        if isinstance(self.tree.root, ChanceNode):
            raise RuntimeError
        return super().plan(*args, **kwargs)
