from src import MCTS
from src.evaluators.frozen_lake_evaluator import FrozenLakeEvaluator
from src.explainers.frozen_lake_explainer import FrozenLakeExplainer
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
        self.trajectory = None
        if alpha is None:
            self._evaluator = None
        else:
            self._evaluator = FrozenLakeEvaluator(self.transition_model.desc,
                                                  max_episode_length=self.transition_model.max_episode_length,
                                                  alpha=alpha)
        self._explainer = FrozenLakeExplainer()

    def _build_tree(self):
        return ChanceTree(self.transition_model.legal_actions, self.transition_model.backup())

    def _select(self):
        node = self.tree.root
        self.trajectory = [node]
        while not node.is_leaf and node.is_fully_expanded:
            chance_node = self.select_ucb(node)
            s, _, _, _, _ = self.transition_model.step(chance_node.action)
            # TODO: HASHING. FOR THE MOMENT (FROZEN LAKE) THE STATE IS JUST AN INTEGER
            self.t += 1

            self.trajectory.append(chance_node)

            if chance_node.children[s] is not None:
                node = chance_node.children[s]
                self.trajectory.append(node)
            else:
                # return current node for expansion
                return chance_node
        return node

    def _expand(self, node):
        if isinstance(node, ChoiceNode):
            random_action = node.random_action()
            support_random_action = self.transition_model.next_states(random_action)
            s, _, _, _, _ = self.transition_model.step(random_action)

            # insert first a chance node
            new_chance_node = self.tree.insert_node(node.id,
                                                    action=random_action,
                                                    legal_actions=support_random_action,
                                                    node_data=None,
                                                    chance=True)
            self.trajectory.append(new_chance_node)
        elif isinstance(node, ChanceNode):
            new_chance_node = node
        else:
            raise RuntimeError

        # then insert a choice node if it's not hashed
        new_choice_node = self._insert_or_get_choice_node(new_chance_node, self.transition_model.backup())
        self.t += 1

        self.trajectory.append(new_choice_node)
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

    def _backpropagate(self, _, score, visits=1):
        assert len(self.trajectory) > 0
        while len(self.trajectory) > 0:
            node = self.trajectory.pop()
            if isinstance(node, ChoiceNode):
                node.update_score(score)
            node.visit(visits)

    def _backpropagate_parents(self, node, score, visits=1):
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

    def plan(self, *args, explain=False, **kwargs):
        if isinstance(self.tree.root, ChanceNode):
            raise RuntimeError
        best_action = super().plan(*args, **kwargs)
        if explain:
            assert not self._keep_subtree
            pass
        return best_action

    def _insert_or_get_choice_node(self, parent_node, node_data):
        s = node_data['state']
        node_data = self.transition_model.backup()
        node_hash = ChoiceNode.generate_node_hash(node_data)
        hashed_node = self.tree.get_choice_node_if_existing(node_hash)
        if hashed_node is None:
            new_choice_node = self.tree.insert_node(parent_node.id,
                                                    action=s,
                                                    legal_actions=self.transition_model.legal_actions,
                                                    node_data=self.transition_model.backup(),
                                                    chance=False)
        else:
            new_choice_node = hashed_node
            # IMPORTANT: when a hashed node is detected, it's important to first backpropagate its statistics to the root
            # and then go on with the simulation. At that point, backpropagate the new results along all paths
            self._backpropagate_parents(parent_node, hashed_node.score, hashed_node.visits)

            parent_node.add_child(hashed_node)
            hashed_node.add_parent(parent_node)

        return new_choice_node