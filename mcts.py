import random
import time

import numpy as np
from tree import Tree


class MCTS:
    def __init__(self, transition_model, seed=None):
        legal_actions = transition_model.legal_actions
        self.tree = Tree(legal_actions, transition_model.backup())
        self.transition_model = transition_model
        random.seed(seed)
        np.random.seed(seed)

    def _select(self):
        node = self.tree.root
        while not node.is_leaf and node.is_fully_expanded:
            node = self.select_ucb(node)
            self.transition_model.step(node.action)
        return node

    def _expand(self, node):
        random_action = node.random_action()
        self.transition_model.step(random_action)
        new_node = self.tree.insert_node(node.id,
                                         random_action,
                                         self.transition_model.legal_actions,
                                         self.transition_model.backup())
        return new_node

    def _simulate(self, node):
        ret = 0
        while True:
            action = random.choice(self.transition_model.legal_actions)
            _, r, d, _, _ = self.transition_model.step(action)
            # sparse / non-sparse setting
            ret += r
            if d:
                break
        return ret

    def _backpropagate(self, node, score):
        while node is not None:
            node.visit()
            node.update_score(score)
            node = node.parent

    def _plan_iteration(self):
        """
        The core of the MCTS algorithm, i.e. the sequence of the four steps: Select, Expand, Simulate, Backpropagate.
        """
        # save the game state
        checkpoint = self.transition_model.backup()

        # 1. SELECT
        selected_node = self._select()

        # NB: very uncommon in practice, the following lines handle small game trees where it's possible to reach a
        # terminal state during the expansion phase

        if not selected_node.is_terminal:
            # 2. EXPAND
            expanded_node = self._expand(selected_node)
            terminal_node = expanded_node

            if not expanded_node.is_terminal:
                # 3. SIMULATE
                score = self._simulate(expanded_node)

            else:
                score = expanded_node.game_reward

        else:
            terminal_node = selected_node
            score = selected_node.game_reward

        # 4. BACKPROPAGATE
        self._backpropagate(terminal_node, score)

        # restore the game state
        self.transition_model.load(checkpoint)

    def plan(self, iterations_budget=None, time_budget=None):
        """
        Run a bunch of `_plan_iteration`s until either the iterations budget or the time budget is reached.

        :param iterations_budget: the maximum number of iterations to run
        :param time_budget: the maximum available time for a single action
        :return: the chosen action
        """

        if iterations_budget is None and time_budget is None:
            raise ValueError("Either iterations_budget or time_budget must be set")
        elif iterations_budget is None:
            iterations_budget = np.inf
        elif time_budget is None:
            time_budget = np.inf

        elapsed_time = 0
        iteration = 0

        start_time = time.time()

        while elapsed_time < time_budget and iteration < iterations_budget:
            self._plan_iteration()
            elapsed_time = time.time() - start_time
            iteration += 1

        best_child = self.tree.root.best_child
        self.tree.keep_subtree(best_child)
        return best_child.action

    def opponent_action(self, action):
        if self.tree.root.is_leaf:
            self.tree.root.ply(action)
        else:
            new_root = self.tree.root.children[action]
            self.tree.keep_subtree(new_root)

    @staticmethod
    def _ucb(node, parent, c=np.sqrt(2)):
        """
        Calculates the Upper Confidence Bound for a tree.
        :param node: the node for which it calculates the UCB
        :param parent: the parent node of `node`
        :param c: the coefficient of the formula
        """

        exploitation = node.score / node.visits
        if parent.visits == 0:
            exploration = 0
        else:
            exploration = np.sqrt(
                np.log(parent.visits) / node.visits
            )
        return exploitation + c * exploration

    @staticmethod
    def select_ucb(parent):
        scores = [(idx, MCTS._ucb(node, parent)) for idx, node in parent.children.items()]
        best_action = max(scores, key=lambda x: x[1])[0]
        return parent.children[best_action]
