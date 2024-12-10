import random
import time

import numpy as np
from tree2 import Tree


class MCTS:
    def __init__(self, transition_model):
        legal_actions = transition_model.legal_actions
        self.tree = Tree(legal_actions)
        self.transition_model = transition_model

    def _select(self):
        node = self.tree.root
        while not node.is_leaf and node.is_fully_expanded:
            move, node = self.select_ucb(node)
            # TODO: do i need the output of step?
            self.transition_model.step(move)
        return node

    def _expand(self, node):
        random_move = node.random_move(exclude=True)
        self.transition_model.step(random_move)
        new_node = self.tree.insert_node(node.id, random_move, self.transition_model.legal_actions)
        return new_node

    def _simulate(self, node):
        ret = 0
        while True:
            action = random.choice(node.available_moves)
            _, r, d, _ = self.transition_model.step(action)
            # sparse / non-sparse setting
            ret += r
            if d:
                break

        return ret

    def _backpropagate(self, node, score):
        while not node.is_root:
            node.visit()
            node.increase_score(score)
            node = node.parent

    def _plan_iteration(self):
        """
        The core of the MCTS algorithm, i.e. the sequence of the four steps: Select, Expand, Simulate, Backpropagate.
        """
        selected_node = self._select()
        expanded_node = self._expand(selected_node)
        simulation_result = self._simulate(expanded_node)
        self._backpropagate(expanded_node, simulation_result)

    def plan(self, iterations_budget=None, time_budget=None):
        """
        Run a bunch of `_plan_iteration`s until either the iterations budget or the time budget is reached.

        :param iterations_budget: the maximum number of iterations to run
        :param time_budget: the maximum available time for a single move
        :return: the chosen move
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

        while elapsed_time < iterations_budget and iteration < time_budget:
            self._plan_iteration()
            elapsed_time = time.time() - start_time
            iteration += 1

        best_move = MCTS.select_ucb(self.tree.root)
        return best_move


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
