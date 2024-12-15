from unittest import TestCase
import unittest

from mcts import MCTS
from tictactoe_env import TicTacToeEnv


class TestMCTS(TestCase):

    def test_plan(self):
        env = TicTacToeEnv()

        env.reset(human_first=True)

        # build the following grid
        # O| |O
        #  |X|
        #  | |

        env.step(0)
        env.step(4)
        env.step(2)

        agent = MCTS(env)

        action = agent.plan(iterations_budget=1000)

        assert action == 1



if __name__ == '__main__':
    unittest.main()
