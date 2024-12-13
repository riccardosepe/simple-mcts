import numpy as np
import random

from hanoi_env import TowersOfHanoiEnv
from mcts import MCTS
from tictactoe_env import TicTacToeEnv

HUMAN = True
BOT = False
SEED = 0


def main():
    env = TicTacToeEnv()
    player = True

    env.reset(human_first=player)

    agent = MCTS(env)

    env.render()

    done = False

    while not done:
        if player:
            action = int(input("Insert an action: "))
            while action not in env.legal_actions:
                action = int(input("Illegal action. Insert another one: "))
            obs, _, done, _, _ = env.step(action)
        else:
            # NB: iterations_budget < 81 might create problems
            action = agent.plan(iterations_budget=100)
            obs, _, done, _, _ = env.step(action)

        env.render()
        player = not player

    print(env.game_result())

    
if __name__ == '__main__':
    random.seed(SEED)
    np.random.seed(SEED)
    main()
   