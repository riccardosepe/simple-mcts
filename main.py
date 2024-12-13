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

    env.reset()
    env.render()

    player = True
    done = False

    while not done:
        if player:
            action = int(input("Insert an action: "))
            while action not in env.legal_actions:
                action = int(input("Illegal action. Insert another one: "))
            obs, _, done, _, _ = env.step(action)
        else:
            agent = MCTS(env)
            action = agent.plan(iterations_budget=1000)
            obs, _, done, _, _ = env.step(action)

        env.render()
        player = not player

    print(env.game_result())

    
if __name__ == '__main__':
    random.seed(SEED)
    np.random.seed(SEED)
    main()
   