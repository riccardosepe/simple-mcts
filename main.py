import numpy as np

from hanoi_env import TowersOfHanoiEnv
from mcts2 import MCTS
from tictactoe_env import TicTacToeEnv

HUMAN = True
BOT = False


def main():
    env = TicTacToeEnv()

    env.reset()

    player = True

    while True:
        env.render()

        if player:
            action = int(input("Insert an action: "))
            obs, _, done, _ = env.step(action)
        else:
            agent = MCTS(env)
            action = agent.plan(iterations_budget=10)
            obs, _, done, _ = env.step(action)

        if done:
            break

        player = not player

    
if __name__ == '__main__':
    main()
   