import numpy as np

from hanoi_env import TowersOfHanoiEnv
from mcts2 import MCTS
from tictactoe_env import TicTacToeEnv

HUMAN = True
BOT = False


def main():
    env = TicTacToeEnv()

    obs = env.reset()
    agent = MCTS(env)

    player = True

    while True:
        obs = np.array(obs[0]).reshape((3,3))
        print(obs)

        if player:
            action = int(input("Insert an action: "))
            obs, _, done, _ = env.step(action)
        else:
            action = agent.plan(iterations_budget=10)
            obs, _, done, _ = env.step(action)

        if done:
            break

        player = not player

    
if __name__ == '__main__':
    main()
   