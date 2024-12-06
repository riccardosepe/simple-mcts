from hanoi_env import TowersOfHanoiEnv
from mcts import MCTS


def main():
    env = TowersOfHanoiEnv()

    obs = env.reset()
    agent = MCTS(obs)


    
if __name__ == '__main__':
    main()
   