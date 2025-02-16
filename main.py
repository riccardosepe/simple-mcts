from src import MCTS
from envs.tictactoe_env import TicTacToeEnv

HUMAN = True
BOT = False
SEED = 0


def main():
    env = TicTacToeEnv()
    # env = TowersOfHanoiEnv(num_disks=3)
    player = BOT

    # env.reset()
    env.reset(human_first=player)

    agent = MCTS(env, seed=SEED, adversarial=env.adversarial, gamma=1, max_depth=20)

    env.render()

    done = False

    while not done:
        if player is HUMAN:
            action = int(input("Insert an action: "))
            while action not in env.legal_actions:
                action = int(input("Illegal action. Insert another one: "))
            obs, _, done, _, _ = env.step(action)
            agent.opponent_action(action)
        elif player is BOT:
            action = agent.plan(iterations_budget=100)
            obs, _, done, _, _ = env.step(action)

        env.render()
        if env.adversarial:
            player = not player

    print(env.game_result())

    
if __name__ == '__main__':
    main()
   