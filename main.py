from envs.frozenlake_env import MyFrozenLakeEnv
from envs.hanoi_env import TowersOfHanoiEnv
from src.mcts import MCTS
from envs.tictactoe_env import TicTacToeEnv

HUMAN = True
BOT = False
SEED = 0


def main():
    env = TicTacToeEnv()
    env = TowersOfHanoiEnv(num_disks=3)
    player = BOT

    for seed in range(5):
        env = MyFrozenLakeEnv(render_mode='human', is_slippery=False, map_name='8x8')
        transition_model = MyFrozenLakeEnv(render_mode='ansi', is_slippery=False, map_name='8x8')
        env.reset(seed=seed)
        transition_model.reset(seed=seed)
        # env.reset(human_first=player)

        agent = MCTS(transition_model, seed=seed, adversarial=env.adversarial, gamma=0.95, keep_subtree=False)

        env.render()

        done = False
        i = 0
        while not done:
            if player is HUMAN:
                action = int(input("Insert an action: "))
                while action not in env.legal_actions:
                    action = int(input("Illegal action. Insert another one: "))
                obs, _, done, _, _ = env.step(action)
                agent.opponent_action(action)
            elif player is BOT:
                # NB: iterations_budget < b^2 might create problems (b is the branching factor)
                # TODO: handle cases with iterations_budget < 81?
                action = agent.plan(iterations_budget=10000)
                obs, _, done, _, _ = env.step(action)
                transition_model.load(env.backup())
                agent.init_tree(env.legal_actions, env.backup())
                i += 1

            env.render()
            if env.adversarial:
                player = not player

        print(env.game_result())
        env.close()
    
if __name__ == '__main__':
    main()
   