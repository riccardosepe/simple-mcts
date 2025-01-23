from envs.frozenlake_env import MyFrozenLakeEnv
from src.ai.chance_mcts import ChanceMCTS

HUMAN = True
BOT = False
SEED = 0


def main():
    env = MyFrozenLakeEnv(render_mode='human', is_slippery=True, map_name='4x4')
    env.reset(seed=SEED)

    agent = ChanceMCTS(env, seed=SEED, adversarial=env.adversarial, gamma=0.95, keep_subtree=True)

    env.render()

    done = False
    i = 0
    while not done:

        # NB: iterations_budget < b^2 might create problems (b is the branching factor)
        # TODO: handle cases with iterations_budget < 81?
        action = agent.plan(iterations_budget=10000)
        obs, _, done, _, _ = env.step(action)
        i += 1

        env.render()

    print(env.game_result())
    env.close()
    
if __name__ == '__main__':
    main()
   