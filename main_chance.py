import random

from envs.frozenlake_env import MyFrozenLakeEnv
from src.ai.chance_mcts import ChanceMCTS

HUMAN = True
BOT = False
SEED = random.randint(0, 1000)


def main():
    print("Using seed ", SEED)
    env = MyFrozenLakeEnv(render_mode='human', is_slippery=True, map_name='4x4')
    env.reset(seed=SEED)
    keep_subtree = False

    agent = ChanceMCTS(env, seed=SEED, adversarial=env.adversarial, gamma=1, keep_subtree=keep_subtree, max_depth=50)

    env.render()

    done = False
    i = 0
    while not done:
        action = agent.plan(iterations_budget=10000)
        obs, _, done, _, _ = env.step(action)
        i += 1
        env.render()

        if keep_subtree:
            agent.determinize_chance_node(obs)  # TODO: hash states

    print(env.game_result())
    env.close()
    
if __name__ == '__main__':
    main()
   