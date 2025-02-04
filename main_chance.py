from envs.frozenlake_env import MyFrozenLakeEnv
from src.ai.chance_mcts import ChanceMCTS

HUMAN = True
BOT = False
SEED = 0

SIMPLE_MAP = [
    "SFFF",
    "FFHF",
    "FFFF",
    "FFFG"
]


def main(seed):
    print("Using seed ", seed)
    max_depth = 20
    env = MyFrozenLakeEnv(
        render_mode='human',
        is_slippery=True,
        # map_name='4x4',
        desc=SIMPLE_MAP,
        max_episode_length=max_depth)
    env.reset(seed=seed)
    keep_subtree = False

    alpha = 0.85

    agent = ChanceMCTS(env, seed=seed, adversarial=env.adversarial, gamma=1, keep_subtree=keep_subtree, max_depth=max_depth, alpha=alpha)

    env.render()

    done = False
    i = 0
    while not done and env.t < max_depth:
        action = agent.plan(iterations_budget=100)
        obs, _, done, _, _ = env.step(action)
        i += 1
        env.render()

        if keep_subtree:
            agent.determinize_chance_node(obs)  # TODO: hash states

    print(env.game_result())
    env.close()


if __name__ == '__main__':
    for s in range(10):
        main(s)
   