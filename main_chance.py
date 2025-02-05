from envs.frozenlake_env import MyFrozenLakeEnv
from src.ai.chance_mcts import ChanceMCTS

HUMAN = True
BOT = False
SEED = 0


def main(seed):
    print("Using seed ", seed)
    max_depth = 20
    env = MyFrozenLakeEnv(
        render_mode='human',
        is_slippery=True,
        map_name='4x4',
        # desc=["FSFF", "FHFH", "FFFH", "HFFG"],
        max_episode_length=max_depth)
    env.reset(seed=seed)

    alpha = None

    agent = ChanceMCTS(env,
                       seed=seed,
                       adversarial=env.adversarial,
                       gamma=1,
                       keep_subtree=True,
                       max_depth=max_depth,
                       use_tqdm=True,
                       alpha=alpha)

    env.render()

    done = False
    i = 0
    while not done and env.t < max_depth:
        action = agent.plan(iterations_budget=100)
        obs, _, done, _, _ = env.step(action)
        i += 1
        env.render()
        # agent.reset()
        agent.determinize_chance_node(obs)

    print(env.game_result())
    env.close()


if __name__ == '__main__':
    for s in range(10):
        main(s)
   