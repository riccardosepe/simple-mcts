from tqdm import tqdm

from envs.frozenlake_env import MyFrozenLakeEnv
from src.ai.chance_mcts import ChanceMCTS


def run(seed, conf):
    is_slippery = conf.get('is_slippery')
    max_depth = conf.get('max_depth')
    iterations_budget = conf.get('iterations_budget')
    # group = conf.get('group')
    alpha = conf.get('alpha')

    # group = group + f"_seed-{seed}"

    env = MyFrozenLakeEnv(is_slippery=is_slippery)
    env.reset(seed=seed)

    agent = ChanceMCTS(env,
                       seed=seed,
                       adversarial=False,
                       keep_subtree=False,
                       max_depth=max_depth,
                       alpha=alpha,)

    done = False
    reward = 0
    trunc = False
    hole = False
    while not done and env.t <= max_depth:
        action = agent.plan(iterations_budget=iterations_budget)
        _, reward, done, trunc, _ = env.step(action)

    if not done and env.t >= max_depth and reward == 0:
        trunc = True

    if done and env.t < max_depth and reward == 0:
        hole = True
    env.close()
    # return reward (also success, since it's boolean), episode length, whether the episode was truncated and whether the agent fell in a hole
    return reward, env.t, trunc, hole


def main():

    # wandb.init(
    #     project=PROJECT_NAME,
    #     entity=ENTITY_NAME,
    #     config=conf,
    #     group=group,
    #     name='rand' if alpha is None else f'alpha-{alpha}',
    # )
    # wandb.finish()
    #

    for alpha in [0.8, 0.85, 0.9, 0.95]:

        conf = {
            'is_slippery': True,
            'iterations_budget': 100,
            'max_depth': 100,
            'num_seeds': 1000,
            'group': 'rand_vs_function',
            'alpha': alpha
        }

        num_seeds = conf['num_seeds']
        cumulative_return = 0
        cumulative_episode_length = 0
        num_truncated = 0
        num_holes = 0
        for s in tqdm(range(num_seeds)):
            ret, ep_length, trunc, hole = run(seed=s, conf=conf)
            cumulative_return += ret
            cumulative_episode_length += ep_length
            num_truncated += trunc
            num_holes += hole

        print(conf)
        print("Average episode return: ", cumulative_return/num_seeds)
        print("Average episode length: ", cumulative_episode_length/num_seeds)
        print("Truncated episodes ratio: ", num_truncated/num_seeds)
        print("Fall in hole ratio: ", num_holes/num_seeds)

    
if __name__ == '__main__':
    main()
   