import wandb

from envs.frozenlake_env import MyFrozenLakeEnv
from src.ai.chance_mcts import ChanceMCTS
from src.constants import PROJECT_NAME, ENTITY_NAME


def run(seed, conf):
    conf['seed'] = seed
    is_slippery = conf.get('is_slippery')
    evaluation_method = conf.get('evaluation_method')
    max_depth = conf.get('max_depth')
    iterations_budget = conf.get('iterations_budget')

    env = MyFrozenLakeEnv(is_slippery=is_slippery)
    env.reset(seed=seed)

    agent = ChanceMCTS(env,
                       seed=seed,
                       adversarial=False,
                       evaluation_method=evaluation_method,
                       keep_subtree=False,
                       max_depth=max_depth)

    done = False
    while not done:
        action = agent.plan(iterations_budget=iterations_budget)
        _, _, done, _, _ = env.step(action)

    env.close()


def main():
    conf = {
        'is_slippery': False,
        'evaluation_method': 'rand',
        'iterations_budget': 100,
        'max_depth': 100,
        'num_seeds': 50
    }
    wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY_NAME,
        config=conf,
    )

    num_seeds = conf['num_seeds']
    for s in range(num_seeds):
        run(seed=s, conf=conf)
    
if __name__ == '__main__':
    main()
   