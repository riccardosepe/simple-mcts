from envs.frozenlake_env import MyFrozenLakeEnv
from src.evaluators.frozen_lake_evaluator import FrozenLakeEvaluator

SEED = 0


def main():
    max_ep_length = 100
    env = MyFrozenLakeEnv(render_mode='human', is_slippery=False, max_episode_length=max_ep_length)

    alpha = 0.5

    evaluator = FrozenLakeEvaluator(env.desc, alpha=alpha, max_episode_length=max_ep_length)

    obs, _ = env.reset()
    print("Score: ", evaluator.evaluate(obs, env.t))
    env.render()

    done = False

    while not done:
        action = int(input("Insert an action: "))
        obs, r, done, _, _ = env.step(action)

        print("Got reward ", r)
        print("Score: ", evaluator.evaluate(obs, env.t))

        env.render()


    print(env.game_result())

    
if __name__ == '__main__':
    main()
   