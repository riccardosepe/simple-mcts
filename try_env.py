from envs.frozenlake_env import MyFrozenLakeEnv
from src.evaluators.frozen_lake_evaluator import FrozenLakeEvaluator

SEED = 0


def main():
    env = MyFrozenLakeEnv(render_mode='human', is_slippery=False)

    evaluator = FrozenLakeEvaluator(env.desc, alpha = 0.3)

    obs, _ = env.reset()
    print("Score: ", evaluator.evaluate(obs))
    env.render()

    done = False

    while not done:
        action = int(input("Insert an action: "))
        obs, r, done, _, _ = env.step(action)

        print("Got reward ", r)
        print("Score: ", evaluator.evaluate(obs))

        env.render()


    print(env.game_result())

    
if __name__ == '__main__':
    main()
   