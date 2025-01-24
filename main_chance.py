from envs.frozenlake_env import MyFrozenLakeEnv
from src.ai.chance_mcts import ChanceMCTS

HUMAN = True
BOT = False
SEED = 0


def main():
    env = MyFrozenLakeEnv(render_mode='human', is_slippery=True, map_name='4x4')
    env.reset(seed=SEED)

    agent = ChanceMCTS(env, seed=SEED, adversarial=env.adversarial, gamma=1, keep_subtree=True, max_depth=50)

    env.render()

    done = False
    i = 0
    while not done:
        action = agent.plan(iterations_budget=10000)
        obs, _, done, _, _ = env.step(action)
        agent.determinize_chance_node(obs)  # TODO: hash states
        i += 1

        env.render()

    print(env.game_result())
    env.close()
    
if __name__ == '__main__':
    main()
   