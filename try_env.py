import gymnasium as gym


SEED = 0


def main():
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human')

    env.reset()
    env.render()

    done = False

    while not done:
        action = int(input("Insert an action: "))
        obs, _, done, _, _ = env.step(action)

        env.render()


    # print(env.game_result())

    
if __name__ == '__main__':
    main()
   