from envs.frozenlake_env import MyFrozenLakeEnv

SEED = 0


def main():
    env = MyFrozenLakeEnv(render_mode='human', is_slippery=False)

    env.reset()
    env.render()

    done = False

    while not done:
        action = int(input("Insert an action: "))
        obs, r, done, _, _ = env.step(action)

        print("Got reward ", r)

        env.render()


    print(env.game_result())

    
if __name__ == '__main__':
    main()
   