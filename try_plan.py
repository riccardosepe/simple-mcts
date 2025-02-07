from envs.frozenlake_env import MyFrozenLakeEnv

SEED = 0


def main():
    max_ep_length = 100
    num_iterations = 1000
    env = MyFrozenLakeEnv(render_mode=None, is_slippery=True, max_episode_length=max_ep_length)

    plans = {
        0: [0, 4, 8, 9, 13, 14],
        1: [0, 4, 8, 9, 10, 14],
        2: [0, 1, 2, 6, 10, 14],
        3: [0, 1, 2, 6, 10, 14],
    }

    action_tables = {
        0: {
            0: 1,
            1: 2,
            2: 1,
            3: 0,
            4: 1,
            6: 1,
            8: 2,
            9: 1,
            10: 1,
            13: 2,
            14: 2,
        },
        1: {
            0: 1,
            1: 2,
            2: 1,
            3: 0,
            4: 1,
            6: 1,
            8: 2,
            9: 2,
            10: 1,
            13: 2,
            14: 2,
        },
        2: {
            0: 2,
            1: 2,
            2: 1,
            3: 0,
            4: 1,
            6: 1,
            8: 2,
            9: 0,
            10: 1,
            13: 2,
            14: 2,
        },
        3: {
            0: 2,
            1: 2,
            2: 1,
            3: 0,
            4: 1,
            6: 1,
            8: 2,
            9: 1,
            10: 1,
            13: 2,
            14: 2,
        },
    }

    checkpoint = {
        'state': None,
        'last_action': None,
        'reward': 0,
        'done': False,
        't': 0
    }

    results = dict.fromkeys(plans.keys(), None)

    for plan in action_tables:
        trunc = 0
        success = 0
        hole = 0
        tot_length = 0
        for cell in plans[plan]:
            for _ in range(num_iterations):
                checkpoint['state'] = cell
                env.reset()
                env.load(checkpoint)
                obs = cell
                done = False
                while not done and env.t < max_ep_length:
                    action = action_tables[plan][obs]
                    obs, reward, done, _, _ = env.step(action)

                if not done:
                    # episode was truncated
                    trunc += 1
                else:
                    if reward == 1:
                        success += 1
                    else:
                        hole += 1
                tot_length += env.t

        results[plan] = (success / (num_iterations * len(plans[plan])),
                         hole / (num_iterations * len(plans[plan])),
                         trunc / (num_iterations * len(plans[plan])),
                         tot_length / (num_iterations * len(plans[plan])))

    for p, result in results.items():
        s, h, t, l = result
        print(f"Plan {p}: success {s*100}% holes {h*100}% trunc {t*100}% avg_ep_length {l}")
    
if __name__ == '__main__':
    main()
   