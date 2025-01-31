from copy import deepcopy

from gymnasium.envs.toy_text import FrozenLakeEnv

from envs.base_env import BaseEnv
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class MyFrozenLakeEnv(BaseEnv, FrozenLakeEnv):
    def __init__(self, *args, p=1/3, **kwargs):
        self.max_episode_length = kwargs.pop('max_episode_length', 1000)
        super().__init__(*args, **kwargs)
        self._last_reward = None
        self.done = False
        self.lastaction = None
        self.is_slippery = kwargs.get('is_slippery', False)
        self.t = 0
        ps = [(1-p)/2, p, (1-p)/2]

        nA = 4
        nS = self.nrow * self.ncol

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * self.ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, self.nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, self.ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            new_row, new_col = inc(row, col, action)
            new_state = to_s(new_row, new_col)
            new_letter = self.desc[new_row, new_col]
            terminated = bytes(new_letter) in b"GH"
            reward = float(new_letter == b"G")
            return new_state, reward, terminated

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = self.desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if self.is_slippery:
                            for k, b in enumerate([(a - 1) % 4, a, (a + 1) % 4]):
                                li.append(
                                    (ps[k], *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))


    def reset(self, *args, **kwargs):
        self._last_reward = None
        self.done = False
        self.t = 0
        return super().reset(*args, **kwargs)

    @property
    def legal_actions_old(self):
        # NB:
        # - 0: Move left
        # - 1: Move down
        # - 2: Move right
        # - 3: Move up
        i, j = self.s // self.ncol, self.s % self.ncol
        actions = []
        if i > 0:
            actions.append(3)
        if i < self.nrow - 1:
            actions.append(1)
        if j > 0:
            actions.append(0)
        if j < self.ncol - 1:
            actions.append(2)
        return actions

    @property
    def legal_actions(self):
        return list(range(4))

    @property
    def _last_action(self):
        return self.lastaction

    def step(self, a):
        self.render_mode = ''
        s, r, d, t, i = super().step(a)
        self.render_mode = 'human'
        self._last_reward = r
        self.done = d
        self.t += 1
        return s, r, d, t, i

    @property
    def adversarial(self):
        return False

    def backup(self):
        checkpoint = {
            'state': deepcopy(self.s),
            'last_action': self._last_action,
            'done': self.done,
            'reward': self.reward(),
            'player': 'Agent',
            't': self.t
        }
        return checkpoint

    def load(self, checkpoint):
        self.s = checkpoint['state']
        self.lastaction = checkpoint['last_action']
        self._last_reward = checkpoint['reward']
        self.done = checkpoint['done']
        self.t = checkpoint['t']

    def game_result(self):
        if not self.done:
            return "Game still running"
        else:
            if self.desc.flatten()[self.s] == b'G':
                return f"You made it after {self.t} steps!"
            else:
                return f"You fell into an ice pit after {self.t} steps :("

    def reward(self):
        return self._last_reward

    def next_states(self, action):
        # TODO: is this logic ok here?
        i, j = self.s // self.ncol, self.s % self.ncol

        def to_s(ii, jj):
            return ii*self.ncol + jj

        states = {
            1: to_s(max(i - 1, 0), j),
            3: to_s(min(i + 1, self.nrow - 1), j),
            2: to_s(i, max(j - 1, 0)),
            0: to_s(i, min(j + 1, self.ncol - 1)),
        }
        if self.is_slippery:
            del states[action]
            return list(states.values())

        else:
            return [states[(action+2)%4]]

