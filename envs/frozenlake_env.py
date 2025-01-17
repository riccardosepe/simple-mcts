from copy import deepcopy

from gymnasium.envs.toy_text import FrozenLakeEnv

from envs.base_env import BaseEnv


class MyFrozenLakeEnv(BaseEnv, FrozenLakeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_reward = None
        self.done = False
        self.lastaction = None

    def reset(self, *args, **kwargs):
        self._last_reward = None
        self.done = False
        return super().reset(*args, **kwargs)

    @property
    def legal_actions(self):
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
    def _last_action(self):
        return self.lastaction

    def step(self, a):
        self.render_mode = ''
        s, r, d, t, i = super().step(a)
        self.render_mode = 'human'
        if d and r==0:
            r = -1.
        self._last_reward = r
        self.done = d
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
            'player': 'Agent'
        }
        return checkpoint

    def load(self, checkpoint):
        self.s = checkpoint['state']
        self.lastaction = checkpoint['last_action']
        self._last_reward = checkpoint['reward']
        self.done = checkpoint['done']

    def game_result(self):
        if not self.done:
            return "Game still running"
        else:
            if self.desc.flatten()[self.s] == b'G':
                return "You made it!"
            else:
                return "You fell into an ice pit :("

    def reward(self):
        return self._last_reward