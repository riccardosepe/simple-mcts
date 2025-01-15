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
        return list(range(4))

    @property
    def _last_action(self):
        return self.lastaction

    def step(self, a):
        s, r, d, t, i = super().step(a)
        self._last_reward = r
        self.done = d
        return s, r, d, t, i

    @property
    def adversarial(self):
        return False

    def backup(self):
        checkpoint = {
            'state': self.s,
            'last_action': self._last_action,
            'done': self.done,
            'reward': self.reward(),
            'player': 'Agent'
        }

    def load(self, checkpoint):
        try:
            self.s = checkpoint['state']
            self.lastaction = checkpoint['last_action']
            self._last_reward = checkpoint['reward']
            self.done = checkpoint['done']
        except KeyError:
            return False
        return True

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