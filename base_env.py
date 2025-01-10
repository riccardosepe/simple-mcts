from gymnasium import Env
from abc import ABC, abstractmethod


class BaseEnv(ABC, Env):
    @property
    @abstractmethod
    def legal_actions(self):
        pass

    @abstractmethod
    def backup(self):
        pass

    @abstractmethod
    def load(self, checkpoint):
        pass

    @abstractmethod
    def game_result(self):
        pass