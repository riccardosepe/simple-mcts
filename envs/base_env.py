from abc import ABC, abstractmethod


class BaseEnv(ABC):
    @property
    @abstractmethod
    def legal_actions(self):
        pass

    @property
    @abstractmethod
    def _last_action(self):
        pass

    @property
    @abstractmethod
    def adversarial(self):
        pass

    @abstractmethod
    def backup(self):
        # NB the data from the backup function can divide in two different categories
        # 1. the data related to the actual state of the game
        #     - board (remember to take a deepcopy or something)
        #     - mark (whose turn it is)
        #     - done
        #     - last_action
        # 2. the metadata used by MCTS
        #     - reward
        #     - player
        #
        # In particular, the first category can be whatever because it is handled internally by the environment, while
        # the second category must be as it is because it's used outside the class
        pass

    @abstractmethod
    def load(self, checkpoint):
        pass

    @abstractmethod
    def game_result(self):
        pass

    @abstractmethod
    def reward(self):
        pass