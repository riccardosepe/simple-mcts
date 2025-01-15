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
        #     - done
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
        """
        This function is supposed to return a reward associated with the current state. However,
        it is not always true that r = r(s'). sometimes, r = r(a, s') or even r = r(s, a, s'). How do we handle these
        scenarios?. The easiest solution is to edit the step function such that the last reward returned by it is also
        internally stored. An alternative would be to store the previous state, in addition to the last performed
        action. However, since in both cases we need to store an additional piece of information, it's more convenient
        to just store the reward.
        """
        pass