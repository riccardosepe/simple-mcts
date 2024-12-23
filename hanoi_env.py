import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box


class TowersOfHanoiEnv(Env):
    metadata = {'render.modes': ['human']}
    action_dict = {
        'AB': 0,
        'AC': 1,
        'BA': 2,
        'BC': 3,
        'CA': 4,
        'CB': 5
    }

    def __init__(self, num_disks=3, num_pegs=3):
        """
        Initialize the Towers of Hanoi environment.

        Parameters:
            num_disks (int): The number of disks in the Towers of Hanoi puzzle.
        """
        super(TowersOfHanoiEnv, self).__init__()
        self.num_disks = num_disks

        # The state is represented as a list of three lists (one for each peg),
        # where each list contains the disks (integers) on that peg in order.
        self.state = None

        self.num_pegs = num_pegs
        # Action space: moving a disk from one peg to another
        # Represented as a tuple (from_peg, to_peg), encoded as a discrete action.
        self.action_space = Discrete(num_pegs * (num_pegs-1))  # 3 pegs -> 3*2 = 6 possible moves

        # Observation space: state of all three pegs
        self.observation_space = Box(
            low=0, high=self.num_disks, shape=(num_pegs, self.num_disks), dtype=np.int32
        )

        self._base = '\t'.join([chr(i) for i in range(ord('A'), ord('A') + self.num_pegs)])

        self.done = False

    def reset(self, **kwargs):
        """
        Reset the environment to the initial state.

        Returns:
            observation (np.ndarray): The initial state of the environment.
        """
        self.state = [list(range(self.num_disks, 0, -1)), [], []]  # All disks on the first peg
        self.done = False
        return self._get_observation()

    def step(self, action):
        """
        Perform an action in the environment.

        Parameters:
            action (int): The action to take, encoded as a single integer.
                - 0: Move from peg 0 to peg 1
                - 1: Move from peg 0 to peg 2
                - 2: Move from peg 1 to peg 0
                - 3: Move from peg 1 to peg 2
                - 4: Move from peg 2 to peg 0
                - 5: Move from peg 2 to peg 1

        Returns:
            tuple: (observation, reward, done, info)
                - observation (np.ndarray): The current state of the environment.
                - reward (float): The reward for the action.
                - done (bool): Whether the puzzle is solved.
                - info (dict): Additional diagnostic information.
        """
        if self.done:
            raise ValueError("Environment has already finished. Call reset() to restart.")

        from_peg = action // 2
        to_peg = action % 2 + (1 if action % 2 >= from_peg else 0)

        if not self.state[from_peg]:
            return self._get_observation(), -1.0, self.done, {"error": "Invalid move"}

        disk = self.state[from_peg][-1]
        if self.state[to_peg] and self.state[to_peg][-1] < disk:
            return self._get_observation(), -1.0, self.done, {"error": "Invalid move"}

        # Perform the move
        self.state[from_peg].pop()
        self.state[to_peg].append(disk)

        # Check if the puzzle is solved
        self.done = len(self.state[2]) == self.num_disks

        reward = 1.0 if self.done else 0.0
        return self._get_observation(), reward, self.done, {}

    def render(self, mode='human'):
        """
        Render the current state of the environment.
        """
        pegs = [[str(disk) for disk in reversed(peg)] for peg in self.state]
        for peg in pegs:
            while len(peg) < self.num_disks:
                peg.insert(0, '|')  # Fill empty spaces with '|'
        for row in range(self.num_disks):
            print('\t'.join(peg[row] for peg in pegs))

        print(self._base)
        print('-' * 10)

    def close(self):
        """
        Close the environment.
        """
        print("Environment closed.")

    def _get_observation(self):
        """
        Return the current state as an observation.

        Returns:
            np.ndarray: The state as a numpy array.
        """
        obs = np.zeros((3, self.num_disks), dtype=np.int32)
        for i, peg in enumerate(self.state):
            for j, disk in enumerate(reversed(peg)):
                obs[i, j] = disk
        return obs


# Example usage
if __name__ == "__main__":
    env = TowersOfHanoiEnv(num_disks=3)
    observation = env.reset()

    while True:
        env.render()
        action = env.action_dict.get(input("Insert an action in the form XY: "), None)
        if action is None:
            break
        env.step(action)



    env.render()
    env.close()
