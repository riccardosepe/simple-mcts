import gym
from gym import spaces

MAP = {0: ' ', 1: 'O', 2: 'X'}
NUM_LOC = 9
N = 3
O_REWARD = 1
X_REWARD = -1
NO_REWARD = 0


def tocode(mark):
    return 1 if mark == 'O' else 2

def next_mark(mark):
    return 'X' if mark == 'O' else 'O'

def check_game_status(board):
    """Return game status by current board status.

    Args:
        board (list): Current board state

    Returns:
        int:
            -1: game in progress
            0: draw game,
            1 or 2 for finished game(winner mark code).
    """
    for t in [1, 2]:
        for j in range(0, 9, 3):
            if [t] * 3 == [board[i] for i in range(j, j+3)]:
                return t
        for j in range(0, 3):
            if board[j] == t and board[j+3] == t and board[j+6] == t:
                return t
        if board[0] == t and board[4] == t and board[8] == t:
            return t
        if board[2] == t and board[4] == t and board[6] == t:
            return t

    for i in range(9):
        if board[i] == 0:
            # still playing
            return -1

    # draw game
    return 0


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed=None):
        self.action_space = spaces.Discrete(NUM_LOC)
        self.observation_space = spaces.Discrete(NUM_LOC)
        self.start_mark = 'O'
        self.reset(seed=seed)
        self.mark = None
        self.board = None
        self.done = False
        self.last_action = None

    def reset(self, **kwargs):
        self.board = [0] * NUM_LOC
        self.mark = self.start_mark
        self.done = False
        return self.observation

    def reward(self):
        reward = NO_REWARD
        status = check_game_status(self.board)

        if status >= 0:
            self.done = True
            if status in [1, 2]:
                # always called by self
                reward = O_REWARD if self.mark == 'O' else X_REWARD

        return reward

    def step(self, action):
        """Step environment by action.

        Args:
            action (int): Location

        Returns:
            list: Observation
            int: Reward
            bool: Done
            dict: Additional information
        """
        assert self.action_space.contains(action)
        self.last_action = action

        if self.done:
            return self.observation, 0, True, None

        # place
        self.board[action] = tocode(self.mark)

        # calculate reward
        reward = self.reward()

        # switch turn
        self.mark = next_mark(self.mark)
        return self.observation, reward, self.done, None

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            self._show_board(print)  # NOQA
            print('')
        else:
            raise RuntimeError

    def _show_board(self, print_fn):
        for i in range(N):
            for j in range(N):
                print_fn(MAP[self.board[3*i+j]], end='')

                if j < N-1:
                    print_fn('|', end='')
                else:
                    print_fn('')
            if i < N-1:
                print('-----')

    def backup(self):
        state = {
            'board': self.board.copy(),
            'mark': self.mark,
            'done': self.done,
            'last_action': self.last_action,
            'reward': self.reward(),
        }
        return state

    def load(self, checkpoint):
        try:
            self.board = checkpoint['board']
            self.mark = checkpoint['mark']
            self.done = checkpoint['done']
            self.last_action = checkpoint['last_action']
        except KeyError:
            return False
        return True

    @property
    def observation(self):
        return tuple(self.board), self.mark

    @property
    def legal_actions(self):
        return [i for i, c in enumerate(self.board) if c == 0]

