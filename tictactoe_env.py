from itertools import permutations

from gymnasium import spaces

from base_env import BaseEnv


class TicTacToeEnv(BaseEnv):
    metadata = {'render.modes': ['human']}
    _agent_mark = 'X'
    _human_mark = 'O'
    _smart_board = [1, 6, 5, 8, 4, 0, 3, 2, 7]
    _size = 9
    _board_size = 3

    @staticmethod
    def next_mark(mark):
        return TicTacToeEnv._agent_mark if mark is TicTacToeEnv._human_mark else TicTacToeEnv._human_mark

    @staticmethod
    def won(cells):
        return any(sum(h) == 12 for h in permutations(cells, 3))

    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        self.action_space = spaces.Discrete(self._size)
        self.observation_space = spaces.Discrete(self._size)

        self.board = None
        self.done = False
        self.last_action = None
        self.mark = None

    def __str__(self):
        s = ''
        for i in range(self._board_size):
            for j in range(self._board_size):
                v = self.board[3*i + j]
                s += str(v) if v != 0 else ' '
                if j < self._board_size - 1:
                    s += '|'

            s += '\n'

            if i < self._board_size - 1:
                s += '-' * (2 * self._board_size - 1)
                s += '\n'

        return s

    @property
    def observation(self):
        return tuple(self.board), self.mark

    @property
    def legal_actions(self):
        return [i for i, c in enumerate(self.board) if c == 0]

    def reset(self, **kwargs):
        if 'seed' in kwargs:
            super(TicTacToeEnv, self).reset(seed=kwargs['seed'])
        assert 'human_first' in kwargs
        human_first = kwargs['human_first']

        self.board = [0] * self._size
        self.done = False
        self.last_action = None

        # The idea here is to know who is the first player to place a piece on the board. If the first player is human,
        # the first symbol is going to be `O`

        if human_first:
            self.mark = self._human_mark
        else:
            self.mark = self._agent_mark


        return self.observation

    def reward(self):
        agent_places = {self._smart_board[i] for i in range(self._size) if self.board[i] == self._agent_mark}
        human_places = {self._smart_board[i] for i in range(self._size) if self.board[i] == self._human_mark}

        if self.won(agent_places):
            return 1
        elif self.won(human_places):
            return -1
        else:
            return 0

    def step(self, action, human=False):
        assert self.action_space.contains(action)
        self.last_action = action

        assert not self.done

        # place piece on the board
        self.board[action] = self.mark

        reward = self.reward()
        if reward != 0 or self.draw():
            self.done = True

        self.mark = self.next_mark(self.mark)
        return self.observation, reward, self.done, None, {}

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            print(self)
        else:
            raise RuntimeError

    def backup(self):
        state = {
            'board': self.board.copy(),
            'mark': self.mark,
            'done': self.done,
            'last_action': self.last_action,
            'reward': self.reward(),
            'player': 'Human' if self.mark == self._human_mark else 'Agent',
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

    def draw(self):
        try:
            self.board.index(0)
        except ValueError:
            return True

        return False

    def game_result(self):
        if self.done:
            if self.draw():
                return "Draw"
            else:
                return f"{self.next_mark(self.mark)} won"
        else:
            return "Game still running"



def main():
    env = TicTacToeEnv()
    env.reset(human_first=True)
    env.render()

    done = False
    while not done:
        action = int(input('Action: '))
        obs, rew, done, _, _ = env.step(action)
        env.render()

    print(env.game_result())


if __name__ == '__main__':
    main()
