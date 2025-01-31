from collections import deque

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline


class FrozenLakeEvaluator:
    def __init__(self, board, alpha=0.5):
        self.board = board
        self.nrows, self.ncols = board.shape
        self._max_distance = self._manhattan([0, 0], [self.nrows-1, self.ncols-1])
        goal_pos = np.argwhere(self.board == b'G')
        assert len(goal_pos) == 1
        self.goal_pos = goal_pos[0]
        self.alpha = alpha
        self._landscape = self._build_landscape()

    @staticmethod
    def _manhattan(p1, p2):
        """
        Compute the Manhattan distance between two points in the grid
        """
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _pos_to_indices(self, pos):
        return pos // self.ncols, pos % self.ncols

    def _distance(self, obs):
        """
        This function returns 1-M, where M is the normalized manhattan distance between the agent and the goal positions
        """
        agent_pos = np.array([*self._pos_to_indices(obs)])

        agent_goal_distance = self._manhattan(agent_pos, self.goal_pos)

        return (self._max_distance - agent_goal_distance) / self._max_distance

    def _build_landscape(self):
        rows, cols = self.board.shape
        queue = deque()
        landscape = np.zeros_like(self.board, dtype=int)

        for i, j in np.argwhere(self.board == b'H'):
            queue.append((i, j, 0))
            visited = {(ii, jj) for ii, jj in np.argwhere(self.board == b'H')}

            while queue:
                x, y, depth = queue.popleft()

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                        landscape[nx][ny] -= (depth + 1)
                        queue.append((nx, ny, depth + 1))
                        visited.add((nx, ny))

        return (landscape - np.min(landscape)) / -np.min(landscape)


    def _safety(self, obs):
        """
        This function returns how safe a square is, i.e. how many holes are in the neighborhood of the agent.
        Moreover, if the agent is on the goal, safety = 1, if the agent is in an ice pit, safety = 0.
        """
        h = 0
        n = 0

        x, y = self._pos_to_indices(obs)

        if self.board[x, y] == b'G':
            return 1

        if self.board[x, y] == b'H':
            return 0

        for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if 0 <= x + i < self.nrows and 0 <= y + j < self.ncols:
                c = self.board[x + i, y + j]
                n += 1
                if c == b'H':
                    h += 1

        return (n - h) / n


    def evaluate(self, obs):
        return self.alpha * self._distance(obs) + (1 - self.alpha) * self._safety(obs)


    def _visualize_landscape(self):
        xpos, ypos = np.meshgrid(np.arange(self.ncols), np.arange(self.nrows), indexing='ij')
        xpos = xpos.flatten()
        ypos = ypos.flatten()

        zpos = np.zeros_like(xpos)
        heights = self._landscape.flatten()

        dx = dy = 1

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.bar3d(xpos, ypos, zpos, dx, dy, heights, shade=True)

        interp_spline = RectBivariateSpline(np.arange(self.ncols), np.arange(self.nrows), self._landscape)

        off = -0.5

        # High-resolution grid (interpolated)
        x_high = np.linspace(0 + off, self.ncols + off, 100)  # More points along X
        y_high = np.linspace(0 + off, self.nrows + off, 100)  # More points along Y
        X_high, Y_high = np.meshgrid(x_high, y_high, indexing="ij")

        # Evaluate the interpolated heights
        Z_high = interp_spline(y_high, x_high)
        Z_high[Z_high > np.max(self._landscape)] = np.max(self._landscape)

        # Create 3D plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xticks(np.arange(self.ncols + 1))
        ax.set_yticks(np.arange(self.nrows + 1))

        ax.plot_surface(X_high, Y_high, Z_high, cmap='viridis', edgecolor='none')

        print(interp_spline(np.arange(self.ncols), np.arange(self.nrows)))
        plt.show()