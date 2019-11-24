from collections import deque
import numpy as np
import copy


class HER:
    def __init__(self, N):
        self.buffer = deque()
        self.N = N

    def reset(self):
        self.buffer = deque()

    def keep(self, item):
        self.buffer.append(item)

    def backward(self):

        new_buffer = copy.deepcopy(self.buffer)
        num = len(new_buffer)
        goal = [self.buffer[-1][-2][0] + self.buffer[-1][-2][2], self.buffer[-1][-2][1] + self.buffer[-1][-2][3]]
        for i in range(num):
            new_buffer[-1 - i][2] = 0
            new_buffer[-1 - i][-2][:2] = goal
            new_buffer[-1 - i][0][:2] = goal
            new_buffer[-1 - i][4] = False
            if (np.sum(np.abs((new_buffer[-1 - i][-2][:2] - goal))) < 0.01):
                new_buffer[-1 - i][2] = 1
                new_buffer[-1 - i][4] = True
        return new_buffer