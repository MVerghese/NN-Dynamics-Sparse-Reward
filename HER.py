from collections import deque
import numpy as np
import copy


class HER:
    def __init__(self):
        self.buffer = deque()

    def reset(self):
        self.buffer = deque()

    def keep(self, item):
        self.buffer.append(item)

    def backward(self, observations, rewards, next_observations):

        new_observations, new_rewards, new_next_observations = copy.deepcopy(observations), copy.deepcopy(rewards), copy.deepcopy(next_observations)
        num = len(new_observations)
        goal = [next_observations[-1][0] + next_observations[-1][2], next_observations[-1][1] + next_observations[-1][3]]
        for i in range(num):
            new_rewards[-1 - i] = 0
            new_next_observations[-1 - i][:2] = goal
            new_observations[-1 - i][:2] = goal
            if (np.sum(np.abs((new_observations[-1 - i][:2] - goal))) < 0.01):
                new_rewards[-1 - i] = 1
        return new_observations, new_rewards, new_next_observations