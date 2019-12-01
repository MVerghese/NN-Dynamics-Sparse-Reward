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
            new_next_observations_current_state = [new_next_observations[-1-i][0] + new_next_observations[-1-i][2],
                                                   new_next_observations[-1-i][1] + new_next_observations[-1-i][3]]
            new_next_observations[-1 - i][:2] = goal
            new_next_observations[-1-i][2:4] = np.ndarray.tolist(
                np.array(new_next_observations_current_state) - np.array(goal))
            new_observations_current_state = [new_observations[-1-i][0] + new_observations[-1-i][2],
                                              new_observations[-1-i][1] + new_observations[-1-i][3]]
            new_observations[-1 - i][:2] = goal
            new_observations[-1 - i][2:4] = np.ndarray.tolist(
                np.array(new_observations_current_state) - np.array(goal))
            if (np.sum(np.abs(([new_next_observations[-1-i][0] + new_next_observations[-1-i][2] - goal[0],
                                new_next_observations[-1-i][1] + new_next_observations[-1-i][3] - goal[1]]))) == 0):
                new_rewards[-1 - i] = 1
        return new_observations, new_rewards, new_next_observations

    def backward_threaded(self, observations, rewards, next_observations):

        new_observations, new_rewards, new_next_observations = copy.deepcopy(observations), copy.deepcopy(rewards), copy.deepcopy(next_observations)
        num = len(new_observations)
        for i in range(num):
            goal = [next_observations[i][0,-1] + next_observations[i][2,-1],
                    next_observations[i][1,-1] + next_observations[i][3,-1]]
            for j in range(len(new_observations[i])):
                new_rewards[i][-1 - j] = 0
                new_next_observations[i][:2,(-1 - j)] = goal
                new_observations[i][:2,(-1 - j)] = goal
                if (np.sum(np.abs((new_observations[i][:2,(-1 - j)] - goal))) < 0.01):
                    new_rewards[i][-1 - j] = 1
        return new_observations, new_rewards, new_next_observations