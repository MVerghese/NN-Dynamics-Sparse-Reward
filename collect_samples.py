import numpy as np
import rllab
import time
import matplotlib.pyplot as plt
import copy
from collections import deque
from HER import HER
from copy import deepcopy as dc

class CollectSamples(object):

    def __init__(self, env, policy, visualize_rollouts, which_agent, dt_steps, dt_from_xml, follow_trajectories):
        self.env = env
        self.policy = policy
        self.visualize_at_all = visualize_rollouts
        self.which_agent = which_agent

        self.low = self.env.observation_space.low
        self.high = self.env.observation_space.high
        self.shape = self.env.observation_space.shape

        self.use_low = self.low + (self.high-self.low)/3.0
        self.use_high = self.high - (self.high-self.low)/3.0

        self.dt_steps = dt_steps
        self.dt_from_xml = dt_from_xml

        self.follow_trajectories = follow_trajectories
        self.her = HER()
        
    def collect_samples(self, num_rollouts, steps_per_rollout):
        observations_list = []
        actions_list = []
        replay_observations = []
        replay_actions = []
        replay_rewards = []
        starting_states_list=[]
        visualization_frequency = 10
        for rollout_number in range(num_rollouts):
            if(self.which_agent==2):
                if(self.follow_trajectories):
                    observation, starting_state = self.env.reset(returnStartState=True, isSwimmer=True, need_diff_headings=True)
                else:
                    observation, starting_state = self.env.reset(returnStartState=True, isSwimmer=True)
            elif(self.which_agent==3):
                observation = self.env.reset()
                starting_state = observation
            else:
                observation, starting_state = self.env.reset(returnStartState=True)
            observations, actions, rewards, next_observations = self.perform_rollout(observation, steps_per_rollout,
                                                                        rollout_number, visualization_frequency)
            observations = np.array(observations)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_observations = np.array(next_observations)

            new_observations, new_rewards, new_next_observations = self.her.backward(observations, rewards,
                                                                                     next_observations)
            observations_list.append(observations)
            actions_list.append(actions)
            starting_states_list.append(starting_state)
            replay_observations.append(observations)
            replay_observations.append(new_observations)
            replay_actions.append(actions)
            replay_actions.append(actions)
            replay_rewards.append(rewards)
            replay_rewards.append(new_rewards)

        #return list of length = num rollouts
        #each entry of that list contains one rollout
        #each entry is [steps_per_rollout x statespace_dim] or [steps_per_rollout x actionspace_dim]
        return observations_list, actions_list, starting_states_list, replay_observations, replay_actions, replay_rewards

    def perform_rollout(self, observation, steps_per_rollout, rollout_number, visualization_frequency):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        visualize = False
        if((rollout_number%visualization_frequency)==0):
            print("currently performing rollout #", rollout_number)
            if(self.visualize_at_all):
                all_states=[]
                print ("---- visualizing a rollout ----")
                visualize=True

        for step_num in range(steps_per_rollout):
            action, _ = self.policy.get_action(observation)

            observations.append(observation)
            actions.append(action)

            if(self.which_agent==3):
                next_observation, reward, terminal, _ = self.env.step(action)
            else:
                next_observation, reward, terminal, _ = self.env.step(action, collectingInitialData=True)

            rewards.append(reward)
            next_observations.append(next_observation)
            if (terminal == True):
                break
            observation = np.copy(next_observation)
            
            if terminal:
                print("Had to stop rollout because terminal state was reached.")
                break

            if(visualize):
                if(self.which_agent==0):
                    curr_state = self.env.render()
                    all_states.append(np.expand_dims(curr_state, axis=0))
                else:
                    self.env.render()
                    time.sleep(self.dt_steps*self.dt_from_xml)

        if(visualize and (self.which_agent==0)):
            all_states= np.concatenate(all_states, axis=0)
            plt.plot(all_states[:,0], all_states[:,1], 'r')
            plt.show()

        return observations, actions, rewards, next_observations