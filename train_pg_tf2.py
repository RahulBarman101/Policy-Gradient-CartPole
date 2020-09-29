import tensorflow as tf
import gym
import numpy as np
from policy_gradient_tf2 import Agent
import time

env = gym.make('CartPole-v0')
agent = Agent()
num_episodes = 10000

for i in range(num_episodes):
    state = env.reset()
    score = 0
    rewards = []
    states = []
    actions = []
    done = False
    while not done:
        action = agent.choose_action(state)
        state_,reward,done,_ = env.step(action)
        agent.store_reward(reward)
        agent.store_state(state)
        state = state_
        score += reward
        # env.render()
        if done:
            agent.learn()
            print(f'episode done: {i+1}\t score recieved: {score}')


    