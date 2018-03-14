from PolicyNetwork_inc import PolicyNetwork
import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt

# reproducible
np.random.seed(1)
tf.set_random_seed(1)

env = gym.make('CartPole-v0')
env.seed(1)

RENDER = False

agent = PolicyNetwork(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
    )

