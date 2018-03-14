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


class PolicyNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = 0.99
        self.learning_rate = 0.01

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):

        # Placeholder for inputs (states)
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")

        # Hidden layer (l1)
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # Linear Layer (l2)
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        # softmax converts to probability
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

    # Chose an action according to network probability
    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def run_episode(self):
        state = env.reset()
        env.render()
        action = self.choose_action(np.array(state))
        done = False

        while not done:
            state, reward, done, inf = env.step(action)
            #env.render()
            action = self.choose_action(state)
            self.store_transition(state,action,reward)

    def discounted_return(self,t):
        d_return = 0
        for i in range(t,len(self.ep_obs)):
            d_return += (self.gamma**(i-t))*self.ep_rs[i]

    def train(self):
        onehot_labels = tf.one_hot(indices=tf.cast(self.ep_obs, tf.int32), depth=3)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=onehot_labels, logits=self.ep_rs)
        tf.train.AdamOptimizer(self.learning_rate).minimize(loss)


def main():

    agent = PolicyNetwork(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
    )

    agent.run_episode()

    sliding_avg = []

    for _ in range(10):

        reward_list = []

        for i_episode in range(10):
            reward_sum = 0
            state = env.reset()
            env.render()
            action = agent.choose_action(np.array(state))
            done = False

            while not done:
                state, reward, done, inf = env.step(action)
                env.render()
                action = agent.choose_action(state)
                reward_sum += reward

            reward_list.append(reward_sum)

        print(reward_list)
        sliding_avg.append(np.mean(reward_list))

    plt.scatter(range(10),sliding_avg)
    plt.title('sliding average for exercize 1')
    plt.ylim([0,50])
    plt.show()


if __name__ == "__main__":

    main()
