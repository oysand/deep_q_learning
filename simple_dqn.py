import os
import tensorflow as tf
import numpy as np

class DeepQNetwork(object):
    def __init__(self, lr, n_actions, name, input_dims, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/dqn'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_dir = chkpt_dir
        self.input_dims = input_dims
        self.sess = tf.Session()
        self.build_network()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, 'deepqnet.ckpt')

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims],
                                        name='inputs')
            self.actions = tf.placeholder(tf.float32,
                                        shape=[None, self.n_actions],
                                        name='actions')
            self.q_target = tf.placeholder(tf.float32,
                                            shape=[None, self.n_actions],
                                            name='q_value')
        
            flat = tf.layers.flatten()
            dense1 = tf.layers.dense(flat, units=self.fc1_dims
                                    activation=tf.nn.relu)
            dense2 = tf.layers.dense(dense1, units=self.fc2_dims,
                                    activation=tf.nn.relu)
            self.Q_values = tf.layers.dense(dense2, self.n_actions)
            self.loss = tf.reduce_mean(tf.square(self.Q_values-self.q_target))
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def load_checkpoint(self):
        print('..... loading checkpoint .....')
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        self.saver.save(self.sess, self.checkpoint_file)

    