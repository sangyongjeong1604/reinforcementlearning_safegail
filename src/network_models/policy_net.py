import tensorflow as tf


class Policy_net:
    def __init__(self, name: str, observation_space):
        """
        :param name: string
        """

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(observation_space), name='obs')

            with tf.variable_scope('policy_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=20, activation=tf.tanh)
                self.act_probs = tf.layers.dense(inputs=layer_3, units=1, activation=tf.tanh)

            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)

            #self.act_probs = tf.clip_by_value(self.act_probs, -0.22, 0.22)
            #self.act_probs = tf.clip_by_value(self.act_probs, -5, 5)
            self.act_stochastic = tf.reshape(self.act_probs, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, sess, obs, stochastic=True):
        if stochastic:
            return sess.run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return sess.run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, sess, obs):
        return sess.run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

