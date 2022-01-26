import sys
import tensorflow as tf
import numpy as np

sys.path.append('/home/gz67063/projects/sipomdplite-net')
from network.layers import conv_layers, fc_layers


class QMDPNet:
    """
    Class implementing a QMDP-Net for the grid navigation domain
    """

    def __init__(
            self,
            params,
            batch_size=1,
            step_size=1):
        """
        :param params: dotdict describing the domain and network hyperparameters
        :param batch_size: minibatch size for training. Use batch_size=1 for evaluation
        :param step_size: limit the number of steps for backpropagation through time. Use step_size=1 for evaluation.
        """
        self.params = params
        self.batch_size = batch_size
        self.step_size = step_size

        self.placeholders = None
        self.belief = None
        self.update_belief_op = None
        self.pred_action_traj = None
        self.loss = None
        self.decay_step = None
        self.learning_rate = params.learning_rate
        self.train_op = None

    def build_placeholders(self):
        """
        Creates placeholders for all inputs in self.placeholders
        """
        grid_n = self.params.grid_n
        grid_m = self.params.grid_m
        step_size = self.step_size
        batch_size = self.batch_size

        placeholders = list()
        placeholders.append(
            tf.compat.v1.placeholder(
                dtype=tf.int32,
                shape=(batch_size, grid_n, grid_m),
                name='task_param_env_map'))

        placeholders.append(
            tf.compat.v1.placeholder(
                dtype=tf.int32,
                shape=(batch_size, grid_n, grid_m),
                name='task_param_goal_map'))

        placeholders.append(
            tf.compat.v1.placeholder(
                dtype=tf.float32,
                shape=(batch_size, grid_n, grid_m),
                name='task_param_init_belief'))

        placeholders.append(
            tf.compat.v1.placeholder(
                dtype=tf.float32,
                shape=(batch_size,),
                name='is_traj_head'))

        placeholders.append(
            tf.compat.v1.placeholder(
                dtype=tf.int32,
                shape=(step_size, batch_size),
                name='input_action_traj'))

        placeholders.append(
            tf.compat.v1.placeholder(
                dtype=tf.int32,
                shape=(step_size, batch_size),
                name='input_observation_traj'))

        placeholders.append(
            tf.compat.v1.placeholder(
                dtype=tf.float32,
                shape=(step_size, batch_size),
                name='valid_step_mask'))

        placeholders.append(
            tf.compat.v1.placeholder(
                dtype=tf.int32,
                shape=(step_size, batch_size),
                name='label_action_traj'))

        self.placeholders = placeholders

    def build_inference(
            self,
            reuse=False):
        """
        Creates placeholders, ops for inference and loss
        Unfolds filter and planner through time
        Also creates an op to update the belief. It should be always evaluated together with the loss.
        :param reuse: reuse variables if True
        :return: None
        """
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()

        self.build_placeholders()

        env_map, \
            goal_map, \
            init_belief, \
            is_traj_head, \
            input_action_traj, \
            input_observation_traj, \
            valid_step_mask, \
            label_action_traj = self.placeholders

        # types conversions
        env_map = tf.cast(
            env_map,
            dtype=tf.float32)
        goal_map = tf.cast(
            goal_map,
            dtype=tf.float32)
        is_traj_head = tf.reshape(
            is_traj_head,
            [self.batch_size] + [1] * (init_belief.get_shape().ndims - 1))

        pred_action_traj = list()

        # pre-compute context, fixed through time
        with tf.compat.v1.variable_scope("planning_net"):
            q_values, _, _ = PlanningNet.value_iteration(
                env_map=env_map,
                goal_map=goal_map,
                params=self.params)
        with tf.compat.v1.variable_scope("belief_filter_net"):
            observation_function = \
                BeliefFilterNet.get_prior_observation_function(
                    env_map=env_map,
                    params=self.params)

        # create variable for hidden belief (equivalent to the hidden state of an RNN)
        self.belief = tf.compat.v1.get_variable(
            name="hidden_belief",
            initializer=np.zeros(
                init_belief.get_shape().as_list(),
                dtype=np.float32),
            trainable=False)

        # figure out current b. b = b0 if isstart else blast
        belief = (init_belief * is_traj_head) + \
                 (self.belief * (1 - is_traj_head))

        for step in range(self.step_size):
            # Belief filter module
            with tf.compat.v1.variable_scope("belief_filter_net") as step_scope:
                if step >= 1:
                    step_scope.reuse_variables()
                belief = BeliefFilterNet.belief_update(
                    observation_function=observation_function,
                    belief=belief,
                    action=input_action_traj[step],
                    observation=input_observation_traj[step],
                    params=self.params)

            # QMDP planning module
            with tf.compat.v1.variable_scope("planning_net") as step_scope:
                if step >= 1:
                    step_scope.reuse_variables()
                pred_action = PlanningNet.policy(
                    q_values=q_values,
                    belief=belief)
                pred_action_traj.append(pred_action)

        # create op that updates the belief
        self.update_belief_op = self.belief.assign(belief)

        # compute loss (cross-entropy)
        pred_action_traj = tf.stack(
            values=pred_action_traj,
            axis=0)  # shape is [step_size, batch_size, num_action]

        # logits = tf.reshape(logits, [self.step_size*self.batch_size, self.params.num_action])
        # act_label = tf.reshape(act_label, [-1])
        # weight = tf.reshape(weight, [-1])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pred_action_traj,
            labels=label_action_traj)

        # weight loss. weights are 0.0 for steps after the end of a trajectory, otherwise 1.0
        loss = loss * valid_step_mask
        loss = tf.reduce_mean(
            loss,
            axis=[0, 1],
            name='xentropy')

        self.pred_action_traj = pred_action_traj
        self.loss = loss

    def build_train(
            self,
            init_learning_rate):
        """
        """
        # Decay learning rate by manually incrementing decay_step
        decay_step = tf.Variable(0.0, name='decay_step', trainable=False)
        learning_rate = tf.compat.v1.train.exponential_decay(
            learning_rate=init_learning_rate,
            global_step=decay_step,
            decay_steps=1,
            decay_rate=0.8,
            staircase=True,
            name="learning_rate")

        trainable_variables = tf.compat.v1.trainable_variables()

        optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate, decay=0.9)
        # clip gradients
        grads = tf.gradients(self.loss, trainable_variables)
        grads, _ = tf.clip_by_global_norm(
            grads, 1.0, use_norm=tf.compat.v1.global_norm(grads))

        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

        self.decay_step = decay_step
        self.learning_rate = learning_rate
        self.train_op = train_op


class QMDPNetPolicy(object):
    """
    Policy wrapper for QMDPNet. Implements two functions: reset and eval.
    """

    def __init__(
            self,
            network,
            sess):
        self.network = network
        self.sess = sess

        self.belief = None
        self.env_map = None
        self.goal_map = None

        assert self.network.batch_size == 1 and self.network.step_size == 1

    def reset(
            self,
            env_map,
            goal_map,
            belief):
        """
        :param env_map:
        :param goal_map:
        :param belief:
        :return:
        """
        grid_n = self.network.params.grid_n
        grid_m = self.network.params.grid_m

        self.belief = belief.reshape([1, grid_n, grid_m])
        self.env_map = env_map.reshape([1, grid_n, grid_m])
        self.goal_map = goal_map.reshape([1, grid_n, grid_m])

        self.sess.run(
            tf.compat.v1.assign(
                self.network.belief, self.belief))

    def output(
            self,
            input_action,
            input_observation):
        """
        :param input_action:
        :param input_observation:
        :return:
        """
        is_traj_head = np.array([0])
        input_action = input_action.reshape([1, 1])
        input_observation = \
            input_observation.reshape([1, 1])

        # input data. do not neet weight and label for prediction
        data = [self.env_map,
                self.goal_map,
                self.belief,
                is_traj_head,
                input_action,
                input_observation]
        feed_dict = {self.network.placeholders[i]: data[i]
                     for i in range(len(self.network.placeholders) - 2)}

        # evaluate QMDPNet
        pred_action, _, belief = self.sess.run(
            [self.network.pred_action_traj,
             self.network.update_belief_op,
             self.network.belief],
            feed_dict=feed_dict)
        pred_action = pred_action.flatten().argmax()

        return pred_action, belief


class PlanningNet(object):
    @staticmethod
    def get_reward_function(
            env_map,
            goal_map,
            params):
        task_param = tf.stack(
            [env_map, goal_map],
            axis=3)
        reward_function = conv_layers(
            task_param,
            conv_params=np.array(
                [[3, 200, 'relu'],
                 [1, params.num_action, 'lin']]),
            names="reward_func")

        return reward_function

    @staticmethod
    def value_iteration(
            env_map,
            goal_map,
            params):
        """
        builds neural network implementing value iteration.
        this is the first part of planner module. Fixed through time.
        inputs: map (batch x N x N) and goal(batch)
        returns: Q_K, and optionally: R, list of Q_i
        """
        # build reward model R
        reward_function = PlanningNet.get_reward_function(
            env_map=env_map,
            goal_map=goal_map,
            params=params)

        # If not packed with task parameters:
        # (Uncomment to activate)
        transition_function = \
            BeliefFilterNet.get_transition_function4planning(
                name='trans_func',
                params=params)

        # If packed with task parameters:
        # (Uncomment to activate)
        # transition_function = \
        #     BeliefFilterNet.get_transition_function_w_env_cond(
        #         name='trans_func',
        #         params=params)

        # initialize value image
        # If not packed with task parameters:
        # (Uncomment to activate)
        state_utilities = tf.zeros(
            env_map.get_shape().as_list() + [1])

        # If packed with task parameters:
        # (Uncomment to activate)
        # state_utilities = tf.zeros(env_map.get_shape())
        # cond_state_utilities = tf.stack(
        #     [state_utilities, env_map, goal_map],
        #     axis=-1)
        q_values = None

        # repeat value iteration K times
        for i in range(params.K):
            # apply transition and sum
            q_values = tf.nn.conv2d(
                state_utilities,
                transition_function,
                strides=[1, 1, 1, 1],
                padding='SAME')
            q_values += reward_function

            # If not packed with task parameters:
            # (uncomment to activate)
            state_utilities = tf.reduce_max(
                q_values,
                axis=3,
                keepdims=True)

            # If packed with task parameters:
            # (Uncomment to activate)
            # state_utilities = tf.reduce_max(
            #     q_values,
            #     axis=3,
            #     keepdims=False)
            # cond_state_utilities = tf.stack(
            #     [state_utilities, env_map, goal_map],
            #     axis=-1)

        return q_values, state_utilities, reward_function

    @staticmethod
    def policy(q_values, belief):
        """
        second part of planner module
        :param q_values: input Q_K after value iteration
        :param belief: belief at current step
        :return: a_pred,  vector with num_action elements, each has the
        """
        # weight Q by the belief
        belief = tf.expand_dims(
            belief,
            axis=3)
        action_values = tf.reduce_sum(
            tf.math.multiply(
                q_values, belief),
            axis=[1, 2],
            keepdims=False)

        action = tf.math.log(
            tf.nn.softmax(action_values) + 1e-10)

        return action


class BeliefFilterNet(object):
    @staticmethod
    def get_transition_function4filtering(
            name,
            params):
        """

        :param name:
        :param params:
        """
        initializer = tf.compat.v1.truncated_normal_initializer(
            mean=1.0 / 9.0,
            stddev=1.0 / 90.0,
            dtype=tf.float32)
        transition_function = tf.compat.v1.get_variable(
            name=name,
            shape=(3 * 3, params.num_action),
            initializer=initializer,
            dtype=tf.float32)
        transition_function = tf.nn.softmax(
            transition_function,
            axis=0)
        transition_function = tf.reshape(
            transition_function,
            shape=(3, 3, params.num_action, 1))

        return transition_function

    @staticmethod
    def get_transition_function4planning(
            name,
            params):
        initializer = tf.compat.v1.truncated_normal_initializer(
            mean=1.0 / 9.0,
            stddev=1.0 / 90.0,
            dtype=tf.float32)
        transition_function = tf.compat.v1.get_variable(
            name=name,
            shape=(3 * 3, params.num_action),
            initializer=initializer,
            dtype=tf.float32)
        transition_function = tf.nn.softmax(
            transition_function,
            axis=0)
        transition_function = tf.reshape(
            transition_function,
            shape=(3, 3, 1, params.num_action))

        return transition_function

    @staticmethod
    def get_observation_function(
            env_map,
            params):
        """
        :param env_map:
        :param params:
        """
        env_map = tf.expand_dims(
            env_map,
            axis=-1)
        observation_function = conv_layers(
            env_map,
            conv_params=np.array(
                [[3, 200, 'linear'],
                 [1, params.num_action * params.num_obs, 'sigmoid']]),
            names='subj_obs_func')
        observation_function = tf.reshape(
            observation_function,
            shape=observation_function.get_shape().as_list()[:3] +
                  [params.num_action, params.num_obs])
        observation_function = tf.math.divide(
            observation_function,
            tf.reduce_sum(
                observation_function,
                axis=-1,
                keepdims=True) + 1e-10)

        return observation_function

    @staticmethod
    def get_prior_observation_function(
            env_map,
            params):
        env_map = tf.expand_dims(
            env_map,
            axis=-1)
        observation_channel = conv_layers(
            env_map,
            conv_params=np.array(
                [[3, 200, 'linear'],
                 [1, params.num_obs, 'sigmoid']]),
            names='subj_obs_func')
        rest_action_channel = tf.ones(
            observation_channel.get_shape().as_list())
        observation_channel = tf.math.divide(
            observation_channel,
            tf.reduce_sum(
                observation_channel,
                axis=-1,
                keepdims=True) + 1e-10)
        rest_action_channel = tf.math.divide(
            rest_action_channel,
            tf.reduce_sum(
                rest_action_channel,
                axis=-1,
                keepdims=True))
        observation_function = tf.stack(
            [observation_channel,
             rest_action_channel,
             rest_action_channel,
             rest_action_channel,
             rest_action_channel,
             rest_action_channel], axis=-2)

        return observation_function

    @staticmethod
    def get_action_index_vector(
            action,
            num_action):
        """
        :param action:
        :param num_action:
        """
        action_index_vector = tf.one_hot(action, num_action)
        return action_index_vector

    @staticmethod
    def get_observation_index_vector(
            observation,
            num_observation):
        """

        :param observation:
        :param num_observation:
        """
        observation_index_vector = tf.one_hot(
            observation, num_observation)
        return observation_index_vector

    @staticmethod
    def get_observation_soft_index_vector(
            observation,
            num_observation):
        """

        :param observation
        :param num_observation
        """
        with tf.compat.v1.variable_scope(
                "obs_idx_vec", reuse=tf.AUTO_REUSE):
            observation_soft_index_vector = fc_layers(
                observation,
                fc_params=np.array(
                    [[num_observation, 'tanh'],
                     [num_observation, 'softmax']]),
                names='fc_subj_obs')

        return observation_soft_index_vector

    @staticmethod
    def belief_update(
            observation_function,
            belief,
            action,
            observation,
            params):
        """

        :param observation_function
        :param belief: belief (b_i), [batch, N, M, 1]
        :param action: action input (a_i)
        :param observation: observation input (o_i)
        :param params
        :return: updated belief b_(i+1)
        """
        # step 1: update belief with transition
        # get transition kernel (T)
        transition_function = \
            BeliefFilterNet.get_transition_function4filtering(
                name='trans_func',
                params=params)

        # apply convolution which corresponds to the transition function in an MDP (f_T)
        belief = tf.tile(
            tf.expand_dims(
                belief,
                axis=-1),
            multiples=[1, 1, 1, params.num_action])
        # cond_belief = tf.stack(
        #     [belief, env_map, goal_map],
        #     axis=-1)
        predicted_belief = tf.nn.depthwise_conv2d(
            belief,
            transition_function,
            strides=[1, 1, 1, 1],
            padding='SAME')

        # index into the appropriate channel of b_prime
        action_index_vector = \
            BeliefFilterNet.get_action_index_vector(
                action=action,
                num_action=params.num_action)
        action_index_vector = action_index_vector[:, None, None]
        predicted_belief = tf.reduce_sum(
            tf.math.multiply(
                predicted_belief,
                action_index_vector),
            axis=3,
            keepdims=False)
        # predicted_belief = tf.reshape(
        #     predicted_belief,
        #     shape=predicted_belief.get_shape().as_list()[:1] + [params.num_state])
        # predicted_belief = tf.compat.v1.nn.softmax(
        #     predicted_belief,
        #     axis=1)
        # predicted_belief = tf.reshape(
        #     predicted_belief,
        #     shape=predicted_belief.get_shape().as_list()[:1] + [params.grid_n, params.grid_m])

        # step 2: update belief with observation
        # get observation probabilities for the obseravtion input by soft indexing
        observation_index_vector = \
            BeliefFilterNet.get_observation_index_vector(
                observation=observation,
                num_observation=params.num_obs)
        observation_index_vector = observation_index_vector[:, None, None, None]
        observation_weights = tf.reduce_sum(
            tf.math.multiply(
                tf.reduce_sum(
                    tf.math.multiply(
                        observation_function,
                        observation_index_vector),
                    axis=4,
                    keepdims=False),
                action_index_vector),
            axis=3,
            keepdims=False)

        corrected_belief = tf.math.multiply(
            predicted_belief,
            observation_weights)

        # step 3: normalize over the state space
        # add small number to avoid division by zero
        corrected_belief = tf.math.divide(
            corrected_belief,
            tf.reduce_sum(
                corrected_belief,
                axis=[1, 2],
                keepdims=True) + 1e-8)

        return corrected_belief
