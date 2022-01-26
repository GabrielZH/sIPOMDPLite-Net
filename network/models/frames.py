import sys

import tensorflow as tf
import numpy as np

sys.path.append('/home/gz67063/projects/sipomdplite-net')
from network.layers import conv_layers, conv4d_layers


class SIPOMDPLiteFrames(object):
    """

    """
    """
    -----------------------------------------
    """
    """
    Transition models for all agents, including
    each agent's single-agent transition functions
    under situations of either interaction or 
    isolation, and the indicator function that 
    indicates the active transition interaction
    condition
    """

    @staticmethod
    def get_sa_transition_function4filtering(
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
    def get_sa_transition_function4planning(
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
    def process_loaded_sa_transition_function4filtering(
            kernel_weights,
            params):
        sa_transition_function = tf.nn.softmax(
            kernel_weights,
            axis=0)
        sa_transition_function = tf.reshape(
            sa_transition_function,
            shape=(3, 3, params.num_action, 1))

        return sa_transition_function

    @staticmethod
    def process_loaded_sa_transition_function4planning(
            kernel_weights,
            params):
        sa_transition_function = tf.nn.softmax(
            kernel_weights,
            axis=0)
        sa_transition_function = tf.reshape(
            sa_transition_function,
            shape=(3, 3, 1, params.num_action))

        return sa_transition_function

    @staticmethod
    def get_ma_transition_function4filtering(
            name):
        """

        :param name:
        """
        initializer = tf.compat.v1.truncated_normal_initializer(
            mean=1.0 / (9.0 ** 2),
            stddev=1.0 / 90.0,
            dtype=tf.float32)
        transition_function = tf.compat.v1.get_variable(
            name=name,
            shape=((3 ** 2) ** 2,),
            initializer=initializer,
            dtype=tf.float32)
        transition_function = tf.nn.softmax(
            transition_function,
            axis=0)
        transition_function = tf.reshape(
            transition_function,
            shape=(3, 3, 3, 3, 1, 1))

        return transition_function

    @staticmethod
    def get_ma_transition_function4planning(
            name,
            params):
        initializer = tf.compat.v1.truncated_normal_initializer(
            mean=1.0 / (9.0 ** 2),
            stddev=1.0 / 90.0,
            dtype=tf.float32)
        transition_function = tf.compat.v1.get_variable(
            name=name,
            shape=((3 ** 2) ** 2, params.num_joint_action),
            initializer=initializer,
            dtype=tf.float32)
        transition_function = tf.nn.softmax(
            transition_function,
            axis=0)
        transition_function = tf.reshape(
            transition_function,
            shape=(3, 3, 3, 3, 1, params.num_joint_action))

        return transition_function

    @staticmethod
    def get_prior_sa_transition_interaction_indicator(
            goal_map,
            transition_interactive_action,
            params):
        """

        :param goal_map:
        :param transition_interactive_action:
        :param params:
        """
        transition_interaction_indicator = list()
        for action in range(params.num_action):
            if action == transition_interactive_action:
                transition_interaction_indicator.append(goal_map)
            else:
                transition_interaction_indicator.append(
                    tf.zeros(
                        goal_map.get_shape().as_list()))
        transition_interaction_indicator = tf.stack(
            transition_interaction_indicator,
            axis=3)

        return transition_interaction_indicator

    @staticmethod
    def get_sa_transition_interaction_indicator(
            name,
            goal_map,
            params):
        """

        :param name:
        :param goal_map:
        :param params:
        """
        goal_map = tf.expand_dims(goal_map, axis=-1)
        transition_interaction_indicator = conv_layers(
            goal_map,
            conv_params=np.array(
                [[1, params.num_action, 'sigmoid']]),
            add_bias=True,
            names=name)

        return transition_interaction_indicator

    @staticmethod
    def get_sa_observation_function(
            env_map,
            params):
        """

        :param env_map: env map of the subjective agent
        :param params:
        """
        env_map = tf.expand_dims(env_map, axis=-1)

        sa_observation_function = conv_layers(
            env_map,
            conv_params=np.array(
                [[3, 200, 'linear'],
                 [1, params.num_action * params.num_obs, 'sigmoid']]),
            names='subj_obs_func')
        sa_observation_function = tf.reshape(
            sa_observation_function,
            shape=sa_observation_function.get_shape().as_list()[:3] +
                  [params.num_action, params.num_obs])
        sa_observation_function = tf.math.divide(
            sa_observation_function,
            tf.reduce_sum(
                sa_observation_function,
                axis=-1,
                keepdims=True) + 1e-10)

        return sa_observation_function

    @staticmethod
    def get_prior_sa_observation_function(
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
        sa_observation_function = tf.stack(
            [observation_channel,
             rest_action_channel,
             rest_action_channel,
             rest_action_channel,
             rest_action_channel,
             rest_action_channel], axis=-2)

        return sa_observation_function

    @staticmethod
    def process_loaded_sa_observation_function(
            env_map,
            kernel_weights,
            biases=None):
        input_tensor = tf.expand_dims(
            env_map,
            axis=-1)

        sa_observation_function = None
        for layer_i in range(len(kernel_weights)):
            sa_observation_function = tf.nn.conv2d(
                input_tensor,
                kernel_weights[layer_i],
                strides=[1, 1, 1, 1],
                padding='SAME')
            input_tensor = sa_observation_function
            if biases is not None:
                sa_observation_function = tf.nn.bias_add(
                    sa_observation_function,
                    bias=biases[layer_i])

        return sa_observation_function

    @staticmethod
    def process_loaded_prior_sa_observation_function(
            env_map,
            kernel_weights,
            biases=None):
        input_tensor = tf.expand_dims(
            env_map,
            axis=-1)

        observation_channel = None
        for layer_i in range(len(kernel_weights)):
            observation_channel = tf.nn.conv2d(
                input_tensor,
                kernel_weights[layer_i],
                strides=[1, 1, 1, 1],
                padding='SAME')
            input_tensor = observation_channel
            if biases is not None:
                observation_channel = tf.nn.bias_add(
                    observation_channel,
                    bias=biases[layer_i])

        rest_action_channel = tf.ones(
            observation_channel.get_shape().as_list())
        rest_action_channel = tf.math.divide(
            rest_action_channel,
            tf.reduce_sum(
                rest_action_channel,
                axis=-1,
                keepdims=True))

        sa_observation_function = tf.stack(
            [observation_channel,
             rest_action_channel,
             rest_action_channel,
             rest_action_channel,
             rest_action_channel,
             rest_action_channel], axis=-2)

        return sa_observation_function

    @staticmethod
    def get_ma_observation_function(
            sa_observation_function,
            params):
        ma_observation_function = tf.tile(
            sa_observation_function[:, :, :, None, None],
            multiples=[1, 1, 1, params.grid_n, params.grid_m, 1, 1])

        return ma_observation_function

    """
    Reward models for all agents, including
    each agent's single-agent reward functions
    under situations of either interaction or 
    isolation, and the indicator function that
    indicates the active reward interaction 
    condition.
    """
    @staticmethod
    def get_sa_reward_function(
            name,
            env_map,
            goal_map,
            params):
        """

        :param name:
        :param env_map:
        :param goal_map:
        :param params
        """
        task = tf.stack([env_map, goal_map], axis=-1)
        reward_function = conv_layers(
            task,
            conv_params=np.array(
                [[3, 200, 'relu'],
                 [1, params.num_action, 'lin']]),
            names=name)

        return reward_function

    @staticmethod
    def process_loaded_sa_reward_function(
            env_map,
            goal_map,
            kernel_weights,
            biases=None):
        input_tensor = tf.stack([env_map, goal_map], axis=-1)

        sa_reward_function = None
        for layer_i in range(len(kernel_weights)):
            sa_reward_function = tf.nn.conv2d(
                input_tensor,
                kernel_weights[layer_i],
                strides=[1, 1, 1, 1],
                padding='SAME')
            if biases is not None:
                sa_reward_function = tf.nn.bias_add(
                    sa_reward_function,
                    bias=biases[layer_i])
            input_tensor = sa_reward_function

        return sa_reward_function

    @staticmethod
    def get_subj_ma_reward_function(
            joint_env_map,
            joint_goal_map,
            subj_sa_isolate_reward_function,
            ma_reward_interaction_indicator,
            params):
        """

        :param joint_env_map:
        :param joint_goal_map:
        :param subj_sa_isolate_reward_function
        :param ma_reward_interaction_indicator
        """
        # Process the non-interactive part of R.
        expand_subj_sa_reward_function = tf.tile(
            subj_sa_isolate_reward_function[:, :, :, None, None, :, None],
            multiples=[1, 1, 1, params.grid_n, params.grid_m, 1, params.num_action])

        # Derive the interactive part of R.
        joint_map_feature = tf.stack(
            [joint_env_map, joint_goal_map],
            axis=1)
        subj_ma_reward_function = conv4d_layers(
            joint_map_feature,
            conv_params=np.array(
                [[3, 100, 'relu'],
                 [1, params.num_joint_action, 'lin']]),
            names='subj_ma_interact_reward_func')
        subj_ma_reward_function = tf.transpose(
            subj_ma_reward_function,
            perm=[0, 2, 3, 4, 5, 1])
        subj_ma_reward_function = tf.reshape(
            subj_ma_reward_function,
            shape=subj_ma_reward_function.get_shape().as_list()[:-1] +
                  [params.num_action, params.num_action])

        ma_reward_non_interaction_indicator = \
            1.0 - ma_reward_interaction_indicator

        subj_ma_reward_function = tf.math.multiply(
            subj_ma_reward_function,
            ma_reward_interaction_indicator) + \
                                  tf.math.multiply(
                                      expand_subj_sa_reward_function,
                                      ma_reward_non_interaction_indicator)

        return subj_ma_reward_function

    @staticmethod
    def get_prior_sa_reward_interaction_indicator(
            goal_map,
            reward_interactive_action,
            params):
        """

        :param goal_map:
        :param reward_interactive_action:
        :param params:
        """
        reward_interaction_indicator = list()
        for action in range(params.num_action):
            if action == reward_interactive_action:
                reward_interaction_indicator.append(goal_map)
            else:
                reward_interaction_indicator.append(
                    tf.zeros(
                        goal_map.get_shape().as_list()))
        reward_interaction_indicator = tf.stack(
            reward_interaction_indicator,
            axis=3)

        return reward_interaction_indicator

    @staticmethod
    def get_sa_reward_interaction_indicator(
            name,
            goal_map,
            params):
        """

        :param name:
        :param goal_map:
        :param params:
        """
        goal_map = tf.expand_dims(goal_map, axis=-1)
        reward_interaction_indicator = conv_layers(
            goal_map,
            conv_params=np.array(
                [[1, params.num_action, 'sigmoid']]),
            add_bias=True,
            names=name)

        return reward_interaction_indicator

    @staticmethod
    def get_subj_ma_interaction_indicator(
            subj_sa_interaction_indicator,
            obj_sa_interaction_indicator):
        """

        :param subj_sa_interaction_indicator:
        subjective agent's single-agent transition
        or reward interaction indicator.
        :param obj_sa_interaction_indicator:
        objective agent's single-agent transition
        or reward interaction indicator (should be
        consistent with the first parameter in
        the function type, i.e., transition or
        reward.)
        """
        obj_sa_unit_tensor = tf.ones(
            obj_sa_interaction_indicator.get_shape().as_list())
        obj_sa_unit_tensor = obj_sa_unit_tensor[:, None, None, :, :, None]
        subj_sa_interaction_indicator = \
            subj_sa_interaction_indicator[:, :, :, None, None, :, None]
        subj_ma_interaction_indicator = \
            tf.math.multiply(
                subj_sa_interaction_indicator,
                obj_sa_unit_tensor)

        return subj_ma_interaction_indicator

    @staticmethod
    def get_obj_ma_interaction_indicator(
            subj_sa_interaction_indicator,
            obj_sa_interaction_indicator):
        """

        :param subj_sa_interaction_indicator:
        subjective agent's single-agent transition
        or reward interaction indicator.
        :param obj_sa_interaction_indicator:
        objective agent's single-agent transition
        or reward interaction indicator (should be
        consistent with the first parameter in
        the function type, i.e., transition or
        reward.
        """
        subj_sa_unit_tensor = tf.ones(
            subj_sa_interaction_indicator.get_shape().as_list())
        subj_sa_unit_tensor = subj_sa_unit_tensor[:, :, :, None, None, :, None]
        obj_sa_interaction_indicator = \
            obj_sa_interaction_indicator[:, None, None, :, :, None]
        obj_ma_interaction_indicator = \
            tf.math.multiply(
                subj_sa_unit_tensor,
                obj_sa_interaction_indicator)

        return obj_ma_interaction_indicator

    @staticmethod
    def get_common_ma_interaction_indicator(
            subj_sa_interaction_indicator,
            obj_sa_interaction_indicator):
        subj_sa_interaction_indicator = \
            subj_sa_interaction_indicator[:, :, :, None, None, :, None]
        obj_sa_interaction_indicator = \
            obj_sa_interaction_indicator[:, None, None, :, :, None]
        common_ma_interaction_indicator = \
            tf.math.multiply(
                subj_sa_interaction_indicator,
                obj_sa_interaction_indicator)
        return common_ma_interaction_indicator
