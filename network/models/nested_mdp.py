import sys
import tensorflow as tf
import numpy as np
sys.path.append('/home/gz67063/projects/sipomdplite-net')
from network.models.frames import SIPOMDPLiteFrames


class PlanningNet(object):
    """

    """
    @staticmethod
    def value_iteration(
            subj_env_map,
            subj_goal_map,
            obj_env_map,
            obj_goal_map,
            params,
            subj_ma_transition_interaction_indicator,
            obj_ma_transition_interaction_indicator,
            common_ma_transition_interaction_indicator,
            subj_sa_isolate_reward_function=None,
            subj_sa_isolate_transition_function=None,
            obj_sa_isolate_reward_function=None,
            obj_sa_isolate_transition_function=None,
            ma_interact_transition_function=None,
            nested_policy=None):
        """

        :param subj_env_map:
        :param subj_goal_map:
        :param obj_env_map:
        :param obj_goal_map:
        :param params:
        :param subj_ma_transition_interaction_indicator
        :param obj_ma_transition_interaction_indicator
        :param common_ma_transition_interaction_indicator:
        :param subj_sa_isolate_reward_function:
        :param subj_sa_isolate_transition_function:
        :param obj_sa_isolate_reward_function:
        :param obj_sa_isolate_transition_function:
        :param ma_interact_transition_function
        :param nested_policy:
        """
        # Get transition functions.
        if subj_sa_isolate_transition_function is None:
            subj_sa_isolate_transition_function = \
                SIPOMDPLiteFrames.get_sa_transition_function4planning(
                    name='subj_isolate_trans_func',
                    params=params)
        else:
            subj_sa_isolate_transition_function = \
                SIPOMDPLiteFrames.process_loaded_sa_transition_function4planning(
                    kernel_weights=subj_sa_isolate_transition_function,
                    params=params)

        if obj_sa_isolate_transition_function is None:
            obj_sa_isolate_transition_function = \
                SIPOMDPLiteFrames.get_sa_transition_function4planning(
                    name='obj_sa_isolate_trans_func',
                    params=params)
        else:
            obj_sa_isolate_transition_function = \
                SIPOMDPLiteFrames.process_loaded_sa_transition_function4planning(
                    kernel_weights=obj_sa_isolate_transition_function,
                    params=params)

        # Process transition interaction indicators
        subj_ma_transition_interaction_indicator -= \
            common_ma_transition_interaction_indicator
        obj_ma_transition_interaction_indicator -= \
            common_ma_transition_interaction_indicator
        ma_transition_non_interaction_indicator = \
            1.0 - \
            subj_ma_transition_interaction_indicator - \
            obj_ma_transition_interaction_indicator - \
            common_ma_transition_interaction_indicator

        # Get reward functions.
        if subj_sa_isolate_reward_function is None:
            subj_sa_isolate_reward_function = \
                SIPOMDPLiteFrames.get_sa_reward_function(
                    name='subj_isolate_reward_func',
                    env_map=subj_env_map,
                    goal_map=subj_goal_map,
                    params=params)
        else:
            subj_sa_isolate_reward_function = \
                SIPOMDPLiteFrames.process_loaded_sa_reward_function(
                    env_map=subj_env_map,
                    goal_map=subj_goal_map,
                    kernel_weights=subj_sa_isolate_reward_function['weights'],
                    biases=subj_sa_isolate_reward_function['biases'])

        subj_ma_reward_function = SIPOMDPLiteFrames.get_subj_ma_reward_function(
            subj_sa_isolate_reward_function=subj_sa_isolate_reward_function,
            params=params)
        if obj_sa_isolate_reward_function is None:
            obj_sa_isolate_reward_function = \
                SIPOMDPLiteFrames.get_sa_reward_function(
                    name='obj_isolate_reward_func',
                    env_map=obj_env_map,
                    goal_map=obj_goal_map,
                    params=params)
        else:
            obj_sa_isolate_reward_function = \
                SIPOMDPLiteFrames.process_loaded_sa_reward_function(
                    env_map=obj_env_map,
                    goal_map=obj_goal_map,
                    kernel_weights=obj_sa_isolate_reward_function['weights'],
                    biases=obj_sa_isolate_reward_function['biases'])

        obj_ma_reward_function = SIPOMDPLiteFrames.get_obj_ma_reward_function(
            obj_sa_isolate_reward_function=obj_sa_isolate_reward_function,
            params=params)
        tensor_shape = \
            obj_ma_reward_function.get_shape().as_list()
        utilities = tf.zeros(
            tensor_shape[:1] +
            [np.prod(tensor_shape[1:3])] +
            tensor_shape[3:5] + [1])
        q_values = None

        for i in range(params.K):
            q_values_w_subj_interaction = tf.compat.v1.nn.conv3d(
                utilities,
                obj_sa_interact_transition_function,
                strides=[1, 1, 1, 1, 1],
                padding='SAME')
            tensor_shape = \
                q_values_w_subj_interaction.get_shape().as_list()
            q_values_w_subj_interaction = tf.reshape(
                q_values_w_subj_interaction,
                shape=tensor_shape[:1] +
                      [params.grid_n, params.grid_m] +
                      tensor_shape[2:])
            q_values_w_subj_interaction = tf.transpose(
                q_values_w_subj_interaction,
                perm=[0, 3, 4, 5, 1, 2])
            tensor_shape = \
                q_values_w_subj_interaction.get_shape().as_list()
            q_values_w_subj_interaction = tf.reshape(
                q_values_w_subj_interaction,
                shape=tensor_shape[:1] +
                      [np.prod(tensor_shape[1:4])] +
                      tensor_shape[4:] + [1])
            q_values_w_subj_obj_interaction = tf.compat.v1.nn.conv3d(
                q_values_w_subj_interaction,
                subj_sa_interact_transition_function,
                strides=[1, 1, 1, 1, 1],
                padding='SAME')
            tensor_shape = \
                q_values_w_subj_obj_interaction.get_shape().as_list()
            q_values_w_subj_obj_interaction = tf.reshape(
                q_values_w_subj_obj_interaction,
                shape=tensor_shape[:1] +
                      [params.grid_n,
                       params.grid_m,
                       params.num_action] +
                      tensor_shape[2:])
            q_values_w_subj_obj_interaction = tf.transpose(
                q_values_w_subj_obj_interaction,
                perm=[0, 4, 5, 1, 2, 6, 3])
            q_values_section_w_subj_obj_interaction = tf.math.multiply(
                q_values_w_subj_obj_interaction,
                common_ma_transition_interaction_indicator)
            q_values_w_subj_wo_obj_interaction = tf.compat.v1.nn.conv3d(
                q_values_w_subj_interaction,
                subj_sa_isolate_transition_function,
                strides=[1, 1, 1, 1, 1],
                padding='SAME')
            tensor_shape = \
                q_values_w_subj_wo_obj_interaction.get_shape().as_list()
            q_values_w_subj_wo_obj_interaction = tf.reshape(
                q_values_w_subj_wo_obj_interaction,
                shape=tensor_shape[:1] +
                      [params.grid_n,
                       params.grid_m,
                       params.num_action] +
                      tensor_shape[2:])
            q_values_w_subj_wo_obj_interaction = tf.transpose(
                q_values_w_subj_wo_obj_interaction,
                perm=[0, 4, 5, 1, 2, 6, 3])
            q_values_section_w_subj_wo_obj_interaction = tf.math.multiply(
                q_values_w_subj_wo_obj_interaction,
                subj_ma_transition_interaction_indicator)

            q_values_wo_subj_interaction = tf.compat.v1.nn.conv3d(
                utilities,
                obj_sa_isolate_transition_function,
                strides=[1, 1, 1, 1, 1],
                padding='SAME')
            tensor_shape = \
                q_values_wo_subj_interaction.get_shape().as_list()
            q_values_wo_subj_interaction = tf.reshape(
                q_values_wo_subj_interaction,
                shape=tensor_shape[:1] +
                      [params.grid_n, params.grid_m] +
                      tensor_shape[2:])
            q_values_wo_subj_interaction = tf.transpose(
                q_values_wo_subj_interaction,
                perm=[0, 3, 4, 5, 1, 2])
            tensor_shape = \
                q_values_wo_subj_interaction.get_shape().as_list()
            q_values_wo_subj_interaction = tf.reshape(
                q_values_wo_subj_interaction,
                shape=tensor_shape[:1] +
                      [np.prod(tensor_shape[1:4])] +
                      tensor_shape[4:] + [1])
            q_values_wo_subj_w_obj_interaction = tf.compat.v1.nn.conv3d(
                q_values_wo_subj_interaction,
                subj_sa_interact_transition_function,
                strides=[1, 1, 1, 1, 1],
                padding='SAME')
            tensor_shape = \
                q_values_wo_subj_w_obj_interaction.get_shape().as_list()
            q_values_wo_subj_w_obj_interaction = tf.reshape(
                q_values_wo_subj_w_obj_interaction,
                shape=tensor_shape[:1] +
                      [params.grid_n,
                       params.grid_m,
                       params.num_action] +
                      tensor_shape[2:])
            q_values_wo_subj_w_obj_interaction = tf.transpose(
                q_values_wo_subj_w_obj_interaction,
                perm=[0, 4, 5, 1, 2, 6, 3])
            q_values_section_wo_subj_w_obj_interaction = tf.math.multiply(
                q_values_wo_subj_w_obj_interaction,
                obj_ma_transition_interaction_indicator)
            q_values_wo_subj_obj_interaction = tf.compat.v1.nn.conv3d(
                q_values_wo_subj_interaction,
                subj_sa_isolate_transition_function,
                strides=[1, 1, 1, 1, 1],
                padding='SAME')
            tensor_shape = \
                q_values_wo_subj_obj_interaction.get_shape().as_list()
            q_values_wo_subj_obj_interaction = tf.reshape(
                q_values_wo_subj_obj_interaction,
                shape=tensor_shape[:1] +
                      [params.grid_n,
                       params.grid_m,
                       params.num_action] +
                      tensor_shape[2:])
            q_values_wo_subj_obj_interaction = tf.transpose(
                q_values_wo_subj_obj_interaction,
                perm=[0, 4, 5, 1, 2, 6, 3])
            q_values_section_wo_subj_obj_interaction = tf.math.multiply(
                q_values_wo_subj_obj_interaction,
                ma_transition_non_interaction_indicator)

            q_values = q_values_section_w_subj_obj_interaction + \
                q_values_section_w_subj_wo_obj_interaction + \
                q_values_section_wo_subj_w_obj_interaction + \
                q_values_section_wo_subj_obj_interaction

            q_values += subj_ma_reward_function
            q_values = tf.reduce_sum(
                tf.math.multiply(
                    q_values, nested_policy),
                axis=-1,
                keepdims=False)
            utilities = tf.reduce_max(
                q_values,
                axis=-1,
                keepdims=True)
            utilities = tf.reshape(
                utilities,
                shape=utilities.get_shape().as_list()[:1] +
                      [np.prod(utilities.get_shape().as_list()[1:3])] +
                      utilities.get_shape().as_list()[3:])

        return q_values, utilities

    @staticmethod
    def nested_mdp_policy(
            q_values,
            lower_level_policy,
            params):
        if params.reason_agent == "objective":
            self_q_values = tf.reduce_sum(
                tf.math.multiply(
                    q_values,
                    lower_level_policy),
                axis=-2,
                keepdims=True)
            policy = tf.nn.softmax(
                self_q_values,
                axis=-1)
        else:
            self_q_values = tf.reduce_sum(
                tf.math.multiply(
                    q_values,
                    lower_level_policy),
                axis=-1,
                keepdims=True)
            policy = tf.nn.softmax(
                self_q_values,
                axis=-2)

        return policy