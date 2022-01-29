import sys
import tensorflow as tf
import numpy as np
sys.path.append('/home/gz67063/projects/sipomdplite-net')
from network.models.frames import SIPOMDPLiteFrames
from network.layers import conv4d


class PlanningNet(object):
    """

    """
    @staticmethod
    def value_iteration(
            subj_env_map,
            subj_goal_map,
            joint_env_map,
            joint_goal_map,
            params,
            ma_reward_interaction_indicator,
            ma_transition_interaction_indicator,
            subj_sa_isolate_reward_function=None,
            subj_sa_isolate_transition_function=None,
            obj_sa_isolate_transition_function=None,
            nested_policy=None):
        """

        :param subj_env_map:
        :param subj_goal_map:
        :param joint_env_map:
        :param joint_goal_map:
        :param params:
        :param ma_reward_interaction_indicator:
        :param ma_transition_interaction_indicator:
        :param subj_sa_isolate_reward_function:
        :param subj_sa_isolate_transition_function:
        :param obj_sa_isolate_transition_function:
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

        ma_transition_function = \
            SIPOMDPLiteFrames.get_ma_transition_function4planning(
                name='ma_trans_func_planning',
                params=params)

        ma_transition_non_interaction_indicator = \
            1.0 - ma_transition_interaction_indicator

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
            joint_env_map=joint_env_map,
            joint_goal_map=joint_goal_map,
            subj_sa_isolate_reward_function=subj_sa_isolate_reward_function,
            ma_reward_interaction_indicator=ma_reward_interaction_indicator,
            params=params)

        utilities = tf.zeros(
            subj_ma_reward_function.get_shape().as_list()[:1] +
            [np.prod(subj_ma_reward_function.get_shape().as_list()[1:3])] +
            subj_ma_reward_function.get_shape().as_list()[3:5] + [1])
        q_values = None
        reuse = False

        for i in range(params.K):
            # Value iteration for non-interactive data points.
            utilities_obj_ls = tf.unstack(
                utilities,
                axis=1,
                name='unstack_u_over_subj')
            q_vals = list()
            for u in utilities_obj_ls:
                q = tf.compat.v1.nn.conv2d(
                    u,
                    filter=obj_sa_isolate_transition_function,
                    strides=[1, 1, 1, 1],
                    padding='SAME')
                q_vals.append(q)
            q_vals = tf.stack(
                q_vals,
                axis=-1,
                name='stack_q_over_subj')
            q_vals = tf.reshape(
                q_vals,
                shape=utilities.get_shape().as_list()[:1] +
                      [np.prod(q_vals.get_shape().as_list()[1:4])] +
                      subj_ma_reward_function.get_shape().as_list()[1:3])
            q_val_ls = tf.unstack(
                q_vals,
                axis=1,
                name='unstack_q_over_obj')
            q_vals = list()
            for q in q_val_ls:
                q = tf.expand_dims(q, axis=-1)
                qq = tf.compat.v1.nn.conv2d(
                    q,
                    filter=subj_sa_isolate_transition_function,
                    strides=[1, 1, 1, 1],
                    padding='SAME')
                q_vals.append(qq)
            q_vals = tf.stack(
                q_vals,
                axis=-1,
                name='stack_q_over_obj')
            q_vals = tf.reshape(
                q_vals,
                shape=subj_ma_reward_function.get_shape().as_list()[:3] +
                      obj_sa_isolate_transition_function.get_shape().as_list()[-1:] +
                      subj_ma_reward_function.get_shape().as_list()[3:5] +
                      subj_sa_isolate_transition_function.get_shape().as_list()[-1:])
            q_vals_non_interact = tf.transpose(
                q_vals,
                perm=[0, 1, 2, 4, 5, 3, 6])

            # Value iteration for interactive data points.
            utilities = tf.reshape(
                utilities,
                shape=subj_ma_reward_function.get_shape().as_list()[:-2])
            utilities = utilities[:, None]
            q_vals_interact = conv4d(
                utilities,
                filters=params.num_joint_action,
                kernel_size=(3, 3, 3, 3),
                strides=[1, 1, 1, 1],
                padding='SAME',
                data_format='channels_first',
                kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                    mean=1.0 / (9.0 ** 2),
                    stddev=1.0 / 90.0,
                    dtype=tf.float32),
                name='val_func',
                reuse=reuse)
            reuse = True
            q_vals_interact = tf.reshape(
                q_vals_interact,
                shape=q_vals_interact.get_shape().as_list()[:1] +
                      [params.num_action, params.num_action] +
                      utilities.get_shape().as_list()[2:])
            q_vals_interact = tf.transpose(
                q_vals_interact,
                perm=[0, 3, 4, 5, 6, 1, 2])
            q_values = tf.math.multiply(
                q_vals_interact,
                ma_transition_interaction_indicator) + \
                       tf.math.multiply(
                           q_vals_non_interact,
                           ma_transition_non_interaction_indicator)
            q_values += subj_ma_reward_function
            q_values = tf.reduce_sum(
                tf.math.multiply(
                    q_values,
                    nested_policy),
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

        utilities = tf.reshape(
            utilities,
            shape=subj_ma_reward_function.get_shape().as_list()[:-2])

        return q_values, utilities

    @staticmethod
    def maqmdp_policy(
            q_values,
            belief):
        belief = belief[:, :, :, :, :, None]
        action_values = tf.reduce_sum(
            tf.math.multiply(
                q_values,
                belief),
            axis=[1, 2, 3, 4],
            keepdims=False)
        action = tf.math.log(
            tf.nn.softmax(action_values) + 1e-10)

        return action
