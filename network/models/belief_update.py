import sys
import tensorflow as tf
import numpy as np

sys.path.append('/home/gz67063/projects/sipomdplite-net')
from network.models.frames import SIPOMDPLiteFrames
from network.layers import fc_layers, conv4d


class BeliefFilterNet(object):
    """

    """
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
    def belief_prediction(
            belief,
            subj_sa_isolate_transition_function,
            obj_sa_isolate_transition_function,
            ma_transition_interaction_indicator,
            subj_action,
            obj_action,
            num_action):
        """

        :param belief:
        :param subj_sa_isolate_transition_function:
        :param obj_sa_isolate_transition_function:
        :param ma_transition_interaction_indicator
        :param subj_action:
        :param obj_action:
        :param num_action:
        """
        belief = belief[:, :, :, :, :, None, None]
        b_interaction = tf.math.multiply(
            belief,
            ma_transition_interaction_indicator)
        b_interaction = tf.reshape(
            b_interaction,
            shape=b_interaction.get_shape().as_list()[:-2] +
                  np.prod(b_interaction.get_shape().as_list()[-2:]))
        b_interaction_ls = tf.unstack(
            b_interaction,
            axis=-1,
            name='unstack_b_interact_over_act')
        pred_b_interaction = list()
        i = 0
        for b in b_interaction_ls:
            b = tf.expand_dims(b, axis=1)
            ma_transition_function = \
                SIPOMDPLiteFrames.get_ma_transition_function4filtering(
                    name='ma_trans_func_filtering_' + str(i))
            pred_b = conv4d(
                b,
                filters=1,
                kernel_size=(3, 3, 3, 3),
                strides=[1, 1, 1, 1],
                padding='SAME',
                data_format='channels_first',
                kernel_initializer=ma_transition_function)
            pred_b_interaction.append(pred_b)
            i += 1
        pred_b_interaction = tf.stack(
            pred_b_interaction,
            axis=-1,
            name='stack_b_interact_over_act')
        pred_b_interaction = tf.reshape(
            pred_b_interaction,
            shape=ma_transition_interaction_indicator.get_shape().as_list())
        subj_a_idx = BeliefFilterNet.get_action_index_vector(
            action=subj_action,
            num_action=num_action)
        subj_a_idx = subj_a_idx[:, None, None, None, None]
        obj_a_idx = BeliefFilterNet.get_action_index_vector(
            action=obj_action,
            num_action=num_action)
        obj_a_idx = obj_a_idx[:, None, None, None, None]
        pred_b_interaction = tf.reduce_sum(
            tf.math.multiply(
                pred_b_interaction,
                subj_a_idx),
            axis=-2,
            keepdims=False)
        pred_b_interaction = tf.reduce_sum(
            tf.math.multiply(
                pred_b_interaction,
                obj_a_idx),
            axis=-1,
            keepdims=False)

        b_non_interaction = tf.math.multiply(
            belief,
            1.0 - ma_transition_interaction_indicator)
        b_non_interaction = tf.transpose(
            b_non_interaction,
            perm=[0, 1, 2, 5, 3, 4, 6])
        b_non_interaction = tf.reshape(
            b_non_interaction,
            shape=b_non_interaction.get_shape().as_list()[:1] +
                  [np.prod(b_non_interaction.get_shape().as_list()[1:4])] +
                  b_non_interaction.get_shape().as_list()[4:])
        b_non_interaction_obj_ls = tf.unstack(
            b_non_interaction,
            axis=1,
            name='unstack_b_non_interact_over_subj')
        pred_b_non_interaction_obj = list()
        obj_a_idx = BeliefFilterNet.get_action_index_vector(
            action=obj_action,
            num_action=num_action)
        obj_a_idx = obj_a_idx[:, None, None, None]
        for b in b_non_interaction_obj_ls:
            pred_b = tf.compat.v1.depthwise_conv2d(
                b,
                filter=obj_sa_isolate_transition_function,
                strides=[1, 1, 1, 1],
                padding='SAME')
            pred_b_a = tf.reduce_sum(
                tf.math.multiply(pred_b, obj_a_idx),
                axis=-1,
                keepdims=False)
            pred_b_non_interaction_obj.append(pred_b_a)
        pred_b_non_interaction_obj = tf.stack(
            pred_b_non_interaction_obj,
            axis=-1,
            name='stack_b_non_interact_over_subj')
        pred_b_non_interaction_obj = tf.reshape(
            pred_b_non_interaction_obj,
            shape=pred_b_non_interaction_obj.get_shape().as_list()[:1] +
                  [np.prod(pred_b_non_interaction_obj.get_shape().as_list()[1:3])] +
                  pred_b_non_interaction_obj.get_shape().as_list()[3:])
        pred_b_non_interaction_subj_ls = tf.unstack(
            pred_b_non_interaction_obj,
            axis=1,
            name='unstack_b_non_interact_over_obj')
        pred_b_non_interaction = list()
        subj_a_idx = BeliefFilterNet.get_action_index_vector(
            action=subj_action,
            num_action=num_action)
        subj_a_idx = subj_a_idx[:, None, None, None]
        for b in pred_b_non_interaction_subj_ls:
            pred_b = tf.compat.v1.depthwise_conv2d(
                b,
                filter=subj_sa_isolate_transition_function,
                strides=[1, 1, 1, 1],
                padding='SAME')
            pred_b_a = tf.reduce_sum(
                tf.math.multiply(pred_b, subj_a_idx),
                axis=-1,
                keepdims=False)
            pred_b_non_interaction.append(pred_b_a)
        pred_b_non_interaction = tf.stack(
            pred_b_non_interaction,
            axis=-1,
            name='stack_b_non_interact_over_obj')
        pred_b_non_interaction = tf.reshape(
            pred_b_non_interaction,
            shape=pred_b_non_interaction_obj.get_shape().as_list()[:-1])

        # Combine the predicted belief for interaction and
        # non-interaction.
        pred_belief = pred_b_interaction + pred_b_non_interaction

        return pred_belief

    @staticmethod
    def belief_correction(
            pred_belief,
            subj_ma_observation_function,
            subj_action,
            num_action,
            subj_observation,
            num_observation):
        """

        :param pred_belief:
        :param subj_ma_observation_function:
        :param subj_action:
        :param num_action:
        :param subj_observation:
        :param num_observation:
        """
        # Input action -> action index vector
        action_index_vector = BeliefFilterNet.get_action_index_vector(
            action=subj_action,
            num_action=num_action)
        action_index_vector = action_index_vector[:, None, None, None, None]
        # Input local observation -> local observation index vector
        observation_index_vector = BeliefFilterNet.get_observation_index_vector(
            observation=subj_observation,
            num_observation=num_observation)
        observation_index_vector = \
            observation_index_vector[:, None, None, None, None, None]

        # Compute the conditional probability distribution
        # over local observations given each of the subjective
        # agent's single-agent state.
        observation_weights = tf.reduce_sum(
            tf.math.multiply(
                subj_ma_observation_function,
                observation_index_vector),
            axis=-1,
            keepdims=False)
        observation_weights = tf.reduce_sum(
            tf.math.multiply(
                observation_weights,
                action_index_vector),
            axis=-1,
            keepdims=False)
        # Update the subjective agent's belief over
        # its single-agent states with its local
        # observation, where we weight the previously-
        # predicted belief by the conditional distribution
        # computed above.
        corr_belief = tf.math.multiply(
            pred_belief,
            observation_weights)
        # Normalization over subjective agent's single-agent
        # state space.
        corr_belief = tf.math.divide(
            corr_belief,
            tf.reduce_sum(
                corr_belief,
                axis=[1, 2, 3, 4],
                keepdims=True) + 1e-10)

        return corr_belief

    @staticmethod
    def belief_update(
            params,
            belief,
            subj_action,
            obj_action,
            subj_observation,
            subj_ma_observation_function,
            ma_transition_interaction_indicator,
            subj_sa_isolate_transition_function=None,
            obj_sa_isolate_transition_function=None):
        """

        :param params
        :param belief:
        :param subj_action:
        :param obj_action:
        :param subj_observation:
        :param subj_ma_observation_function:
        :param subj_sa_isolate_transition_function:
        :param obj_sa_isolate_transition_function:
        :param ma_transition_interaction_indicator:
        """
        if subj_sa_isolate_transition_function is None:
            subj_sa_isolate_transition_function = \
                SIPOMDPLiteFrames.get_sa_transition_function4filtering(
                    name='subj_sa_isolate_trans_func',
                    params=params)
        else:
            subj_sa_isolate_transition_function = \
                SIPOMDPLiteFrames.process_loaded_sa_transition_function4filtering(
                    kernel_weights=subj_sa_isolate_transition_function,
                    params=params)
        if obj_sa_isolate_transition_function is None:
            obj_sa_isolate_transition_function = \
                SIPOMDPLiteFrames.get_sa_transition_function4filtering(
                    name='obj_sa_isolate_trans_func',
                    params=params)
        else:
            obj_sa_isolate_transition_function = \
                SIPOMDPLiteFrames.process_loaded_sa_transition_function4filtering(
                    kernel_weights=obj_sa_isolate_transition_function,
                    params=params)

        pred_belief = BeliefFilterNet.belief_prediction(
            belief=belief,
            subj_sa_isolate_transition_function=subj_sa_isolate_transition_function,
            obj_sa_isolate_transition_function=obj_sa_isolate_transition_function,
            ma_transition_interaction_indicator=ma_transition_interaction_indicator,
            subj_action=subj_action,
            obj_action=obj_action,
            num_action=params.num_action)
        corr_belief = BeliefFilterNet.belief_correction(
            pred_belief=pred_belief,
            subj_ma_observation_function=subj_ma_observation_function,
            subj_action=subj_action,
            num_action=params.num_action,
            subj_observation=subj_observation,
            num_observation=params.num_obs)

        return corr_belief
