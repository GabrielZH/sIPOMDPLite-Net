import sys

import tensorflow as tf
import numpy as np

sys.path.append('/home/gz67063/projects/sipomdplite-net')

from network.models.frames import SIPOMDPLiteFrames
from network.models.belief_update import BeliefFilterNet
from network.models.planning_net import PlanningNet


PRETRAINED_SA_MODEL_PATH = \
    '/home/gz67063/projects/sipomdplite-net/' \
    'pretrained_single_agent_model/' \
    'models/tiger_grid/env_6x6/' \
    'sa_pretrained_model_K24_further/model.chk-0'


class SIPOMDPLiteNet(object):
    """
    The class implements a SIPOMDPLite-net for the multiagent navigation domain.
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
        Create placeholders for all inputs in self.placeholders
        """
        grid_n = self.params.grid_n
        grid_m = self.params.grid_m
        num_action = self.params.num_action
        step_size = self.step_size
        batch_size = self.batch_size

        placeholders = list()
        placeholders.append(
            tf.compat.v1.placeholder(
                dtype=tf.int32,
                shape=(batch_size, 2, grid_n, grid_m),
                name='task_param_env_maps'))

        placeholders.append(
            tf.compat.v1.placeholder(
                tf.int32,
                shape=(batch_size, 2, grid_n, grid_m),
                name='task_param_goal_maps'))

        placeholders.append(
            tf.compat.v1.placeholder(
                tf.float32,
                shape=(batch_size, 2, grid_n, grid_m),
                name='task_param_init_belief'))

        placeholders.append(
            tf.compat.v1.placeholder(
                tf.float32,
                shape=(batch_size,
                       grid_n,
                       grid_m,
                       grid_n,
                       grid_m,
                       num_action),
                name='nested_policy'))

        placeholders.append(
            tf.compat.v1.placeholder(
                tf.float32,
                shape=(batch_size,),
                name='is_traj_head'))

        placeholders.append(
            tf.compat.v1.placeholder(
                tf.int32,
                shape=(step_size, batch_size),
                name='input_subj_action_traj'))  # a_i_{t-1}

        placeholders.append(
            tf.compat.v1.placeholder(
                tf.int32,
                shape=(step_size, batch_size),
                name='input_obj_action_traj'))  # a_j_{t-1}

        placeholders.append(
            tf.compat.v1.placeholder(
                tf.int32,
                shape=(step_size, batch_size),
                name='input_observation_traj'))  # o_t

        placeholders.append(
            tf.compat.v1.placeholder(
                tf.float32,
                shape=(step_size, batch_size),
                name='valid_step_mask'))

        placeholders.append(
            tf.compat.v1.placeholder(
                tf.int32,
                shape=(step_size, batch_size),
                name='label_subj_action_traj'))  # label a_i_t

        placeholders.append(
            tf.compat.v1.placeholder(
                tf.int32,
                shape=(step_size, batch_size),
                name='label_obj_action_traj'))  # label a_j_t

        self.placeholders = placeholders

    def build_inference(
            self,
            reuse=False):
        """
        Creates placeholders, ops for inference and loss
        Unfolds filter and planner through time
        Also creates an op to update the belief. It should be always evaluated
        together with the loss.
        :param reuse: reuse variables if True
        :return: None
        """
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()

        self.build_placeholders()

        env_maps, \
            goal_maps, \
            init_belief, \
            nested_policy, \
            is_traj_head, \
            input_subj_action_traj, \
            input_obj_action_traj, \
            input_observation_traj, \
            valid_step_mask, \
            label_subj_action_traj, \
            label_obj_action_traj = self.placeholders

        # types conversions
        env_maps = tf.cast(
            env_maps,
            dtype=tf.float32)
        goal_maps = tf.cast(
            goal_maps,
            dtype=tf.float32)
        init_belief_over_subj_states, \
            init_belief_over_obj_states = tf.unstack(
                init_belief,
                axis=1)
        init_belief = tf.math.multiply(
            init_belief_over_subj_states[:, :, :, None, None],
            init_belief_over_obj_states[:, None, None])

        nested_policy = tf.expand_dims(
            nested_policy,
            axis=-2)

        is_traj_head = tf.reshape(
            is_traj_head,
            [self.batch_size] + [1] * (init_belief.get_shape().ndims - 1))

        subj_env_map = env_maps[:, 0]
        obj_env_map = env_maps[:, 1]
        joint_env_map = tf.math.multiply(
            subj_env_map[:, :, :, None, None],
            obj_env_map[:, None, None])
        subj_goal_map = goal_maps[:, 0]
        obj_goal_map = goal_maps[:, 1]
        joint_goal_map = tf.math.multiply(
            subj_goal_map[:, :, :, None, None],
            obj_goal_map[:, None, None])

        # pre-compute context, fixed through time.
        with tf.compat.v1.variable_scope('belief_filter_net'):
            if self.params.load_trained_sa_trans_func:
                filter_net_subj_sa_isolate_transition_function = \
                    tf.train.load_variable(
                        PRETRAINED_SA_MODEL_PATH,
                        name='belief_filter_net/trans_func')
                filter_net_obj_sa_isolate_transition_function = \
                    tf.train.load_variable(
                        PRETRAINED_SA_MODEL_PATH,
                        name='belief_filter_net/trans_func')

            if self.params.load_trained_sa_obs_func:
                kernel_weights_layer_0 = tf.train.load_variable(
                    PRETRAINED_SA_MODEL_PATH,
                    name='belief_filter_net/w_subj_obs_func_0')
                bias_layer_0 = tf.train.load_variable(
                    PRETRAINED_SA_MODEL_PATH,
                    name='belief_filter_net/b_subj_obs_func_0')
                kernel_weights_layer_1 = tf.train.load_variable(
                    PRETRAINED_SA_MODEL_PATH,
                    name='belief_filter_net/w_subj_obs_func_1')
                bias_layer_1 = tf.train.load_variable(
                    PRETRAINED_SA_MODEL_PATH,
                    name='belief_filter_net/b_subj_obs_func_1')
                kernel_weights = [kernel_weights_layer_0, kernel_weights_layer_1]
                biases = [bias_layer_0, bias_layer_1]
                subj_sa_observation_function = \
                    SIPOMDPLiteFrames.process_loaded_prior_sa_observation_function(
                        env_map=subj_env_map,
                        kernel_weights=kernel_weights,
                        biases=biases)
            else:
                subj_sa_observation_function = \
                    SIPOMDPLiteFrames.get_sa_observation_function(
                        env_map=subj_env_map,
                        params=self.params)
            subj_ma_observation_function = \
                SIPOMDPLiteFrames.get_ma_observation_function(
                    sa_observation_function=subj_sa_observation_function,
                    params=self.params)
            if self.params.use_prior_trans_interaction_indicator:
                subj_sa_transition_interaction_indicator = \
                    SIPOMDPLiteFrames.get_prior_sa_transition_interaction_indicator(
                        goal_map=subj_goal_map,
                        transition_interactive_action=self.params.dooropen,
                        params=self.params)
                obj_sa_transition_interaction_indicator = \
                    SIPOMDPLiteFrames.get_prior_sa_transition_interaction_indicator(
                        goal_map=obj_goal_map,
                        transition_interactive_action=self.params.dooropen,
                        params=self.params)
            else:
                subj_sa_transition_interaction_indicator = \
                    SIPOMDPLiteFrames.get_sa_transition_interaction_indicator(
                        name='subj_sa_trans_interaction_indicator',
                        goal_map=subj_goal_map,
                        params=self.params)
                obj_sa_transition_interaction_indicator = \
                    SIPOMDPLiteFrames.get_sa_transition_interaction_indicator(
                        name='obj_sa_trans_interaction_indicator',
                        goal_map=obj_goal_map,
                        params=self.params)
            subj_ma_transition_interaction_indicator = \
                SIPOMDPLiteFrames.get_subj_ma_interaction_indicator(
                    subj_sa_interaction_indicator=subj_sa_transition_interaction_indicator,
                    obj_sa_interaction_indicator=obj_sa_transition_interaction_indicator)
            obj_ma_transition_interaction_indicator = \
                SIPOMDPLiteFrames.get_obj_ma_interaction_indicator(
                    subj_sa_interaction_indicator=subj_sa_transition_interaction_indicator,
                    obj_sa_interaction_indicator=obj_sa_transition_interaction_indicator)
            common_ma_transition_interaction_indicator = \
                SIPOMDPLiteFrames.get_common_ma_interaction_indicator(
                    subj_sa_interaction_indicator=subj_sa_transition_interaction_indicator,
                    obj_sa_interaction_indicator=obj_sa_transition_interaction_indicator)
            # Process transition interaction indicators
            ma_transition_interaction_indicator = \
                subj_ma_transition_interaction_indicator + \
                obj_ma_transition_interaction_indicator - \
                common_ma_transition_interaction_indicator

        with tf.compat.v1.variable_scope('planning_net'):
            if self.params.load_trained_sa_trans_func:
                planning_net_subj_sa_isolate_transition_function = \
                    tf.train.load_variable(
                        PRETRAINED_SA_MODEL_PATH,
                        name='planning_net/trans_func')
                planning_net_obj_sa_isolate_transition_function = \
                    tf.train.load_variable(
                        PRETRAINED_SA_MODEL_PATH,
                        name='planning_net/trans_func')
            if self.params.use_prior_rwd_interaction_indicator:
                subj_sa_reward_interaction_indicator = \
                    SIPOMDPLiteFrames.get_prior_sa_reward_interaction_indicator(
                        goal_map=subj_goal_map,
                        reward_interactive_action=self.params.dooropen,
                        params=self.params)
                obj_sa_reward_interaction_indicator = \
                    SIPOMDPLiteFrames.get_prior_sa_reward_interaction_indicator(
                        goal_map=obj_goal_map,
                        reward_interactive_action=self.params.dooropen,
                        params=self.params)
            else:
                subj_sa_reward_interaction_indicator = \
                    SIPOMDPLiteFrames.get_sa_reward_interaction_indicator(
                        name='subj_sa_rwd_interaction_indicator',
                        goal_map=subj_goal_map,
                        params=self.params)
                obj_sa_reward_interaction_indicator = \
                    SIPOMDPLiteFrames.get_sa_reward_interaction_indicator(
                        name='obj_sa_rwd_interaction_indicator',
                        goal_map=obj_goal_map,
                        params=self.params)
            subj_ma_reward_interaction_indicator = \
                SIPOMDPLiteFrames.get_subj_ma_interaction_indicator(
                    subj_sa_interaction_indicator=subj_sa_reward_interaction_indicator,
                    obj_sa_interaction_indicator=obj_sa_reward_interaction_indicator)
            obj_ma_reward_interaction_indicator = \
                SIPOMDPLiteFrames.get_obj_ma_interaction_indicator(
                    subj_sa_interaction_indicator=subj_sa_reward_interaction_indicator,
                    obj_sa_interaction_indicator=obj_sa_reward_interaction_indicator)
            common_ma_reward_interaction_indicator = \
                SIPOMDPLiteFrames.get_common_ma_interaction_indicator(
                    subj_sa_interaction_indicator=subj_sa_reward_interaction_indicator,
                    obj_sa_interaction_indicator=obj_sa_reward_interaction_indicator)
            # Process transition interaction indicators
            ma_reward_interaction_indicator = \
                subj_ma_reward_interaction_indicator + \
                obj_ma_reward_interaction_indicator - \
                common_ma_reward_interaction_indicator

            if self.params.load_trained_sa_rwd_func:
                subj_sa_isolate_reward_function = dict()
                kernel_weights_layer_0 = tf.train.load_variable(
                    PRETRAINED_SA_MODEL_PATH,
                    name='planning_net/w_reward_func_0')
                bias_layer_0 = tf.train.load_variable(
                    PRETRAINED_SA_MODEL_PATH,
                    name='planning_net/b_reward_func_0')
                kernel_weights_layer_1 = tf.train.load_variable(
                    PRETRAINED_SA_MODEL_PATH,
                    name='planning_net/w_reward_func_1')
                bias_layer_1 = tf.train.load_variable(
                    PRETRAINED_SA_MODEL_PATH,
                    name='planning_net/b_reward_func_1')
                kernel_weights = [kernel_weights_layer_0, kernel_weights_layer_1]
                biases = [bias_layer_0, bias_layer_1]
                subj_sa_isolate_reward_function['weights'] = kernel_weights
                subj_sa_isolate_reward_function['biases'] = biases

            Q_values, _ = PlanningNet.value_iteration(
                subj_env_map=subj_env_map,
                subj_goal_map=subj_goal_map,
                joint_env_map=joint_env_map,
                joint_goal_map=joint_goal_map,
                params=self.params,
                ma_reward_interaction_indicator=ma_reward_interaction_indicator,
                ma_transition_interaction_indicator=ma_transition_interaction_indicator,
                subj_sa_isolate_reward_function=subj_sa_isolate_reward_function,
                subj_sa_isolate_transition_function=planning_net_subj_sa_isolate_transition_function,
                obj_sa_isolate_transition_function=planning_net_obj_sa_isolate_transition_function,
                nested_policy=nested_policy)

        # Create variable for hidden belief, equivalent to the hidden state
        # of an RNN.
        self.belief = tf.compat.v1.get_variable(
            name='belief_state',
            initializer=np.zeros(
                init_belief.get_shape().as_list(),
                dtype=np.float32),
            trainable=False)

        # figure out current b. b = b0 if is_start else blast
        belief = init_belief * is_traj_head + self.belief * (1 - is_traj_head)

        pred_action_traj = list()

        for step in range(self.step_size):
            # Belief update
            with tf.compat.v1.variable_scope('belief_filter_net') as step_scope:
                if step >= 1:
                    step_scope.reuse_variables()
                belief = BeliefFilterNet.belief_update(
                    params=self.params,
                    belief=belief,
                    subj_ma_observation_function=subj_ma_observation_function,
                    subj_sa_isolate_transition_function=filter_net_subj_sa_isolate_transition_function,
                    obj_sa_isolate_transition_function=filter_net_obj_sa_isolate_transition_function,
                    ma_transition_interaction_indicator=ma_transition_interaction_indicator,
                    subj_action=input_subj_action_traj[step],
                    obj_action=input_obj_action_traj[step],
                    subj_observation=input_observation_traj[step])

            # MAQMDP planner
            with tf.compat.v1.variable_scope('planning_net') as step_scope:
                if step >= 1:
                    step_scope.reuse_variables()
                pred_action = PlanningNet.maqmdp_policy(
                    q_values=Q_values,
                    belief=belief)
                pred_action_traj.append(pred_action)

        # create op that updates the belief
        self.update_belief_op = self.belief.assign(belief)

        # Compute cross-entropy loss, shape=(step_size, batch_size, num_action)
        pred_action_traj = tf.stack(
            pred_action_traj,
            axis=0)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pred_action_traj,
            labels=label_subj_action_traj)

        # Weight the loss. weights are 0.0 for steps after the end of a
        # trajectory otherwise 1.0.
        loss = loss * valid_step_mask
        loss = tf.reduce_mean(loss, axis=[0, 1], name='xentropy')

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
        # train_op = optimizer.minimize(self.loss, var_list=trainable_variables)

        self.decay_step = decay_step
        self.learning_rate = learning_rate
        self.train_op = train_op
