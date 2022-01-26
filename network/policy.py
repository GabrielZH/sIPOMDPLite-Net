import tensorflow as tf
import numpy as np


class SIPOMDPLiteNetPolicy(object):
    """
    Policy wrapper for IPOMDPLite-net. Implements two functions: reset and eval.
    """
    def __init__(
            self,
            network,
            sess):
        self.network = network
        self.sess = sess

        self.belief = None
        self.nested_policy = None
        self.env_maps = None
        self.goal_maps = None

        assert self.network.batch_size == 1 and self.network.step_size == 1

    def reset(
            self,
            env_maps,
            goal_maps,
            belief,
            nested_policy):
        """

        :param env_maps:
        :param goal_maps:
        :param belief:
        :param nested_policy:
        :return:
        """
        grid_n = self.network.params.grid_n
        grid_m = self.network.params.grid_m
        num_action = self.network.params.num_action

        subj_belief = belief[0]
        obj_belief = belief[1]
        belief = np.multiply(subj_belief[:, :, None, None], obj_belief[None, None])

        self.belief = belief.reshape([1, grid_n, grid_m, grid_n, grid_m])
        self.env_maps = env_maps.reshape([1, 2, grid_n, grid_m])
        self.goal_maps = goal_maps.reshape([1, 2, grid_n, grid_m])
        self.nested_policy = nested_policy.reshape([1, grid_n, grid_m, grid_n, grid_m, num_action])

        self.sess.run(
            tf.compat.v1.assign(
                self.network.belief, self.belief))

    def output(
            self,
            input_subj_action,
            input_obj_action,
            input_observation):
        """

        :param input_subj_action:
        :param input_obj_action:
        :param input_observation:
        :return:
        """
        is_traj_head = np.array([0])
        input_subj_action = input_subj_action.reshape([1, 1])
        input_obj_action = input_obj_action.reshape([1, 1])
        input_observation = \
            input_observation.reshape([1, 1])

        data = [self.env_maps,
                self.goal_maps,
                self.belief,
                self.nested_policy,
                is_traj_head,
                input_subj_action,
                input_obj_action,
                input_observation]
        feed_dict = {self.network.placeholders[i]: data[i]
                     for i in range(len(self.network.placeholders) - 3)}

        # Evaluate SIPOMDPLite-net policy.
        pred_action, _, belief = self.sess.run(
            [self.network.pred_action_traj,
             self.network.update_belief_op,
             self.network.belief],
            feed_dict=feed_dict)
        pred_action = pred_action.flatten().argmax()

        return pred_action, belief
