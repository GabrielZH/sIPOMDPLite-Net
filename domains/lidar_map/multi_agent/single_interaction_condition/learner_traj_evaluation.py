"""
The expert demonstration trajectory generator
under the imitation learning setting.
"""

import sys
import numpy as np

sys.path.append('/home/gz67063/projects/sipomdplite-net/')

from domains.tiger_grid.multi_agent.single_interaction_condition.env_simulator import MATigerGridEnv


# Print setting
np.set_printoptions(threshold=3000)


def simulate_policy(
        policy,
        env_map,
        processed_env_maps,
        init_belief,
        processed_init_belief,
        goal,
        processed_goals,
        params,
        init_state,
        init_action=None):
    """

    :param policy:
    :param env_map:
    :param processed_env_maps
    :param init_belief:
    :param processed_init_belief
    :param goal:
    :param processed_goals
    :param params
    :param init_state:
    :param init_action:
    """
    if init_action is None:
        init_action = params.staynlisten

    expert_lin_comm_s = init_state
    learner_lin_comm_s = init_state
    expert_comm_s = np.unravel_index(
        expert_lin_comm_s,
        (params.num_cell, params.num_cell))
    learner_comm_s = np.unravel_index(
        learner_lin_comm_s,
        (params.num_cell, params.num_cell))
    init_subj_belief, init_obj_belief = init_belief
    init_belief = np.kron(init_subj_belief, init_obj_belief)
    expert_b_i, expert_b_j = init_belief.copy(), init_belief.copy()
    learner_b_i, learner_b_j = init_belief.copy(), init_belief.copy()

    policy.reset(
        env_maps=processed_env_maps,
        goal_maps=processed_goals,
        belief=processed_init_belief)

    domain = MATigerGridEnv(params)
    domain.grid = env_map
    domain.build_sipomdplite(goal)
    maqmdp = domain.get_maqmdp()
    maqmdp.solve()

    expert_actions_i = list()
    expert_reward_sum = 0.0
    expert_collisions = 0
    expert_wrong_opens_i = 0
    expert_wrong_opens_j = 0
    expert_succ_opens_i = 0
    expert_succ_opens_j = 0
    expert_failed = False
    learner_actions_i = list()
    learner_reward_sum = 0.0
    learner_collisions = 0
    learner_wrong_opens_i = 0
    learner_wrong_opens_j = 0
    learner_succ_opens_i = 0
    learner_succ_opens_j = 0
    learner_failed = False
    step_i = 0

    while True:
        if step_i >= params.max_traj_len \
                or expert_succ_opens_i >= 3 \
                or expert_succ_opens_j >= 3:
            if expert_succ_opens_i < 1:
                expert_failed = True
            if expert_collisions > 1:
                expert_failed = True
            step_i = 0
            break

            # choose action
        if step_i == 0:
            # dummy first action
            expert_act_i, expert_act_j = init_action, init_action
        else:
            expert_act_i = maqmdp.maqmdp_action(expert_b_i, identity='subjective')
            expert_act_j = maqmdp.maqmdp_action(expert_b_j, identity='objective')

        expert_actions_i.append(expert_act_i)
        expert_joint_a = [expert_act_i, expert_act_j]
        expert_lin_joint_a = np.ravel_multi_index(
            expert_joint_a,
            (params.num_action, params.num_action))

        if expert_act_i == params.dooropen and expert_comm_s[0] != goal:
            expert_wrong_opens_i += 1

        if expert_act_i == params.dooropen and expert_comm_s[0] == goal:
            expert_succ_opens_i += 1

        if expert_act_j == params.dooropen and expert_comm_s[1] != goal:
            expert_wrong_opens_j += 1

        if expert_act_j == params.dooropen and expert_comm_s[1] == goal:
            expert_succ_opens_j += 1

        # simulate action
        expert_comm_s, expert_lin_comm_s, expert_reward_i = maqmdp.transition(
            comm_s=expert_comm_s,
            lin_comm_s=expert_lin_comm_s,
            joint_a=expert_joint_a,
            lin_joint_a=expert_lin_joint_a)
        # subj_s, obj_s = comm_s
        # print("AGENT i IS AT [%d], AGENT j IS AT [%d]" % (subj_s, obj_s))

        expert_obs_i = maqmdp.random_obs(
            lin_common_s=expert_lin_comm_s,
            lin_joint_act=expert_lin_joint_a,
            identity='subjective')
        expert_obs_j = maqmdp.random_obs(
            lin_common_s=expert_lin_comm_s,
            lin_joint_act=expert_lin_joint_a,
            identity='objective')

        _, expert_b_i = maqmdp.belief_update_with_obs(
            belief=expert_b_i,
            self_priv_a=expert_act_i,
            obs=expert_obs_i)
        _, expert_b_j = maqmdp.belief_update_with_obs(
            belief=expert_b_j,
            self_priv_a=expert_act_j,
            obs=expert_obs_j)

        if np.array_equal(expert_actions_i[-2:], [5, 5]) or \
                np.array_equal(expert_actions_i[-2:], [1, 3]) or \
                np.array_equal(expert_actions_i[-2:], [3, 1]) or \
                np.array_equal(expert_actions_i[-2:], [4, 2]) or \
                np.array_equal(expert_actions_i[-2:], [2, 4]):
            expert_failed = True

        expert_reward_sum += expert_reward_i

        # count collisions
        if np.isclose(expert_reward_i, params.R_obst):
            expert_collisions += 1

        step_i += 1

    while True:
        if step_i >= params.max_traj_len \
                or learner_succ_opens_i >= 3 \
                or learner_succ_opens_j >= 3:
            if learner_succ_opens_i < 1:
                learner_failed = True
            if learner_collisions > 1:
                learner_failed = True
            break

            # choose action
        if step_i == 0:
            # dummy first action
            learner_act_i, learner_act_j = init_action, init_action
        else:
            learner_act_i, _ = policy.output(
                input_action=learner_act_i,
                input_observation=learner_obs_i)
            learner_act_j = maqmdp.maqmdp_action(learner_b_j, identity='objective')

        learner_actions_i.append(learner_act_i)
        learner_joint_a = [learner_act_i, learner_act_j]
        learner_lin_joint_a = np.ravel_multi_index(
            learner_joint_a,
            (params.num_action, params.num_action))

        if learner_act_i == params.dooropen and learner_comm_s[0] != goal:
            learner_wrong_opens_i += 1

        if learner_act_i == params.dooropen and learner_comm_s[0] == goal:
            learner_succ_opens_i += 1

        if learner_act_j == params.dooropen and learner_comm_s[1] != goal:
            learner_wrong_opens_j += 1

        if learner_act_j == params.dooropen and learner_comm_s[1] == goal:
            learner_succ_opens_j += 1

        # simulate action
        learner_comm_s, learner_lin_comm_s, learner_reward_i = maqmdp.transition(
            comm_s=learner_comm_s,
            lin_comm_s=learner_lin_comm_s,
            joint_a=learner_joint_a,
            lin_joint_a=learner_lin_joint_a)
        # subj_s, obj_s = comm_s
        # print("AGENT i IS AT [%d], AGENT j IS AT [%d]" % (subj_s, obj_s))

        learner_obs_i = maqmdp.random_obs(
            lin_common_s=learner_lin_comm_s,
            lin_joint_act=learner_lin_joint_a,
            identity='subjective')
        learner_obs_j = maqmdp.random_obs(
            lin_common_s=learner_lin_comm_s,
            lin_joint_act=learner_lin_joint_a,
            identity='objective')

        _, learner_b_i = maqmdp.belief_update_with_obs(
            belief=learner_b_i,
            self_priv_a=learner_act_i,
            obs=learner_obs_i)
        _, learner_b_j = maqmdp.belief_update_with_obs(
            belief=learner_b_j,
            self_priv_a=learner_act_j,
            obs=learner_obs_j)

        if np.array_equal(learner_actions_i[-2:], [5, 5]) or \
                np.array_equal(learner_actions_i[-2:], [1, 3]) or \
                np.array_equal(learner_actions_i[-2:], [3, 1]) or \
                np.array_equal(learner_actions_i[-2:], [4, 2]) or \
                np.array_equal(learner_actions_i[-2:], [2, 4]):
            learner_failed = True

        learner_reward_sum += learner_reward_i

        # count collisions
        if np.isclose(learner_reward_i, params.R_obst):
            learner_collisions += 1

        step_i += 1

    return not expert_failed, expert_collisions, \
           expert_wrong_opens_i, expert_reward_sum, \
           not learner_failed, learner_collisions, \
           learner_wrong_opens_i, learner_reward_sum
