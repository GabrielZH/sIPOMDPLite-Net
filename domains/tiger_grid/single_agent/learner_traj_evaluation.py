"""
The expert demonstration trajectory generator
under the imitation learning setting.
"""

import sys
import numpy as np

sys.path.append('/home/gz67063/projects/sipomdplite-net/')

from domains.tiger_grid.single_agent.env_simulator import SATigerGridEnv


# Print setting
np.set_printoptions(threshold=3000)


def simulate_policy(
        policy,
        env_map,
        init_belief,
        processed_init_belief,
        goal,
        processed_goal,
        params,
        init_state,
        init_action=None):
    """

    :param policy:
    :param env_map:
    :param init_belief:
    :param processed_init_belief
    :param goal:
    :param processed_goal
    :param params
    :param init_state:
    :param init_action:
    """
    if init_action is None:
        init_action = params.staynlisten

    expert_state = init_state
    learner_state = init_state
    expert_belief = init_belief.copy()
    learner_belief = init_belief.copy()

    policy.reset(
        env_map=env_map[None],
        goal_map=processed_goal,
        belief=processed_init_belief)

    domain = SATigerGridEnv(params)
    domain.grid = env_map
    domain.build_pomdp(goal)
    qmdp = domain.get_qmdp()
    qmdp.solve()

    expert_actions = list()
    expert_reward_sum = 0.0
    expert_collisions = 0
    expert_wrong_opens = 0
    expert_succ_opens = 0
    expert_failed = False
    learner_actions = list()
    learner_reward_sum = 0.0
    learner_collisions = 0
    learner_wrong_opens = 0
    learner_succ_opens = 0
    learner_failed = False
    step_i = 0

    while True:
        if step_i >= params.max_traj_len \
                or expert_succ_opens >= 3:
            if expert_succ_opens < 1:
                expert_failed = True
            if expert_collisions > 1:
                expert_failed = True
            step_i = 0
            break

            # choose action
        if step_i == 0:
            # dummy first action
            expert_action = init_action
        else:
            expert_action = qmdp.qmdp_action(expert_belief)

        expert_actions.append(expert_action)

        if expert_action == params.dooropen and expert_state != goal:
            expert_wrong_opens += 1

        if expert_action == params.dooropen and expert_state == goal:
            expert_succ_opens += 1

        # simulate action
        expert_state, expert_reward = qmdp.transition(
            state=expert_state,
            action=expert_action)
        # subj_s, obj_s = comm_s
        # print("AGENT i IS AT [%d], AGENT j IS AT [%d]" % (subj_s, obj_s))

        expert_obs = qmdp.random_obs(
            state=expert_state,
            action=expert_action)

        _, expert_belief = qmdp.belief_update_with_obs(
            belief=expert_belief,
            action=expert_action,
            obs=expert_obs)

        if np.array_equal(expert_actions[-2:], [5, 5]) or \
                np.array_equal(expert_actions[-2:], [1, 3]) or \
                np.array_equal(expert_actions[-2:], [3, 1]) or \
                np.array_equal(expert_actions[-2:], [4, 2]) or \
                np.array_equal(expert_actions[-2:], [2, 4]):
            expert_failed = True

        expert_reward_sum += expert_reward

        # count collisions
        if np.isclose(expert_reward, params.R_obst):
            expert_collisions += 1

        step_i += 1

    while True:
        if step_i >= params.max_traj_len \
                or learner_succ_opens >= 3:
            if learner_succ_opens < 1:
                learner_failed = True
            if learner_collisions > 1:
                learner_failed = True
            break

            # choose action
        if step_i == 0:
            # dummy first action
            learner_action = init_action
        else:
            learner_action, learner_belief_from_network = policy.output(
                input_action=learner_action,
                input_observation=learner_obs)

        learner_actions.append(learner_action)

        if learner_action == params.dooropen and learner_state != goal:
            learner_wrong_opens += 1

        if learner_action == params.dooropen and learner_state == goal:
            learner_succ_opens += 1

        # simulate action
        learner_state, learner_reward_i = qmdp.transition(
            state=learner_state,
            action=learner_action)
        # subj_s, obj_s = comm_s
        # print("AGENT i IS AT [%d], AGENT j IS AT [%d]" % (subj_s, obj_s))

        learner_obs = qmdp.random_obs(
            state=learner_state,
            action=learner_action)

        _, learner_belief = qmdp.belief_update_with_obs(
            belief=learner_belief,
            action=learner_action,
            obs=learner_obs)

        if np.array_equal(learner_actions[-2:], [5, 5]) or \
                np.array_equal(learner_actions[-2:], [1, 3]) or \
                np.array_equal(learner_actions[-2:], [3, 1]) or \
                np.array_equal(learner_actions[-2:], [4, 2]) or \
                np.array_equal(learner_actions[-2:], [2, 4]):
            learner_failed = True

        learner_reward_sum += learner_reward_i

        # count collisions
        if np.isclose(learner_reward_i, params.R_obst):
            learner_collisions += 1

        step_i += 1

    return not expert_failed, expert_collisions, \
        expert_wrong_opens, expert_reward_sum, \
        not learner_failed, learner_collisions, \
        learner_wrong_opens, learner_reward_sum
