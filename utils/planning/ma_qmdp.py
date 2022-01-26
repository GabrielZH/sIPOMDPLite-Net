import time
import numpy as np
import scipy.sparse
import libs.mdptoolbox as mdptoolbox

try:
    import ipdb as pdb
except ImportError:
    import pdb


class MAQMDP:
    def __init__(
            self,
            params=None,
            grid=None):
        self.ma_T = None
        self.subj_sa_T = None
        self.expanded_subj_sa_T = None
        self.obj_sa_T = None
        self.expanded_obj_sa_T = None
        self.subj_ma_R = None
        self.subj_sa_R = None
        self.expanded_subj_sa_R = None
        self.obj_ma_R = None
        self.obj_sa_R = None
        self.expanded_obj_sa_R = None
        self.subj_ma_O = None
        self.obj_ma_O = None
        self.sa_O = None
        self.expanded_subj_sa_O = None
        self.expanded_obj_sa_O = None
        self.subj_Q = None
        self.obj_Q = None
        self.subj_V = None
        self.obj_V = None
        self.X_T = None
        self.X_R = None
        self.l0_policy = None
        self.opponent_policy = None
        self.policy_memory = list()
        self.b0 = None

        self.issparse = False

        if params:
            self.grid_shape = [params.grid_n, params.grid_m]
            self.num_cell = params.num_cell
            self.num_state = params.num_state
            self.num_action = params.num_action
            self.num_joint_action = params.num_joint_action
            self.num_obs = params.num_obs
            self.moves = params.moves
            self.discount = params.discount
            self.reason_level = params.reason_level
            self.params = params

        if grid is not None:
            self.grid = grid

    def solve(self):
        self.l0_policy = self.get_l0_policy()
        self.opponent_policy = self.get_mixed_strategy(
            level=self.reason_level - 1)
        self.subj_V = self.compute_subj_V(
            nested_policy=self.opponent_policy,
            max_iter=10000)
        self.subj_Q = self.compute_subj_Q(self.subj_V)

    def get_l0_policy(self):
        return np.ones((self.num_action, self.num_state)) / self.num_action

    def compute_subj_V(
            self,
            nested_policy,
            max_iter=10000):
        try:
            sivi = mdptoolbox.mdp.SparseInteractionValueIteration(
                singleagent_transitions=self.subj_sa_T,
                singleagent_rewards=self.subj_sa_R,
                expanded_singleagent_transitions=self.expanded_subj_sa_T,
                expanded_singleagent_rewards=self.expanded_subj_sa_R,
                multiagent_transitions=self.ma_T,
                multiagent_rewards=self.subj_ma_R,
                transition_interaction_indicator=self.X_T,
                reward_interaction_indicator=self.X_R,
                opponent_policy=nested_policy,
                discount=self.discount,
                max_iter=max_iter,
                identity='subjective',
                skip_check=True)

        except TypeError:
            # try without skip_check
            sivi = mdptoolbox.mdp.SparseInteractionValueIteration(
                singleagent_transitions=self.subj_sa_T,
                singleagent_rewards=self.subj_sa_R,
                expanded_singleagent_transitions=self.expanded_subj_sa_T,
                expanded_singleagent_rewards=self.expanded_subj_sa_R,
                multiagent_transitions=self.ma_T,
                multiagent_rewards=self.subj_ma_R,
                transition_interaction_indicator=self.X_T,
                reward_interaction_indicator=self.X_R,
                opponent_policy=nested_policy,
                discount=self.discount,
                max_iter=max_iter,
                identity='subjective')

        sivi.run()

        return sivi.V

    def comparison(
            self,
            max_iter=10000):
        vi = mdptoolbox.mdp.ValueIteration(
            self.ma_T,
            self.subj_ma_R,
            discount=self.discount,
            max_iter=max_iter,
            skip_check=True)
        vi.run()

        return vi.V

    def compute_obj_V(
            self,
            nested_policy,
            max_iter=10000):
        try:
            sivi = mdptoolbox.mdp.SparseInteractionValueIteration(
                singleagent_transitions=self.obj_sa_T,
                singleagent_rewards=self.obj_sa_R,
                expanded_singleagent_transitions=self.expanded_obj_sa_T,
                expanded_singleagent_rewards=self.expanded_obj_sa_R,
                multiagent_transitions=self.ma_T,
                multiagent_rewards=self.obj_ma_R,
                transition_interaction_indicator=self.X_T,
                reward_interaction_indicator=self.X_R,
                opponent_policy=nested_policy,
                discount=self.discount,
                max_iter=max_iter,
                identity='objective',
                skip_check=True)
        except TypeError:
            sivi = mdptoolbox.mdp.SparseInteractionValueIteration(
                singleagent_transitions=self.obj_sa_T,
                singleagent_rewards=self.obj_sa_R,
                expanded_singleagent_transitions=self.expanded_obj_sa_T,
                expanded_singleagent_rewards=self.expanded_obj_sa_R,
                multiagent_transitions=self.ma_T,
                multiagent_rewards=self.obj_ma_R,
                transition_interaction_indicator=self.X_T,
                reward_interaction_indicator=self.X_R,
                opponent_policy=nested_policy,
                discount=self.discount,
                max_iter=max_iter,
                identity='objective')

        sivi.run()

        return sivi.V

    def compute_subj_Q(
            self,
            subj_V):
        Q = np.empty((self.num_joint_action, self.num_state))
        for action in range(self.num_joint_action):
            ma_R = self.ma_T[action].multiply(self.subj_ma_R[action]).sum(1).A.squeeze()
            Q[action] = ma_R + self.discount * self.ma_T[action].dot(subj_V)

        return Q

    def compute_obj_Q(
            self,
            obj_V):
        Q = np.empty((self.num_joint_action, self.num_state))
        for action in range(self.num_joint_action):
            ma_R = self.ma_T[action].multiply(self.obj_ma_R[action]).sum(1).A.squeeze()
            Q[action] = ma_R + self.discount * self.ma_T[action].dot(obj_V)

        return Q

    def get_mixed_strategy(
            self,
            level=1):
        assert level >= 0, "Level must be larger than or equal to 0."
        if level == 0:
            return self.l0_policy

        nested_policy = self.get_mixed_strategy(level=level - 1)
        self.policy_memory.append(nested_policy)

        if level % 2 == self.reason_level % 2:
            V = self.compute_subj_V(nested_policy=nested_policy, max_iter=10000)
            Q = self.compute_subj_Q(V)
            Q = np.multiply(
                Q.reshape((self.num_action, self.num_action, self.num_state)),
                np.expand_dims(nested_policy, axis=0)).sum(1)
            policy = np.zeros((self.num_action, self.num_state))
            for s in np.arange(self.num_state):
                equal_actions = np.arange(self.num_action)[np.isclose(Q[:, s], Q[:, s].max()).squeeze()]
                policy[equal_actions, s] = 1.0 / equal_actions.shape[0]
        else:
            V = self.compute_obj_V(nested_policy=nested_policy, max_iter=10000)
            Q = self.compute_obj_Q(V)
            if level == self.reason_level - 1:
                self.obj_Q = Q
            Q = np.multiply(
                Q.reshape((self.num_action, self.num_action, self.num_state)),
                np.expand_dims(nested_policy, axis=1)).sum(0)
            policy = np.zeros((self.num_action, self.num_state))
            for s in np.arange(self.num_state):
                equal_actions = np.arange(self.num_action)[np.isclose(Q[:, s], Q[:, s].max()).squeeze()]
                policy[equal_actions, s] = 1.0 / equal_actions.shape[0]

        return policy

    def maqmdp_action(
            self,
            b,
            identity):
        """
        random_actions: select randomly from actions with near equal values.
        Lowest index by default
        """
        if identity == 'subjective':
            Q = np.multiply(
                self.subj_Q.reshape((self.num_action, self.num_action, self.num_state)),
                np.expand_dims(self.opponent_policy, axis=0)).sum(1)
            Q_a = b.dot(Q.transpose()).squeeze()
            equal_actions = np.isclose(Q_a, Q_a.max())
            action = np.random.choice(
                [i for i in range(self.num_action) if equal_actions[i]])

        elif identity == 'objective':
            Q = np.multiply(
                self.obj_Q.reshape((self.num_action, self.num_action, self.num_state)),
                np.expand_dims(self.policy_memory[self.reason_level - 2], axis=1)).sum(0)
            Q_a = b.dot(Q.transpose()).squeeze()
            equal_actions = np.isclose(Q_a, Q_a.max())
            action = np.random.choice(
                [i for i in range(self.num_action) if equal_actions[i]])

        else:
            raise ValueError("The input agent identity does not exist.")

        return action

    def transition(
            self,
            comm_s,
            lin_comm_s,
            joint_a,
            lin_joint_a):
        if self.X_T[0, lin_comm_s]:
            next_lin_comm_s = self.sparse_choice(self.ma_T[lin_joint_a][lin_comm_s])
            next_comm_s = np.unravel_index(next_lin_comm_s, (self.num_cell, self.num_cell))
            subj_reward = self.subj_ma_R[lin_joint_a][lin_comm_s, next_lin_comm_s]
        else:
            subj_s, obj_s = comm_s
            subj_a, obj_a = joint_a
            next_subj_s = self.sparse_choice(self.subj_sa_T[subj_a][subj_s])
            next_obj_s = self.sparse_choice(self.obj_sa_T[obj_a][obj_s])
            next_comm_s = (next_subj_s, next_obj_s)
            next_lin_comm_s = np.ravel_multi_index(next_comm_s, (self.num_cell, self.num_cell))
            subj_reward = self.subj_sa_R[subj_a][subj_s, next_subj_s]

        return next_comm_s, next_lin_comm_s, subj_reward

    def sparse_choice(
            self,
            probs,
            **kwargs):
        if self.issparse:
            if probs.shape[1] == 1:
                vals, _, p = scipy.sparse.find(probs)
            else:
                assert probs.shape[0] == 1
                _, vals, p = scipy.sparse.find(probs)
        else:
            vals = len(probs)
            p = probs

        return np.random.choice(vals, p=p, **kwargs)

    def random_obs(
            self,
            lin_common_s,
            lin_joint_act,
            identity='subjective'):
        """
        Sample an observation
        :param lin_common_s: i's private state after taking the action
        :param lin_joint_act: last action
        :param identity:
        :return: observation
        """
        if identity == "subjective":
            pobs = self.subj_ma_O[lin_joint_act][lin_common_s]
        elif identity == "objective":
            pobs = self.obj_ma_O[lin_joint_act][lin_common_s]
        else:
            raise ValueError("The input agent identity does not exist.")

        obs = self.sparse_choice(pobs)

        return obs

    def random_obs_over_belief(
            self,
            bprime,
            lin_joint_act,
            identity='subjective'):
        """
        Random observation given a belief
        :param bprime: belief after taking an action (updated)
        :param lin_joint_act: last action
        :param identity:
        :return: observation
        """
        if self.issparse and not scipy.sparse.issparse(bprime):
            bprime = scipy.sparse.csr_matrix(bprime)

        if identity == 'subjective':
            pobs_given_s = self.subj_ma_O[lin_joint_act]
            pobs = bprime.dot(pobs_given_s)
        elif identity == 'objective':
            pobs_given_s = self.obj_ma_O[lin_joint_act]
            pobs = bprime.dot(pobs_given_s)
        else:
            raise ValueError("The input agent identity does not exist.")

        # normalize
        pobs = pobs / pobs.sum()
        # sample weighted with p_obs
        obs = self.sparse_choice(pobs)

        return obs

    def propagate_act(
            self,
            belief,
            self_priv_a,
            identity='subjective'):
        """
        Propagate belief when taking an action
        :param belief:  belief
        :param self_priv_a: action of the level-0 agent
        :param identity:
        :return: updated belief
        """
        belief = scipy.sparse.csr_matrix(belief)
        pred_belief = scipy.sparse.csr_matrix(
            np.zeros(self.num_state, dtype=np.float32))
        if identity == "subjective":
            for opponent_priv_a in range(self.num_action):
                joint_a = np.ravel_multi_index(
                    (self_priv_a, opponent_priv_a),
                    (self.num_action, self.num_action))
                pred_belief += belief.multiply(
                    self.opponent_policy[opponent_priv_a]).dot(self.ma_T[joint_a])
        elif identity == "objective":
            for opponent_priv_a in range(self.num_action):
                joint_a = np.ravel_multi_index(
                    (opponent_priv_a, self_priv_a), (self.num_action, self.num_action))
                pred_belief += belief.multiply(
                    self.opponent_policy[opponent_priv_a]).dot(self.ma_T[joint_a])
        else:
            raise ValueError("The input agent identity does not exist.")

        pred_belief /= pred_belief.sum()

        return pred_belief

    def propagate_obs(
            self,
            pred_belief,
            self_priv_a,
            obs,
            identity='subjective'):
        """
        Propagate belief when taking an action
        :param pred_belief:  belief propagated by an action
        :param self_priv_a: last action that produced the observation
        :param obs: observation
        :param identity:
        :return: updated belief
        """
        pred_belief = scipy.sparse.csr_matrix(pred_belief)
        corr_belief = scipy.sparse.csr_matrix(
            np.zeros(self.num_state, dtype=np.float32))
        if identity == "subjective":
            for opponent_priv_a in range(self.num_action):
                joint_a = np.ravel_multi_index(
                    (self_priv_a, opponent_priv_a),
                    (self.num_action, self.num_action))
                corr_belief += pred_belief.multiply(self.subj_ma_O[joint_a][:, obs].transpose())
        elif identity == "objective":
            for opponent_priv_a in range(self.num_action):
                joint_a = np.ravel_multi_index(
                    (opponent_priv_a, self_priv_a),
                    (self.num_action, self.num_action))
                corr_belief += pred_belief.multiply(self.obj_ma_O[joint_a][:, obs].transpose())
        else:
            raise ValueError("The input agent identity does not exist.")

        corr_belief /= corr_belief.sum()

        return corr_belief

    def belief_update(
            self,
            belief,
            self_priv_a,
            others_priv_a,
            next_lin_comm_s=None,
            identity='subjective'):
        """
        Update belief with action. Sample an observation for the current state.
        If state is not specified observation is sampled according to the belief.
        :param belief: belief
        :param self_priv_a: action of the higher-level agent
        :param others_priv_a:
        :param next_lin_comm_s: state after executing the action
        :param identity:
        Return: bprime (belief after taking action), observation, belief after
        taking action and receiving observation
        """
        pred_belief = self.propagate_act(belief, self_priv_a)

        # sample observation
        lin_joint_act = np.ravel_multi_index(
            (self_priv_a, others_priv_a),
            (self.num_action, self.num_action))
        if next_lin_comm_s is None:
            obs = self.random_obs_over_belief(
                bprime=pred_belief,
                lin_joint_act=lin_joint_act,
                identity=identity)
        else:
            obs = self.random_obs(
                lin_common_s=next_lin_comm_s,
                lin_joint_act=lin_joint_act,
                identity=identity)

        # update belief with observation
        corr_belief = self.propagate_obs(pred_belief, self_priv_a, obs)

        return pred_belief, obs, corr_belief

    def belief_update_with_obs(
            self,
            belief,
            self_priv_a,
            obs):
        pred_belief = self.propagate_act(belief, self_priv_a)
        corr_belief = self.propagate_obs(pred_belief, self_priv_a, obs)
        return pred_belief, corr_belief

    def processR(
            self,
            subj_sa_R,
            subj_ma_R,
            obj_sa_R,
            obj_ma_R,
            subj_sa_T,
            obj_sa_T):
        # Agent i's single-agent R.
        if isinstance(subj_sa_R, list) and \
                isinstance(obj_sa_R, list):
            self.subj_sa_R = \
                [subj_sa_R[a].copy()
                 for a in range(self.num_action)]
            self.expanded_subj_sa_R = list()
            if isinstance(obj_sa_T, list):
                for subj_a in range(self.num_action):
                    for obj_a in range(self.num_action):
                        self.expanded_subj_sa_R.append(
                            scipy.sparse.kron(
                                subj_sa_R[subj_a],
                                (obj_sa_T[obj_a] != 0)))
            elif isinstance(obj_sa_T, np.ndarray):
                for subj_a in range(self.num_action):
                    for obj_a in range(self.num_action):
                        self.expanded_subj_sa_R.append(
                            scipy.sparse.kron(
                                subj_sa_R[subj_a],
                                (scipy.sparse.csr_matrix(obj_sa_T[obj_a] != 0))))
            else:
                raise TypeError("A transition function can only be a list "
                                "of sparse matrices or an N-d array.")

            self.obj_sa_R = \
                [obj_sa_R[a].copy()
                 for a in range(self.num_action)]
            self.expanded_obj_sa_R = list()
            if isinstance(subj_sa_T, list):
                for subj_a in range(self.num_action):
                    for obj_a in range(self.num_action):
                        self.expanded_obj_sa_R.append(
                            scipy.sparse.kron(
                                obj_sa_R[obj_a],
                                (subj_sa_T[subj_a] != 0)))
            elif isinstance(subj_sa_T, np.ndarray):
                for subj_a in range(self.num_action):
                    for obj_a in range(self.num_action):
                        self.expanded_obj_sa_R.append(
                            scipy.sparse.kron(
                                obj_sa_R[obj_a],
                                (scipy.sparse.csr_matrix(
                                    subj_sa_T[subj_a] != 0))))
            else:
                raise TypeError("A transition function can only be a "
                                "list of sparse matrices or an N-d array.")

        elif isinstance(subj_sa_R, np.ndarray) and \
                isinstance(obj_sa_R, np.ndarray):
            if subj_sa_R.ndim == 3 and obj_sa_R.ndim == 3:
                if isinstance(obj_sa_T, list):
                    pass
                elif isinstance(obj_sa_T, np.ndarray):
                    pass
                else:
                    raise TypeError("A transition function can only be a "
                                    "list of sparse matrices or an N-d array.")
            else:
                raise NotImplementedError
        else:
            raise TypeError("The data type of the input reward functions "
                            "from the two agents should be consistent.")

        # Agent i's multiagent R.
        if isinstance(subj_ma_R, list) and \
                isinstance(obj_ma_R, list):
            self.subj_ma_R = \
                [subj_ma_R[a].copy()
                 for a in range(self.num_joint_action)]
            self.obj_ma_R = \
                [obj_ma_R[a].copy()
                 for a in range(self.num_joint_action)]
        elif isinstance(subj_ma_R, np.ndarray) and \
                isinstance(obj_ma_R, np.ndarray):
            if subj_ma_R.ndim == 3:
                self.subj_ma_R = subj_ma_R.copy()
                self.obj_ma_R = obj_ma_R.copy()
            elif subj_ma_R.ndim == 2:
                assert self.ma_T is not None

                self.subj_ma_R = \
                    [scipy.sparse.lil_matrix(
                        (self.num_state, self.num_state))
                     for _ in range(self.num_joint_action)]
                for action in range(self.num_joint_action):
                    non0 = self.ma_T[action].nonzero()
                    self.subj_ma_R[action][non0[0], non0[1]] = subj_ma_R[non0[0], action]

                self.obj_ma_R = \
                    [scipy.sparse.lil_matrix(
                        (self.num_state, self.num_state))
                     for _ in range(self.num_joint_action)]
                for action in range(self.num_joint_action):
                    non0 = self.ma_T[action].nonzero()
                    self.obj_ma_R[action][non0[0], non0[1]] = obj_ma_R[non0[0], action]
            else:
                raise NotImplementedError
        else:
            raise TypeError("The data type of the input reward functions from the two "
                            "agents should be consistent.")

    def processT(
            self,
            subj_sa_T,
            obj_sa_T,
            ma_T):
        # Agent i's single-agent T
        if isinstance(subj_sa_T, list) and \
                isinstance(obj_sa_T, list):
            self.subj_sa_T = \
                [subj_sa_T[a].copy()
                 for a in range(self.num_action)]
            self.expanded_subj_sa_T = list()
            for subj_a in range(self.num_action):
                for obj_a in range(self.num_action):
                    self.expanded_subj_sa_T.append(
                        scipy.sparse.kron(
                            subj_sa_T[subj_a],
                            obj_sa_T[obj_a]))
            self.obj_sa_T = \
                [obj_sa_T[a].copy()
                 for a in range(self.num_action)]
            self.expanded_obj_sa_T = self.expanded_subj_sa_T.copy()
        elif isinstance(obj_sa_T, np.ndarray) and \
                isinstance(subj_sa_T, np.ndarray):
            if subj_sa_T.ndim == 3:
                self.subj_sa_T = subj_sa_T.copy()
                self.obj_sa_T = obj_sa_T.copy()
                self.expanded_subj_sa_T = np.kron(subj_sa_T, obj_sa_T)
                self.expanded_obj_sa_T = self.expanded_subj_sa_T.copy()
            elif subj_sa_T.ndim == 2:
                # (state, action) -> state
                self.subj_sa_T = \
                    [scipy.sparse.lil_matrix((self.num_cell, self.num_cell))
                     for _ in range(self.num_action)]
                for action in range(self.num_action):
                    self.subj_sa_T[action][np.arange(self.num_cell), subj_sa_T[:, action].astype('i')] = 1.0
            else:
                raise NotImplementedError
        else:
            raise TypeError("The data type of the input transition functions from the two "
                            "agents should be consistent.")

        # Both agent's common T
        if isinstance(ma_T, list):
            self.ma_T = \
                [ma_T[a].copy()
                 for a in range(self.num_joint_action)]
        elif ma_T.ndim == 3:
            self.ma_T = ma_T.copy()
        elif ma_T.ndim == 2:
            # (state, action) -> state
            self.ma_T = \
                [scipy.sparse.lil_matrix((self.num_state, self.num_state))
                 for _ in range(self.num_joint_action)]
            for action in range(self.num_joint_action):
                self.ma_T[action][np.arange(self.num_cell), ma_T[:, action].astype('i')] = 1.0
        else:
            assert False

    def processO(
            self,
            sa_O,
            expanded_subj_sa_O,
            expanded_obj_sa_O,
            subj_ma_O,
            obj_ma_O):
        if isinstance(subj_ma_O, list):
            self.subj_ma_O = \
                [subj_ma_O[a].copy()
                 for a in range(self.num_joint_action)]
        elif subj_ma_O.ndim == 3:
            # normalize to avoid representation issues when loaded from file
            self.subj_ma_O = \
                subj_ma_O / subj_ma_O.sum(
                    axis=2,
                    keepdims=True)
        elif subj_ma_O.ndim == 2:
            self.subj_ma_O = scipy.sparse.csr_matrix(subj_ma_O)
        elif subj_ma_O.ndim == 1:
            self.subj_ma_O = scipy.sparse.lil_matrix(
                (self.num_state, self.num_obs))
            self.subj_ma_O[np.arange(self.num_state),
                           subj_ma_O.astype('i')] = 1.0
            self.subj_ma_O = scipy.sparse.csr_matrix(self.subj_ma_O)
        else:
            raise ValueError("subj_ma_O is of the invalid type or shape.")

        if isinstance(obj_ma_O, list):
            self.obj_ma_O = \
                [obj_ma_O[a].copy()
                 for a in range(self.num_joint_action)]
        elif obj_ma_O.ndim == 3:
            self.obj_ma_O = obj_ma_O / obj_ma_O.sum(
                axis=2,
                keepdims=True)
        elif obj_ma_O.ndim == 2:
            self.obj_ma_O = scipy.sparse.csr_matrix(obj_ma_O)
        elif obj_ma_O.ndim == 1:
            self.obj_ma_O = scipy.sparse.lil_matrix(
                (self.num_state, self.num_obs))
            self.obj_ma_O[np.arange(self.num_state), obj_ma_O.astype('i')] = 1.0
            self.obj_ma_O = scipy.sparse.csr_matrix(self.obj_ma_O)
        else:
            raise ValueError("obj_ma_O is of the invalid type or shape.")

        if isinstance(sa_O, list):
            self.sa_O = [sa_O[a].copy() for a in range(self.num_action)]
        elif sa_O.ndim == 3:
            # normalize to avoid representation issues when loaded from file
            self.sa_O = sa_O / sa_O.sum(
                axis=2,
                keepdims=True)
        elif sa_O.ndim == 2:
            self.sa_O = scipy.sparse.csr_matrix(sa_O)
        elif sa_O.ndim == 1:
            self.sa_O = scipy.sparse.lil_matrix(
                (self.num_cell, self.num_obs))
            self.sa_O[np.arange(self.num_cell),
                      sa_O.astype('i')] = 1.0
            self.sa_O = scipy.sparse.csr_matrix(self.sa_O)
        else:
            raise ValueError("sa_O is of the invalid type or shape.")

        if isinstance(expanded_subj_sa_O, list):
            self.expanded_subj_sa_O = \
                [expanded_subj_sa_O[a].copy()
                 for a in range(self.num_action)]
        elif expanded_subj_sa_O.ndim == 3:
            # normalize to avoid representation issues when loaded from file
            self.expanded_subj_sa_O = \
                expanded_subj_sa_O / expanded_subj_sa_O.sum(
                    axis=2,
                    keepdims=True)
        elif expanded_subj_sa_O.ndim == 2:
            self.expanded_subj_sa_O = \
                scipy.sparse.csr_matrix(expanded_subj_sa_O)
        elif expanded_subj_sa_O.ndim == 1:
            self.expanded_subj_sa_O = \
                scipy.sparse.lil_matrix((self.num_cell, self.num_obs))
            self.expanded_subj_sa_O[np.arange(self.num_cell),
                                    expanded_subj_sa_O.astype('i')] = 1.0
            self.expanded_subj_sa_O = \
                scipy.sparse.csr_matrix(self.expanded_subj_sa_O)
        else:
            raise ValueError("sa_O is of the invalid type or shape.")

        if isinstance(expanded_obj_sa_O, list):
            self.expanded_obj_sa_O = \
                [expanded_obj_sa_O[a].copy()
                 for a in range(self.num_action)]
        elif expanded_obj_sa_O.ndim == 3:
            # normalize to avoid representation issues when loaded from file
            self.expanded_obj_sa_O = \
                expanded_obj_sa_O / expanded_obj_sa_O.sum(
                    axis=2,
                    keepdims=True)
        elif expanded_obj_sa_O.ndim == 2:
            self.expanded_obj_sa_O = \
                scipy.sparse.csr_matrix(expanded_obj_sa_O)
        elif expanded_obj_sa_O.ndim == 1:
            self.expanded_obj_sa_O = \
                scipy.sparse.lil_matrix((self.num_cell, self.num_obs))
            self.expanded_obj_sa_O[np.arange(self.num_cell),
                                   expanded_obj_sa_O.astype('i')] = 1.0
            self.expanded_obj_sa_O = \
                scipy.sparse.csr_matrix(self.expanded_obj_sa_O)
        else:
            raise ValueError("sa_O is of the invalid type or shape.")

    def getX_TR(
            self,
            X_T,
            X_R):
        self.X_T = X_T
        self.X_R = X_R

    def transfer_all_sparse(self):
        self.ma_T = self.transfer_sparse(self.ma_T)
        self.subj_sa_T = self.transfer_sparse(self.subj_sa_T)
        self.obj_sa_T = self.transfer_sparse(self.obj_sa_T)
        self.subj_ma_R = self.transfer_sparse(self.subj_ma_R)
        self.subj_sa_R = self.transfer_sparse(self.subj_sa_R)
        self.obj_ma_R = self.transfer_sparse(self.obj_ma_R)
        self.obj_sa_R = self.transfer_sparse(self.obj_sa_R)
        self.subj_ma_O = self.transfer_sparse(self.subj_ma_O)
        self.obj_ma_O = self.transfer_sparse(self.obj_ma_O)
        self.sa_O = self.transfer_sparse(self.sa_O)
        self.expanded_subj_sa_O = self.transfer_sparse(self.expanded_subj_sa_O)
        self.expanded_obj_sa_O = self.transfer_sparse(self.expanded_obj_sa_O)
        self.X_T = self.transfer_sparse(self.X_T)
        self.X_R = self.transfer_sparse(self.X_R)
        self.issparse = True

    def transfer_sparse(
            self,
            mat):
        if type(mat) == np.ndarray:
            if mat.shape[0] == self.num_action:
                return [scipy.sparse.csr_matrix(mat[a]) for a in range(self.num_action)]
            elif mat.shape[0] == self.num_joint_action:
                return [scipy.sparse.csr_matrix(mat[a]) for a in range(self.num_joint_action)]
            elif mat.shape[0] == self.num_state:
                assert mat.ndim == 1, "Only indicator functions (X_T, X_R) have their first dimensions states."
                return scipy.sparse.csr_matrix(mat)
            else:
                raise ValueError("The first dimension of the input N-d array is required to represent actions.")
        elif type(mat) == list or type(mat) == tuple:
            if len(mat) == self.num_action:
                return [scipy.sparse.csr_matrix(mat[a]) for a in range(self.num_action)]
            elif len(mat) == self.num_joint_action:
                return [scipy.sparse.csr_matrix(mat[a]) for a in range(self.num_joint_action)]
            elif mat.getnnz() == self.num_state:
                return scipy.sparse.csr_matrix(mat)
            else:
                raise ValueError("The first dimension of the input sparse matrix is required to represent actions.")
        else:
            raise ValueError("The input data structure is of the invalid type.")
