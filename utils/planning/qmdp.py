import time
import numpy as np
import scipy.sparse
import libs.mdptoolbox as mdptoolbox

try:
    import ipdb as pdb
except ImportError:
    import pdb


class QMDP:
    def __init__(
            self,
            params=None,
            grid=None):
        self.T = None
        self.R = None
        self.O = None
        self.Q = None
        self.V = None
        self.b0 = None

        self.issparse = False

        if params:
            self.grid_shape = [params.grid_n, params.grid_m]
            self.num_state = params.num_state
            self.num_action = params.num_action
            self.num_obs = params.num_obs
            self.moves = params.moves
            self.discount = params.discount
            self.params = params

        if grid is not None:
            self.grid = grid

    def solve(self):
        self.V = self.compute_V(max_iter=10000)
        self.Q = self.compute_Q(self.V)

    def compute_V(
            self,
            max_iter=10000):
        try:
            vi = mdptoolbox.mdp.ValueIteration(
                transitions=self.T,
                reward=self.R,
                discount=self.discount,
                max_iter=max_iter,
                skip_check=True)

        except TypeError:
            # try without skip_check
            vi = mdptoolbox.mdp.ValueIteration(
                transitions=self.T,
                reward=self.R,
                discount=self.discount,
                max_iter=max_iter)

        vi.run()

        return vi.V

    def compute_Q(
            self,
            V):
        Q = np.empty((self.num_action, self.num_state))
        for action in range(self.num_action):
            R = self.T[action].multiply(self.R[action]).sum(1).A.squeeze()
            Q[action] = R + self.discount * self.T[action].dot(V)

        return Q

    def qmdp_action(
            self,
            b):
        """
        random_actions: select randomly from actions with near equal values.
        Lowest index by default
        """
        action_values = b.dot(self.Q.transpose()).squeeze()
        equal_actions = np.isclose(
            action_values, action_values.max())
        action = np.random.choice(
            [i for i in range(self.num_action)
             if equal_actions[i]])

        return action

    def transition(
            self,
            state,
            action):
        next_state = self.sparse_choice(self.T[action][state])
        reward = self.R[action][state, next_state]

        return next_state, reward

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
            state,
            action):
        """
        Sample an observation
        :param state: i's private state after taking the action
        :param action: last action
        :return: observation
        """
        pobs = self.O[action][state]
        obs = self.sparse_choice(pobs)

        return obs

    def random_obs_over_belief(
            self,
            pred_belief,
            action):
        """
        Random observation given a belief
        :param pred_belief: belief after taking an action (updated)
        :param action: last action
        :return: observation
        """
        if self.issparse and not scipy.sparse.issparse(pred_belief):
            bprime = scipy.sparse.csr_matrix(pred_belief)

        pobs_given_s = self.O[action]
        pobs = pred_belief.dot(pobs_given_s)

        # normalize
        pobs = pobs / pobs.sum()
        # sample weighted with p_obs
        obs = self.sparse_choice(pobs)

        return obs

    def propagate_act(
            self,
            belief,
            action):
        """
        Propagate belief when taking an action
        :param belief:  belief
        :param action: action of the agent
        :return: updated belief
        """
        belief = scipy.sparse.csr_matrix(belief)
        pred_belief = belief.dot(self.T[action])
        pred_belief /= pred_belief.sum()

        return pred_belief

    def propagate_obs(
            self,
            pred_belief,
            action,
            obs):
        """
        Propagate belief when taking an action
        :param pred_belief:  belief propagated by an action
        :param action: last action that produced the observation
        :param obs: observation
        :return: updated belief
        """
        pred_belief = scipy.sparse.csr_matrix(pred_belief)
        corr_belief = pred_belief.multiply(
            self.O[action][:, obs].transpose())
        corr_belief /= corr_belief.sum()

        return corr_belief

    def belief_update(
            self,
            belief,
            action,
            next_state=None):
        """
        Update belief with action. Sample an observation for the current state.
        If state is not specified observation is sampled according to the belief.
        :param belief: belief
        :param action: action of the higher-level agent
        :param next_state: state after executing the action
        Return: bprime (belief after taking action), observation, belief after
        taking action and receiving observation
        """
        pred_belief = self.propagate_act(
            belief=belief,
            action=action)

        # sample observation
        if next_state is None:
            obs = self.random_obs_over_belief(
                pred_belief=pred_belief,
                action=action)
        else:
            obs = self.random_obs(
                state=next_state,
                action=action)

        # update belief with observation
        corr_belief = self.propagate_obs(
            pred_belief=pred_belief,
            action=action,
            obs=obs)

        return pred_belief, obs, corr_belief

    def belief_update_with_obs(
            self,
            belief,
            action,
            obs):
        pred_belief = self.propagate_act(
            belief=belief,
            action=action)
        corr_belief = self.propagate_obs(
            pred_belief=pred_belief,
            action=action,
            obs=obs)

        return pred_belief, corr_belief

    def processR(
            self,
            R):
        # Agent i's single-agent R.
        if isinstance(R, list):
            self.R = [R[a].copy() for a in range(self.num_action)]
        elif isinstance(R, np.ndarray):
            if R.ndim == 3:
                self.R = R.copy()
            elif R.ndim == 2:
                assert self.T is not None
                R = np.array(R)
                self.R = [scipy.sparse.lil_matrix(
                    (self.num_state, self.num_state))
                    for _ in range(self.num_action)]
                for action in range(self.num_action):
                    non0 = self.T[action].nonzero()
                    self.R[action][non0[0], non0[1]] = R[non0[0], action]
            else:
                raise NotImplementedError
        else:
            raise TypeError("R is of the invalid type.")

    def processT(
            self,
            T):
        # Agent i's single-agent T
        if isinstance(T, list):
            self.T = [T[a].copy() for a in range(self.num_action)]
        elif isinstance(T, np.ndarray):
            if T.ndim == 3:
                self.T = T.copy()
            elif T.ndim == 2:
                # (state, action) -> state
                self.T = [scipy.sparse.lil_matrix((self.num_state, self.num_state))
                          for _ in range(self.num_action)]
                for action in range(self.num_action):
                    self.T[action][np.arange(self.num_state), T[:, action].astype('i')] = 1.0
            else:
                raise NotImplementedError
        else:
            raise TypeError("T is of the invalid type.")

    def processO(
            self,
            O):
        if isinstance(O, list):
            self.O = [O[a].copy()
                      for a in range(self.num_action)]
        elif O.ndim == 3:
            # normalize to avoid representation issues when loaded from file
            self.O = O / O.sum(
                axis=2,
                keepdims=True)
        elif O.ndim == 2:
            self.O = scipy.sparse.csr_matrix(O)
        elif O.ndim == 1:
            self.O = scipy.sparse.lil_matrix((self.num_state, self.num_obs))
            self.O[np.arange(self.num_state), O.astype('i')] = 1.0
            self.O = scipy.sparse.csr_matrix(self.O)
        else:
            raise ValueError("O is of the invalid type or shape.")

    def transfer_all_sparse(self):
        self.T = self.transfer_sparse(self.T)
        self.R = self.transfer_sparse(self.R)
        self.O = self.transfer_sparse(self.O)
        self.issparse = True

    def transfer_sparse(
            self,
            mat):
        return [scipy.sparse.csr_matrix(mat[a])
                for a in range(self.num_action)]
