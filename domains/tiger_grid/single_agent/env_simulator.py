import numpy as np
import scipy.sparse


from utils.envs import dijkstra
from utils.planning.qmdp import QMDP


# Print setting
np.set_printoptions(threshold=3000)

# Grid env representation
FREESPACE = 1.0
OBSTACLE = 0.0


class SATigerGridEnv(object):
    def __init__(
            self,
            params):
        """
        Initialize a tiger-grid env simulator.
        :param params: domain descriptor dotdict
        """
        self.params = params

        self.N = params.grid_n
        self.M = params.grid_m
        self.grid_shape = [self.N, self.M]
        self.moves = params.moves
        self.cardinal_dirs = params.cardinal_dirs
        self.rim_dirs = params.rim_dirs

        self.num_action = params.num_action
        self.num_obs = params.num_obs
        self.obs_dim = len(self.cardinal_dirs)
        self.num_obs = params.num_obs
        self.num_state = params.num_state

        self.grid = None
        self.O = None
        self.T = None
        self.R = None

        self.graph = None

    def random_instance(
            self,
            generate_grid=True):
        """
        Generate a random problem instance for a grid.
        Picks a random initial belief, initial state and goal states.
        :param generate_grid: generate a new grid and pomdp model if True,
        otherwise use self.grid
        :return:
        """
        params = self.params
        while True:
            if generate_grid:
                self.grid = self.random_grid(
                    params.pr_obst_high,
                    params.pr_obst_low,
                    random_pr_obst=True)
                self.build_path_graph()

            # sample initial belief, start, goal
            while True:
                try:
                    init_belief, init_state, goal = self.gen_init_and_goal()
                    break
                except TypeError:
                    self.grid = self.random_grid(
                        params.pr_obst_high,
                        params.pr_obst_low,
                        random_pr_obst=True)
                    self.build_path_graph()
            # print("Tiger-grid env:")
            # print(self.grid)
            # print("i's initial location:", init_s[0])
            # print("Goal:", goal)

            # Call the function to generate sparse I-POMDP Lite models.
            self.build_pomdp(goal)

            # Create maqmdp models.
            qmdp = self.get_qmdp()
            # it will also convert to csr sparse, and set maqmdp.issparse=True

            return qmdp, init_belief, init_state, goal

    def build_pomdp(
            self,
            goal):
        # The function constructs all necessary sparse I-POMDP Lite models, 
        # including both agents' single-agent models for non-interaction 
        # areas and their multiagent counterparts for interaction areas.
        self.O = self.build_O()
        self.T, self.R = self.build_TR(goal)

    def build_path_graph(self):
        #
        T = self.build_deterministic_trans()
        # transform into graph with opposite directional actions, so we can
        # compute path from goal.
        graph = {i: {} for i in range(self.num_state)}
        for action in range(self.num_action - 1):
            for state in range(self.num_state):
                next_state = T[state, action]
                if state != next_state:
                    graph[next_state][state] = 1  # edge with distance 1
        self.graph = graph

    def gen_init_and_goal(
            self,
            maxtrials=10000):
        """
        Pick an initial belief, initial state and goal state randomly
        """
        free_space = np.nonzero((self.grid == FREESPACE).flatten())[0]
        for _ in range(maxtrials):
            try:
                goal = np.random.choice(free_space)
                goal_reachable_cells = list()
                s0_pool = list()
                outsiders = list()
                D, path_pointers = dijkstra.Dijkstra(self.graph, goal)
                for state in free_space:
                    if state != goal:
                        if state in D:
                            goal_reachable_cells.append(state)
                            if D[state] > self.N / 2:
                                s0_pool.append(state)
                            else:
                                outsiders.append(state)
                if not len(goal_reachable_cells) or not len(s0_pool):
                    continue

                # Randomly selects the belief states, not necessarily the uniform
                # distribution over all free cells.
                b0_size = int(
                    np.random.choice(
                        np.arange(
                            1, len(goal_reachable_cells))))
                if b0_size < len(s0_pool):
                    quota = int(
                        np.random.choice(
                            np.arange(
                                1, b0_size + 1)))  # non-zero beliefs assigned to s0_pool
                else:
                    quota = len(s0_pool)
                idx_in_pool = np.random.choice(s0_pool, quota, replace=False)
                idx_out_pool = np.random.choice(outsiders, b0_size - quota, replace=False)
                b0_idx = np.append(idx_in_pool, idx_out_pool)

                # sanity check
                for state in b0_idx:
                    coord = np.unravel_index(state, (self.N, self.M))
                    assert self.check_free(coord)

                b0 = np.zeros(self.num_state)
                # uniform distribution over sampled cells
                b0[b0_idx] = 1.0 / b0_size
                pr_s0 = np.zeros(self.num_state)
                pr_s0[s0_pool] = 1.0 / len(s0_pool)

                # sample initial state from initial belief
                sample_dist = (pr_s0 * b0) / (pr_s0 * b0).sum()
                s0 = np.random.choice(
                    self.num_state, p=sample_dist)

                if not (s0 in D):
                    raise ValueError(
                        "The agent's initial location "
                        "is not reachable from the goal.")
                else:
                    break
            except ValueError:
                continue

        else:
            return None

        return b0, s0, goal

    def build_O(self):
        """
        The function builds agent i's multiagent observation (ma_O) model w.r.t. 
        common states shared by both agents and their joint actions for a given grid.
        :return: multiagent observation model (ma_O)
        """
        params = self.params
        pr_obs_succ = params.pr_obs_succ

        O = np.zeros(
            [self.num_action,
             self.num_state,
             self.num_obs],
            dtype=np.float32)

        # 1. If agent i selects any mobility actions (e.g., move, open the door, etc.), it will 
        #    perceive information regarding its current location, where the probability of receiving
        #    a certain observation in any cell after executing a mobility action is always uniformly
        #    distributed over all observations.
        O[[a for a in np.arange(1, self.num_action)], :, :] = 1 / self.num_obs

        # 2. Only when agent i selects to stay and to observe, it will recceive 
        #    receive useful but imperfect information to help locating itself.
        # i) The first two for-loops builds i's single-agent obs func (sa_O).
        for i in range(self.N):
            for j in range(self.M):
                state_coord = np.array([i, j])
                state = np.ravel_multi_index(
                    state_coord, self.grid_shape)

                # Build observations, which can be utilized for constructing both 
                # sa_O and ma_O.
                true_obs = np.ones(self.obs_dim)
                for dir in range(self.obs_dim):
                    neighbor_coord = self.apply_move(
                        state_coord,
                        np.array(self.cardinal_dirs[dir]))
                    if not self.check_free(neighbor_coord):
                        true_obs[dir] = 0
                for obs in range(self.num_obs):
                    wrong_obs_dim = np.abs(
                        np.array(
                            np.unravel_index(
                                obs, [2, 2, 2, 2]),
                            dtype=np.int32) - true_obs).sum()
                    pr_obs_given_state = np.power(
                        1.0 - pr_obs_succ, wrong_obs_dim) * \
                                        np.power(
                                            pr_obs_succ, self.obs_dim - wrong_obs_dim)

                    # Fill in with posterior probabilities Pr(o_i^{\prime}|s_i^{\prime}, a_i).
                    O[params.staynlisten, state, obs] = pr_obs_given_state

                    # sanity check
                    assert np.all(
                        np.isclose(1.0, O[action, state, :].sum())
                        for action in np.arange(self.num_action))

        return O

    def build_TR(
            self,
            goal):
        """
        The function builds agents' common transition function (ma_T) and agent 
        i's interactive reward function (subj_ma_R) in a multiagent tiger-grid game.
        """
        params = self.params
        pr_move_succ = params.pr_move_succ

        # 
        T = [scipy.sparse.lil_matrix(
            (self.num_state, self.num_state),
            dtype=np.float32)
            for _ in range(self.num_action)]
        R = [scipy.sparse.lil_matrix(
            (self.num_state, self.num_state),
            dtype=np.float32)
            for _ in range(self.num_action)]

        for i in range(self.N):
            for j in range(self.M):
                coord = np.array([i, j])
                state = np.ravel_multi_index(
                    coord,
                    self.grid_shape)
                freenbs = list()
                for dir in self.rim_dirs:
                    nb_coord = self.apply_move(coord, dir)
                    if self.check_free(nb_coord):
                        freenbs.append(
                            np.ravel_multi_index(
                                nb_coord, self.grid_shape))
                num_freenb = len(freenbs)
                if not num_freenb:
                    freenbs = [state]
                    num_freenb = 1

                for action in range(self.num_action):
                    if action == params.dooropen:
                        if state == goal:
                            T[action][state, freenbs] = 1.0 / num_freenb
                            R[action][state, freenbs] = params.R_open_goal

                        else:
                            T[action][state, state] = 1.0
                            R[action][state, state] = params.R_open_wrong

                    elif action == params.staynlisten:
                        T[action][state, state] = 1.0
                        R[action][state, state] = params.R_listen

                    else:
                        next_coord = self.apply_move(
                            coord,
                            np.array(self.moves[action]))
                        if not self.check_free(next_coord):
                            next_coord = coord.copy()
                        next_state = np.ravel_multi_index(
                            next_coord,
                            self.grid_shape)
                        if next_state == state:
                            T[action][state, next_state] = 1.0
                            R[action][state, next_state] = params.R_obst
                        else:
                            T[action][state, next_state] = pr_move_succ
                            T[action][state, state] = 1.0 - pr_move_succ
                            R[action][state, [state, next_state]] = params.R_move[action]

        return T, R

    def build_deterministic_trans(self):
        # maximum likely versions
        # Tml[s, a] --> next state (except for the open action)
        trans_tab = np.zeros(
            [self.num_state, self.num_action - 1], 'i')

        for i in range(self.N):
            for j in range(self.M):
                state_coord = np.array([i, j])
                state = np.ravel_multi_index(
                    state_coord,
                    (self.N, self.M))

                # build T w.r.t. cells
                for action in range(self.num_action - 1):
                    neighbor_coord = self.apply_move(
                        state_coord,
                        np.array(self.moves[action]))
                    if not self.check_free(neighbor_coord):
                        # cannot move if obstacle or edge of world
                        neighbor_coord = [i, j]

                    neighbor = np.ravel_multi_index(
                        neighbor_coord,
                        (self.N, self.M))
                    trans_tab[state, action] = neighbor

        return trans_tab

    def get_qmdp(self):
        model = QMDP(self.params, self.grid)

        model.processT(self.T)
        model.processR(self.R)
        model.processO(self.O)

        model.transfer_all_sparse()

        return model

    @staticmethod
    def outofbounds(grid, coord):
        return (coord[0] < 0 or coord[0] >= grid.shape[0] or coord[1] < 0 or
                coord[1] >= grid.shape[1])

    @staticmethod
    def apply_move(coord_in, move):
        coord = coord_in.copy()
        coord[:2] += move[:2]
        return coord

    def check_free(self, coord):
        return (not self.outofbounds(
            self.grid, coord) and self.grid[coord[0], coord[1]] != OBSTACLE)

    def random_grid(self, pr_obst_high, pr_obst_low=0.0, random_pr_obst=False):
        rand_field = np.random.rand(self.N, self.M)
        if random_pr_obst:
            pr_obst = np.random.uniform(pr_obst_low, pr_obst_high)
        else:
            pr_obst = pr_obst_high
        grid = (rand_field > pr_obst).astype('i')
        return grid
