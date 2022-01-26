import numpy as np
import scipy.sparse


from utils.envs import dijkstra
from utils.planning.ma_qmdp import MAQMDP


# Print setting
np.set_printoptions(threshold=3000)

# Grid env representation
FREESPACE = 1.0
OBSTACLE = 0.0


class MATigerGridEnv(object):
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
        self.num_joint_action = params.num_joint_action
        self.num_obs = params.num_local_obs
        self.obs_dim = len(self.cardinal_dirs)
        self.num_obs = params.num_obs
        self.num_cell = params.num_cell
        self.num_state = params.num_state

        self.grid = None
        self.subj_ma_O = None  # i's obs func with common states and joint actions
        self.obj_ma_O = None
        self.sa_O = None  # common obs func with own states and actions
        self.expanded_subj_sa_O = None
        self.expanded_obj_sa_O = None
        self.ma_T = None  # common trans func with common states and joint actions
        self.subj_sa_T = None  # i's single-agent trans func with own states and actions
        self.obj_sa_T = None  # j's single-agent trans func with own states and actions
        self.subj_ma_R = None  # i's reward func with common states and joint actions
        self.obj_ma_R = None  # j's reward func with common states and joint actions
        self.subj_sa_R = None  # i's single-agent reward func with own states and actions
        self.obj_sa_R = None  # j's single-agent reward func with own states and actions
        self.X_T = None  # indicator function showing the interaction coordinates for trans
        self.X_R = None  # indicator function showing the interaction coordinates for reward

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
                    params.pr_obst_high, params.pr_obst_low, random_pr_obst=True)
                self.build_path_graph()

            # sample initial belief, start, goal
            while True:
                try:
                    b0_comm_s, b0_priv_s, init_s, goal = self.gen_init_and_goal()
                    break
                except TypeError:
                    self.grid = self.random_grid(
                        params.pr_obst_high, params.pr_obst_low, random_pr_obst=True)
                    self.build_path_graph()
            # print("Tiger-grid env:")
            # print(self.grid)
            # print("i's initial location:", init_s[0])
            # print("Goal:", goal)

            # Call the function to generate sparse I-POMDP Lite models.
            self.build_sipomdplite(goal)

            # Create maqmdp models.
            maqmdp = self.get_maqmdp()
            # it will also convert to csr sparse, and set maqmdp.issparse=True

            return maqmdp, b0_comm_s, b0_priv_s, init_s, goal

    def build_sipomdplite(
            self,
            goal):
        # The function constructs all necessary sparse I-POMDP Lite models, 
        # including both agents' single-agent models for non-interaction 
        # areas and their multiagent counterparts for interaction areas.
        self.sa_O, \
            self.expanded_subj_sa_O, \
            self.expanded_obj_sa_O, \
            self.subj_ma_O, \
            self.obj_ma_O = self.build_O()
        self.subj_sa_T, \
            self.ma_T, \
            self.subj_sa_R, \
            self.subj_ma_R, \
            self.obj_ma_R = self.build_TR(goal)
        self.obj_sa_T = self.subj_sa_T.copy()
        self.obj_sa_R = self.subj_sa_R.copy()
        self.X_T, self.X_R = self.getX_TR(goal)

    def build_path_graph(self):
        #
        T = self.build_deterministic_trans()
        # transform into graph with opposite directional actions, so we can
        # compute path from goal.
        graph = {i: {} for i in range(self.num_cell)}
        for action in range(self.num_action - 1):
            for cell in range(self.num_cell):
                next_cell = T[cell, action]
                if cell != next_cell:
                    graph[next_cell][cell] = 1  # edge with distance 1
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
                for cell in free_space:
                    if cell != goal:
                        if cell in D:
                            goal_reachable_cells.append(cell)
                            if D[cell] > self.N / 2:
                                s0_pool.append(cell)
                            else:
                                outsiders.append(cell)
                if not len(goal_reachable_cells) or not len(s0_pool):
                    continue

                # Randomly selects the belief states, not necessarily the uniform
                # distribution over all free cells.
                sa_b0_size = int(np.random.choice(np.arange(1, len(goal_reachable_cells))))
                if sa_b0_size < len(s0_pool):
                    quota = int(np.random.choice(np.arange(1, sa_b0_size + 1)))  # non-zero beliefs assigned to s0_pool
                else:
                    quota = len(s0_pool)
                idx_in_pool = np.random.choice(s0_pool, quota, replace=False)
                idx_out_pool = np.random.choice(outsiders, sa_b0_size - quota, replace=False)
                sa_b0_idx = np.append(idx_in_pool, idx_out_pool)

                # sanity check
                for cell in sa_b0_idx:
                    coord = np.unravel_index(
                        cell,
                        (self.N, self.M))
                    assert self.check_free(coord)

                sa_b0 = np.zeros(self.num_cell)
                # uniform distribution over sampled cells
                sa_b0[sa_b0_idx] = 1.0 / sa_b0_size
                pr_s0 = np.zeros(self.num_cell)
                pr_s0[s0_pool] = 1.0 / len(s0_pool)

                ma_b0_bin = np.array([sa_b0, sa_b0])
                ma_b0_lin = np.array([p0_i * p0_j for p0_i in sa_b0 for p0_j in sa_b0])

                # sample initial state from initial belief
                sample_dist = (pr_s0 * sa_b0) / (pr_s0 * sa_b0).sum()
                ma_s0 = np.random.choice(
                    self.num_cell, 2, p=sample_dist)

                if not (ma_s0[0] in D and ma_s0[1] in D):
                    raise ValueError("At least one agent's initial location is not reachable from the goal.")
                else:
                    break
            except ValueError:
                continue

        else:
            return None

        return ma_b0_lin, ma_b0_bin, ma_s0, goal

    def build_O(self):
        """
        The function builds agent i's multiagent observation (ma_O) model w.r.t. 
        common states shared by both agents and their joint actions for a given grid.
        :return: multiagent observation model (ma_O)
        """
        params = self.params
        pr_obs_succ = params.pr_obs_succ

        sa_O = np.zeros(
            [self.num_action,
             self.num_cell,
             self.num_obs],
            dtype=np.float32)
        subj_ma_O = np.zeros(
            [self.num_joint_action,
             self.num_state,
             self.num_obs],
            dtype=np.float32)
        obj_ma_O = np.zeros(
            [self.num_joint_action,
             self.num_state,
             self.num_obs],
            dtype=np.float32)

        # 1. If agent i selects any mobility actions (e.g., move, open the door, etc.), it will 
        #    perceive information regarding its current location, where the probability of receiving
        #    a certain observation in any cell after executing a mobility action is always uniformly
        #    distributed over all observations.
        sa_O[[a for a in np.arange(1, self.num_action)], :, :] = 1 / self.num_obs
        subj_ma_O[[np.ravel_multi_index(
            (act_subj, act_obj),
            (self.num_action, self.num_action))
                      for act_subj in np.arange(1, self.num_action)
                      for act_obj in np.arange(self.num_action)], :, :] = 1 / self.num_obs
        obj_ma_O[[np.ravel_multi_index(
            (act_subj, act_obj),
            (self.num_action, self.num_action))
                      for act_subj in np.arange(self.num_action)
                      for act_obj in np.arange(1, self.num_action)], :, :] = 1 / self.num_obs

        # 2. Only when agent i selects to stay and to observe, it will recceive 
        #    receive useful but imperfect information to help locating itself.
        # i) The first two for-loops builds i's single-agent obs func (sa_O).
        for i in range(self.N):
            for j in range(self.M):
                cell_coord_subj = np.array([i, j])
                cell_subj = np.ravel_multi_index(cell_coord_subj, self.grid_shape)

                # Build observations, which can be utilized for constructing both 
                # sa_O and ma_O.
                true_subj_obs = np.ones([self.obs_dim])
                for dir in range(self.obs_dim):
                    neighbor_coord_subj = self.apply_move(
                        cell_coord_subj, np.array(self.cardinal_dirs[dir]))
                    if not self.check_free(neighbor_coord_subj):
                        true_subj_obs[dir] = 0
                for obs in range(self.num_obs):
                    wrong_subj_obs_dim = np.abs(
                        np.array(
                            np.unravel_index(
                                obs, [2, 2, 2, 2]),
                            dtype=np.int32) - true_subj_obs).sum()
                    pr_subj_obs_succ = np.power(
                        1.0 - pr_obs_succ, wrong_subj_obs_dim) * \
                                       np.power(
                                           pr_obs_succ,
                                           self.obs_dim - wrong_subj_obs_dim)

                    # Fill in with posterior probabilities Pr(o_i^{\prime}|s_i^{\prime}, a_i).
                    sa_O[params.staynlisten, cell_subj, obs] = pr_subj_obs_succ

                    # sanity check
                    assert np.all(
                        np.isclose(1.0, sa_O[a, cell_subj, :].sum())
                        for a in np.arange(self.num_joint_action))

                    # ii) The last two for-loops builds i's multiagent obs func (ma_O).
                    for k in range(self.N):
                        for l in range(self.M):
                            cell_coord_obj = np.array([k, l])
                            cell_obj = np.ravel_multi_index(cell_coord_obj, self.grid_shape)

                            true_obj_obs = np.ones([self.obs_dim])
                            for dir in range(self.obs_dim):
                                neighbor_coord_obj = self.apply_move(
                                    cell_coord_obj, np.array(self.cardinal_dirs[dir]))
                                if not self.check_free(neighbor_coord_obj):
                                    true_obj_obs[dir] = 0
                            wrong_obj_obs_dim = np.abs(
                                np.array(
                                    np.unravel_index(
                                        obs, [2, 2, 2, 2]),
                                    dtype=np.int32) - true_obj_obs).sum()
                            pr_obj_obs_succ = np.power(
                                1.0 - pr_obs_succ, wrong_obj_obs_dim) * \
                                               np.power(
                                                   pr_obs_succ, self.obs_dim - wrong_obj_obs_dim)

                            state = np.ravel_multi_index(
                                (cell_subj, cell_obj), (self.num_cell, self.num_cell))

                            # Fill in with posterior probabilities Pr(o_i^{\prime}|s^{\prime}, a),
                            # where s^{\prime}=(s_i^{\prime},s_j^{\prime}) is a common state, and 
                            # a=(a_i,a_j) is a joint action.
                            subj_ma_O[[np.ravel_multi_index(
                                (params.staynlisten, act_obj),
                                (self.num_action, self.num_action))
                                          for act_obj in range(self.num_action)], state, obs] = pr_subj_obs_succ

                            obj_ma_O[[np.ravel_multi_index(
                                (act_subj, params.staynlisten),
                                (self.num_action, self.num_action))
                                          for act_subj in range(self.num_action)], state, obs] = pr_obj_obs_succ
                        
                            # sanity check
                            assert np.all(
                                np.isclose(1.0, subj_ma_O[a, state, :].sum())
                                for a in np.arange(self.num_joint_action))
                            assert np.all(
                                np.isclose(1.0, obj_ma_O[a, state, :].sum())
                                for a in np.arange(self.num_joint_action))

        expanded_subj_sa_O = np.tile(
            np.expand_dims(sa_O, axis=2),
            reps=[1, 1, self.num_cell, 1]).reshape(
            (self.num_action, self.num_state, self.num_obs))
        expanded_obj_sa_O = np.tile(
            np.expand_dims(sa_O, axis=1),
            reps=[1, self.num_cell, 1, 1]).reshape(
            (self.num_action, self.num_state, self.num_obs))

        return sa_O, expanded_subj_sa_O, expanded_obj_sa_O, subj_ma_O, obj_ma_O

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
        subj_sa_T = [scipy.sparse.lil_matrix((self.num_cell, self.num_cell), dtype='f')
                     for _ in range(self.num_action)]
        ma_T = [scipy.sparse.lil_matrix((self.num_state, self.num_state), dtype='f')
                for _ in range(self.num_joint_action)]
        subj_sa_R = [scipy.sparse.lil_matrix((self.num_cell, self.num_cell), dtype='f')
                     for _ in range(self.num_action)]
        subj_ma_R = [scipy.sparse.lil_matrix((self.num_state, self.num_state), dtype='f')
                     for _ in range(self.num_joint_action)]
        obj_ma_R = [scipy.sparse.lil_matrix((self.num_state, self.num_state), dtype='f')
                    for _ in range(self.num_joint_action)]

        for i in range(self.N):
            for j in range(self.M):
                subj_coord = np.array([i, j])
                subj_cell = np.ravel_multi_index(subj_coord, self.grid_shape)
                subj_freenbs = list()
                for dir in self.rim_dirs:
                    subj_nb_coord = self.apply_move(subj_coord, dir)
                    if self.check_free(subj_nb_coord):
                        subj_freenbs.append(np.ravel_multi_index(subj_nb_coord, self.grid_shape))
                num_subj_freenb = len(subj_freenbs)
                if not num_subj_freenb:
                    subj_freenbs = [subj_cell]
                    num_subj_freenb = 1

                for subj_a in range(self.num_action):
                    if subj_a == params.dooropen:
                        if subj_cell == goal:
                            subj_sa_T[subj_a][subj_cell, subj_freenbs] = 1.0 / num_subj_freenb
                            subj_sa_R[subj_a][subj_cell, subj_freenbs] = params.R_open_goal

                            for k in range(self.N):
                                for l in range(self.M):
                                    obj_coord = np.array([k, l])
                                    obj_cell = np.ravel_multi_index(obj_coord, self.grid_shape)
                                    common_state = np.ravel_multi_index(
                                        (subj_cell, obj_cell), (self.num_cell, self.num_cell))
                                    obj_freenbs = list()
                                    for dir in self.rim_dirs:
                                        obj_nb_coord = self.apply_move(obj_coord, dir)
                                        if self.check_free(obj_nb_coord):
                                            obj_freenbs.append(np.ravel_multi_index(obj_nb_coord, self.grid_shape))
                                    num_obj_freenb = len(obj_freenbs)
                                    if not num_obj_freenb:
                                        obj_freenbs = [obj_cell]
                                        num_obj_freenb = 1
                                    for obj_a in range(self.num_action):
                                        joint_a = np.ravel_multi_index(
                                            (subj_a, obj_a), (self.num_action, self.num_action))
                                        next_common_state = np.array(
                                            [np.ravel_multi_index((x, y), (self.num_cell, self.num_cell))
                                             for x in subj_freenbs for y in obj_freenbs])
                                        ma_T[joint_a][
                                            common_state, next_common_state] = 1.0 / (num_subj_freenb * num_obj_freenb)
                                        subj_ma_R[joint_a][
                                            common_state, next_common_state] = params.R_open_goal
                                        if obj_a == params.dooropen:
                                            if obj_cell == goal:
                                                obj_ma_R[joint_a][common_state, next_common_state] = params.R_open_goal
                                            else:
                                                obj_ma_R[joint_a][common_state, next_common_state] = params.R_open_wrong
                                        elif obj_a == params.staynlisten:
                                            obj_ma_R[joint_a][common_state, next_common_state] = params.R_listen
                                        else:
                                            obj_ma_R[joint_a][common_state, next_common_state] = params.R_move[obj_a]

                        else:
                            subj_sa_T[subj_a][subj_cell, subj_cell] = 1.0
                            subj_sa_R[subj_a][subj_cell, subj_cell] = params.R_open_wrong

                            for k in range(self.N):
                                for l in range(self.M):
                                    obj_coord = np.array([k, l])
                                    obj_cell = np.ravel_multi_index(obj_coord, self.grid_shape)
                                    common_state = np.ravel_multi_index(
                                        (subj_cell, obj_cell), (self.num_cell, self.num_cell))
                                    obj_freenbs = list()
                                    for dir in self.rim_dirs:
                                        obj_nb_coord = self.apply_move(obj_coord, dir)
                                        if self.check_free(obj_nb_coord):
                                            obj_freenbs.append(np.ravel_multi_index(obj_nb_coord, self.grid_shape))
                                    num_obj_freenb = len(obj_freenbs)
                                    if not num_obj_freenb:
                                        obj_freenbs = [obj_cell]
                                        num_obj_freenb = 1
                                    for obj_a in range(self.num_action):
                                        joint_a = np.ravel_multi_index(
                                            (subj_a, obj_a), (self.num_action, self.num_action))
                                        if obj_a == params.dooropen and obj_cell == goal:
                                            next_common_state = np.array(
                                                [np.ravel_multi_index((x, y), (self.num_cell, self.num_cell))
                                                 for x in subj_freenbs for y in obj_freenbs])
                                            ma_T[joint_a][
                                                common_state, next_common_state] = 1.0 / (num_subj_freenb * num_obj_freenb)
                                            subj_ma_R[joint_a][common_state, next_common_state] = params.R_open_wrong
                                            obj_ma_R[joint_a][common_state, next_common_state] = params.R_open_goal
                                        elif obj_a == params.staynlisten or obj_a == params.dooropen:
                                            next_common_state = common_state
                                            ma_T[joint_a][common_state, next_common_state] = 1.0
                                            subj_ma_R[joint_a][common_state, next_common_state] = params.R_open_wrong
                                            if obj_a == params.staynlisten:
                                                obj_ma_R[joint_a][common_state, next_common_state] = params.R_listen
                                            else:
                                                obj_ma_R[joint_a][common_state, next_common_state] = params.R_open_wrong
                                        else:
                                            obj_next_coord = self.apply_move(obj_coord, np.array(self.moves[obj_a]))
                                            if not self.check_free(obj_next_coord):
                                                obj_next_coord = obj_coord.copy()
                                            obj_next_cell = np.ravel_multi_index(obj_next_coord, self.grid_shape)
                                            subj_next_cell = subj_cell
                                            next_common_state = np.ravel_multi_index(
                                                (subj_next_cell, obj_next_cell), (self.num_cell, self.num_cell))
                                            
                                            if obj_next_cell == obj_cell:
                                                ma_T[joint_a][common_state, next_common_state] = 1.0
                                                subj_ma_R[joint_a][common_state, next_common_state] = params.R_open_wrong
                                                obj_ma_R[joint_a][common_state, next_common_state] = params.R_obst
                                            else:
                                                ma_T[joint_a][common_state, next_common_state] = pr_move_succ
                                                ma_T[joint_a][common_state, common_state] = 1.0 - pr_move_succ
                                                subj_ma_R[joint_a][common_state, [common_state, next_common_state]] = params.R_open_wrong
                                                obj_ma_R[joint_a][common_state, [common_state, next_common_state]] = params.R_move[obj_a]

                    elif subj_a == params.staynlisten:
                        subj_sa_T[subj_a][subj_cell, subj_cell] = 1.0
                        subj_sa_R[subj_a][subj_cell, subj_cell] = params.R_listen

                        for k in range(self.N):
                            for l in range(self.M):
                                obj_coord = np.array([k, l])
                                obj_cell = np.ravel_multi_index(obj_coord, self.grid_shape)
                                common_state = np.ravel_multi_index(
                                    (subj_cell, obj_cell), (self.num_cell, self.num_cell))
                                obj_freenbs = list()
                                for dir in self.rim_dirs:
                                    obj_nb_coord = self.apply_move(obj_coord, dir)
                                    if self.check_free(obj_nb_coord):
                                        obj_freenbs.append(np.ravel_multi_index(obj_nb_coord, self.grid_shape))
                                num_obj_freenb = len(obj_freenbs)
                                if not num_obj_freenb:
                                    obj_freenbs = [obj_cell]
                                    num_obj_freenb = 1
                                for obj_a in range(self.num_action):
                                    joint_a = np.ravel_multi_index(
                                        (subj_a, obj_a), (self.num_action, self.num_action))
                                    if obj_a == params.dooropen and obj_cell == goal:
                                        next_common_state = np.array(
                                            [np.ravel_multi_index((x, y), (self.num_cell, self.num_cell))
                                             for x in subj_freenbs for y in obj_freenbs])
                                        ma_T[joint_a][
                                            common_state, next_common_state] = 1.0 / (num_subj_freenb * num_obj_freenb)
                                        subj_ma_R[joint_a][common_state, next_common_state] = params.R_listen
                                        obj_ma_R[joint_a][common_state, next_common_state] = params.R_open_goal
                                    elif obj_a == params.staynlisten or obj_a == params.dooropen:
                                        next_common_state = common_state
                                        ma_T[joint_a][common_state, next_common_state] = 1.0
                                        subj_ma_R[joint_a][common_state, next_common_state] = params.R_listen
                                        if obj_a == params.staynlisten:
                                            obj_ma_R[joint_a][common_state, next_common_state] = params.R_listen
                                        else:
                                            obj_ma_R[joint_a][common_state, next_common_state] = params.R_open_wrong
                                    else:
                                        obj_next_coord = self.apply_move(obj_coord, np.array(self.moves[obj_a]))
                                        if not self.check_free(obj_next_coord):
                                            obj_next_coord = obj_coord.copy()
                                        obj_next_cell = np.ravel_multi_index(obj_next_coord, self.grid_shape)
                                        subj_next_cell = subj_cell
                                        next_common_state = np.ravel_multi_index(
                                            (subj_next_cell, obj_next_cell), (self.num_cell, self.num_cell))
                                            
                                        if obj_next_cell == obj_cell:
                                            ma_T[joint_a][common_state, next_common_state] = 1.0
                                            subj_ma_R[joint_a][common_state, next_common_state] = params.R_listen
                                            obj_ma_R[joint_a][common_state, next_common_state] = params.R_obst
                                        else:
                                            ma_T[joint_a][common_state, next_common_state] = pr_move_succ
                                            ma_T[joint_a][common_state, common_state] = 1.0 - pr_move_succ
                                            subj_ma_R[joint_a][common_state, [common_state, next_common_state]] = params.R_listen
                                            obj_ma_R[joint_a][common_state, [common_state, next_common_state]] = params.R_move[obj_a]
                    else:
                        subj_next_coord = self.apply_move(subj_coord, np.array(self.moves[subj_a]))
                        if not self.check_free(subj_next_coord):
                            subj_next_coord = subj_coord.copy()
                        subj_next_cell = np.ravel_multi_index(subj_next_coord, self.grid_shape)
                        if subj_next_cell == subj_cell:
                            subj_sa_T[subj_a][subj_cell, subj_next_cell] = 1.0
                            subj_sa_R[subj_a][subj_cell, subj_next_cell] = params.R_obst
                        else:
                            subj_sa_T[subj_a][subj_cell, subj_next_cell] = pr_move_succ
                            subj_sa_T[subj_a][subj_cell, subj_cell] = 1.0 - pr_move_succ
                            subj_sa_R[subj_a][subj_cell, [subj_cell, subj_next_cell]] = params.R_move[subj_a]

                        for k in range(self.N):
                            for l in range(self.M):
                                obj_coord = np.array([k, l])
                                obj_cell = np.ravel_multi_index(obj_coord, self.grid_shape)
                                common_state = np.ravel_multi_index(
                                    (subj_cell, obj_cell), (self.num_cell, self.num_cell))
                                obj_freenbs = list()
                                for dir in self.rim_dirs:
                                    obj_nb_coord = self.apply_move(obj_coord, dir)
                                    if self.check_free(obj_nb_coord):
                                        obj_freenbs.append(np.ravel_multi_index(obj_nb_coord, self.grid_shape))
                                num_obj_freenb = len(obj_freenbs)
                                if not num_obj_freenb:
                                    obj_freenbs = [obj_cell]
                                    num_obj_freenb = 1
                                for obj_a in range(self.num_action):
                                    joint_a = np.ravel_multi_index(
                                        (subj_a, obj_a), (self.num_action, self.num_action))
                                    if obj_a == params.dooropen and obj_cell == goal:
                                        next_common_state = np.array(
                                            [np.ravel_multi_index((x, y), (self.num_cell, self.num_cell))
                                             for x in subj_freenbs for y in obj_freenbs])
                                        ma_T[joint_a][
                                            common_state, next_common_state] = 1.0 / (num_subj_freenb * num_obj_freenb)
                                        subj_ma_R[joint_a][common_state, next_common_state] = params.R_move[subj_a]
                                        obj_ma_R[joint_a][common_state, next_common_state] = params.R_open_goal
                                    elif obj_a == params.staynlisten or obj_a == params.dooropen:
                                        if subj_next_cell == subj_cell:
                                            next_common_state = common_state
                                            ma_T[joint_a][common_state, next_common_state] = 1.0
                                            subj_ma_R[joint_a][common_state, next_common_state] = params.R_obst
                                            if obj_a == params.staynlisten:
                                                obj_ma_R[joint_a][common_state, next_common_state] = params.R_listen
                                            else:
                                                obj_ma_R[joint_a][common_state, next_common_state] = params.R_open_wrong
                                        else:
                                            next_common_state = np.ravel_multi_index(
                                                (subj_next_cell, obj_cell), (self.num_cell, self.num_cell))
                                            ma_T[joint_a][common_state, next_common_state] = pr_move_succ
                                            ma_T[joint_a][common_state, common_state] = 1.0 - pr_move_succ
                                            subj_ma_R[joint_a][common_state, [common_state, next_common_state]] = params.R_move[subj_a]
                                            if obj_a == params.staynlisten:
                                                obj_ma_R[joint_a][common_state, [common_state, next_common_state]] = params.R_listen
                                            else:
                                                obj_ma_R[joint_a][common_state, [common_state, next_common_state]] = params.R_open_wrong
                                    else:
                                        obj_next_coord = self.apply_move(obj_coord, np.array(self.moves[obj_a]))
                                        if not self.check_free(obj_next_coord):
                                            obj_next_coord = obj_coord.copy()
                                        obj_next_cell = np.ravel_multi_index(obj_next_coord, self.grid_shape)

                                        if subj_next_cell == subj_cell:
                                            if obj_next_cell == obj_cell:
                                                next_common_state = common_state
                                                ma_T[joint_a][common_state, next_common_state] = 1.0
                                                subj_ma_R[joint_a][common_state, next_common_state] = params.R_obst
                                                obj_ma_R[joint_a][common_state, next_common_state] = params.R_obst
                                            else:
                                                next_common_state = np.ravel_multi_index(
                                                    (subj_next_cell, obj_next_cell), (self.num_cell, self.num_cell))
                                                ma_T[joint_a][common_state, next_common_state] = pr_move_succ
                                                ma_T[joint_a][common_state, common_state] = 1.0 - pr_move_succ
                                                subj_ma_R[joint_a][common_state, [common_state, next_common_state]] = params.R_obst
                                                obj_ma_R[joint_a][common_state, [common_state, next_common_state]] = params.R_move[obj_a]
                                        else:
                                            if obj_next_cell == obj_cell:
                                                next_common_state = np.ravel_multi_index(
                                                    (subj_next_cell, obj_next_cell), (self.num_cell, self.num_cell))
                                                ma_T[joint_a][common_state, next_common_state] = pr_move_succ
                                                ma_T[joint_a][common_state, common_state] = 1.0 - pr_move_succ
                                                subj_ma_R[joint_a][common_state, [common_state, next_common_state]] = params.R_move[subj_a]
                                                obj_ma_R[joint_a][common_state, [common_state, next_common_state]] = params.R_obst
                                            else:
                                                next_common_state = np.ravel_multi_index(
                                                    (subj_next_cell, obj_next_cell), (self.num_cell, self.num_cell))
                                                unexp_res_1 = np.ravel_multi_index(
                                                    (subj_cell, obj_next_cell), (self.num_cell, self.num_cell))
                                                unexp_res_2 = np.ravel_multi_index(
                                                    (subj_next_cell, obj_cell), (self.num_cell, self.num_cell))
                                                ma_T[joint_a][common_state, next_common_state] = pr_move_succ ** 2
                                                ma_T[joint_a][common_state, unexp_res_1] = pr_move_succ * (1.0 - pr_move_succ)
                                                ma_T[joint_a][common_state, unexp_res_2] = (1.0 - pr_move_succ) * pr_move_succ
                                                ma_T[joint_a][common_state, common_state] = (1.0 - pr_move_succ) ** 2
                                                subj_ma_R[joint_a][
                                                    common_state, 
                                                    [common_state, unexp_res_1, unexp_res_2, next_common_state]] = params.R_move[subj_a]
                                                obj_ma_R[joint_a][
                                                    common_state, 
                                                    [common_state, unexp_res_1, unexp_res_2, next_common_state]] = params.R_move[obj_a]

        return subj_sa_T, ma_T, subj_sa_R, subj_ma_R, obj_ma_R

    def build_deterministic_trans(self):
        # maximum likely versions
        # Tml[s, a] --> next state (except for the open action)
        trans_tab = np.zeros([self.num_cell, self.num_action - 1], 'i')

        for i in range(self.N):
            for j in range(self.M):
                cell_coord = np.array([i, j])
                cell = np.ravel_multi_index(cell_coord, (self.N, self.M))

                # build T w.r.t. cells
                for act in range(self.num_action - 1):
                    neighbor_coord = self.apply_move(cell_coord, np.array(self.moves[act]))
                    if not self.check_free(neighbor_coord):
                        # cannot move if obstacle or edge of world
                        neighbor_coord = [i, j]

                    neighbor = np.ravel_multi_index(neighbor_coord, (self.N, self.M))
                    trans_tab[cell, act] = neighbor

        return trans_tab

    def getX_TR(
            self,
            goal):
        X_T = np.zeros(
            self.num_state, dtype=np.float32)
        X_R = np.zeros(
            self.num_state, dtype=np.float32)
        interaction_states = [np.ravel_multi_index((goal, s_j), (self.num_cell, self.num_cell))
                              for s_j in range(self.num_cell)]
        interaction_states += [np.ravel_multi_index((s_i, goal), (self.num_cell, self.num_cell))
                               for s_i in range(self.num_cell) if s_i != goal]
        X_T[interaction_states] = 1.0
        X_R[interaction_states] = 1.0

        return X_T, X_R

    def get_maqmdp(self):
        model = MAQMDP(self.params, self.grid)

        model.processT(
            subj_sa_T=self.subj_sa_T,
            obj_sa_T=self.obj_sa_T,
            ma_T=self.ma_T)
        model.processR(
            subj_sa_R=self.subj_sa_R,
            subj_ma_R=self.subj_ma_R,
            obj_sa_R=self.obj_sa_R,
            obj_ma_R=self.obj_ma_R,
            subj_sa_T=self.subj_sa_T,
            obj_sa_T=self.obj_sa_T)
        model.processO(
            sa_O=self.sa_O,
            expanded_subj_sa_O=self.expanded_subj_sa_O,
            expanded_obj_sa_O=self.expanded_obj_sa_O,
            subj_ma_O=self.subj_ma_O,
            obj_ma_O=self.obj_ma_O)
        model.getX_TR(
            X_T=self.X_T,
            X_R=self.X_R)

        model.transfer_all_sparse()
        return model

    @staticmethod
    def outofbounds(
            grid,
            coord):
        return (coord[0] < 0 or coord[0] >= grid.shape[0] or coord[1] < 0 or
                coord[1] >= grid.shape[1])

    @staticmethod
    def apply_move(
            coord_in,
            move):
        coord = coord_in.copy()
        coord[:2] += move[:2]
        return coord

    def check_free(
            self,
            coord):
        return (not self.outofbounds(
            self.grid, coord) and self.grid[coord[0], coord[1]] != OBSTACLE)

    def random_grid(
            self,
            pr_obst_high,
            pr_obst_low=0.0,
            random_pr_obst=False):
        rand_field = np.random.rand(self.N, self.M)
        if random_pr_obst:
            pr_obst = np.random.uniform(pr_obst_low, pr_obst_high)
        else:
            pr_obst = pr_obst_high
        grid = (rand_field > pr_obst).astype('i')
        return grid
