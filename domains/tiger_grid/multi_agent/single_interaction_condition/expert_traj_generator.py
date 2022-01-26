"""
The expert demonstration trajectory generator 
under the imitation learning setting.
"""

import multiprocessing
import _pickle as pickle
import shutil
import tables
import sys
import os

import numpy as np

sys.path.append('/home/gz67063/projects/sipomdplite-net/')
from domains.tiger_grid.multi_agent.single_interaction_condition.env_simulator import MATigerGridEnv
from configs.envs import tiger_grid
from utils.envs.dotdict import dotdict

try:
    import ipdb as pdb
except ImportError:
    import pdb

# Print setting
np.set_printoptions(threshold=3000)


class ILExpertDemonstrationGenerator(object):
    def __init__(self, domain, params):
        self.domain = domain
        self.params = params
        self.num_collision_trajs = 0
        self.num_wrongopen_trajs = 0

    def generate_trajectories(
            self,
            db,
            trajs_per_env):
        domain = self.domain
        params = self.params

        for traj_i in range(trajs_per_env):
            # Generate a QMDP object, initial belief, initial state and goal
            # state also generates a random grid for the first iteration
            maqmdp, b0_comm_s, b0_priv_s, init_s, goal = domain.random_instance((traj_i == 0))
            # print("\ngrid map:")
            # print(self.grid)
            # print("initial state of both agents:", init_s)
            # print("goal cell:", goal)
            # print("initial belief of agent i:")
            # print(b0_bin.reshape((2, self.N, self.M)))
            maqmdp.solve()
            nested_policy = np.transpose(
                maqmdp.opponent_policy,
                axes=(1, 0))

            comm_s = init_s.copy()
            lin_comm_s = np.ravel_multi_index(
                comm_s, (domain.num_cell, domain.num_cell))

            b_i, b_j = b0_comm_s.copy(), b0_comm_s.copy()  # binary belief
            b_i_priv_s = b0_priv_s.copy()
            # print("i's initial belief over i:",
            #       b_i.reshape((domain.num_cell, domain.num_cell)).sum(1).
            #       reshape((domain.N, domain.M)))

            reward_sum_i = 0.0
            gamma_acc = 1.0

            beliefs_i = [b_i_priv_s]
            common_states = list()
            actions_i = list()
            actions_j = list()
            observs_i = list()
            # observs_j = list()

            collisions = 0
            wrong_opens_i = 0
            wrong_opens_j = 0
            succ_opens_i = 0
            succ_opens_j = 0
            failed = False
            step_i = 0

            while True:
                # print("\nSTEP", step_i)
                common_states.append(lin_comm_s)

                # stop if trajectory limit reached
                if step_i >= params.max_traj_len \
                        or succ_opens_i >= 5 \
                        or succ_opens_j >= 5:
                    if succ_opens_i < 1:
                        failed = True
                    if collisions > 1:
                        failed = True
                    print("COLLISION TIMES:", collisions)
                    print("SUCCESSFUL OPEN TIMES:", succ_opens_i)
                    print("WRONG OPEN TIMES:", wrong_opens_i)
                    print("ACCUMULATED REWARD:", reward_sum_i)
                    print("failed:", failed)
                    print("traj_len:", step_i)
                    break

                # choose action
                if step_i == 0:
                    # dummy first action
                    act_i, act_j = params.staynlisten, params.staynlisten
                else:
                    act_i = maqmdp.maqmdp_action(b_i, identity='subjective')
                    act_j = maqmdp.maqmdp_action(b_j, identity='objective')

                # print("AGENT i CHOOSES ACTION [%d]" % act_i)
                # print("AGENT j CHOOSES ACTION [%d]" % act_j)

                joint_a = [act_i, act_j]
                lin_joint_a = np.ravel_multi_index(
                    joint_a, (params.num_action, params.num_action))

                if act_i == params.dooropen and comm_s[0] != goal:
                    wrong_opens_i += 1

                if act_i == params.dooropen and comm_s[0] == goal:
                    succ_opens_i += 1

                if act_j == params.dooropen and comm_s[1] != goal:
                    wrong_opens_j += 1

                if act_j == params.dooropen and comm_s[1] == goal:
                    succ_opens_j += 1

                # simulate action
                comm_s, lin_comm_s, reward_i = maqmdp.transition(
                    comm_s, lin_comm_s, joint_a, lin_joint_a)
                # subj_s, obj_s = comm_s
                # print("AGENT i IS AT [%d], AGENT j IS AT [%d]" % (subj_s, obj_s))

                obs_i = maqmdp.random_obs(
                    lin_common_s=lin_comm_s,
                    lin_joint_act=lin_joint_a,
                    identity='subjective')
                obs_j = maqmdp.random_obs(
                    lin_common_s=lin_comm_s,
                    lin_joint_act=lin_joint_a,
                    identity='objective')

                _, b_i = maqmdp.belief_update_with_obs(
                    belief=b_i,
                    self_priv_a=act_i,
                    obs=obs_i)
                _, b_j = maqmdp.belief_update_with_obs(
                    belief=b_j,
                    self_priv_a=act_j,
                    obs=obs_j)

                # print("AGENT i OBSERVES", np.unravel_index(obs_i, (2, 2, 2, 2)))
                # print("AGENT j OBSERVES", np.unravel_index(obs_j, (2, 2, 2, 2)))
                # print("b_i over i's locations:")
                # print(b_i.toarray().squeeze().reshape(
                #     (domain.num_cell, domain.num_cell)).sum(1).reshape((domain.N, domain.M)))
                # print("b_i over j's locations:")
                # print(b_i.toarray().squeeze().reshape(
                #     (domain.num_cell, domain.num_cell)).sum(0).reshape((domain.N, domain.M)))

                actions_i.append(act_i)
                actions_j.append(act_j)
                observs_i.append(obs_i)

                if np.array_equal(actions_i[-2:], [5, 5]) or \
                        np.array_equal(actions_i[-2:], [1, 3]) or \
                        np.array_equal(actions_i[-2:], [3, 1]) or \
                        np.array_equal(actions_i[-2:], [4, 2]) or \
                        np.array_equal(actions_i[-2:], [2, 4]) or \
                        np.array_equal(actions_i[-3:], [0, 0]):
                    failed = True

                reward_sum_i += reward_i

                # count collisions
                if np.isclose(reward_i, params.R_obst):
                    collisions += 1

                step_i += 1

            # print("ACTION TRAJECTORY:", actions_i)

            # Fill into the database.
            if not failed:
                db.root.valid_trajs.append([len(db.root.samples)])
            if wrong_opens_i:
                self.num_wrongopen_trajs += 1
            if collisions:
                self.num_collision_trajs += 1

            traj_len = step_i

            step = np.stack(
                [common_states[:traj_len],
                 actions_i[:traj_len],
                 actions_j[:traj_len],
                 observs_i[:traj_len]], axis=1)

            # The "sample" stores all the indices pointing at envs and steps, along with
            # instance/task-corresponding (goal, trajectory length, collision times, and
            # flag of success or failure.)
            sample = np.array(
                [len(db.root.envs), goal, len(db.root.steps), traj_len,
                 collisions, wrong_opens_i, failed], dtype='i')
            db.root.samples.append(sample[None])
            db.root.beliefs.append(np.array(beliefs_i[:1]))
            db.root.nested_policies.append(nested_policy[None])
            db.root.expRs.append([reward_sum_i])
            db.root.steps.append(step)
        db.root.envs.append(domain.grid[None])

    def generate_trajectories_parallel(
            self,
            db_path,
            db_i,
            envs,
            trajs_per_env,
            write_mode):
        domain = self.domain
        params = self.params

        for env_i in range(int(envs)):
            if write_mode == 'overwrite' and env_i == 0:
                db = self.create_db(
                    db_path=db_path + "data_" + str(db_i) + ".hdf5",
                    total_env_count=envs,
                    trajs_per_env=trajs_per_env)
            else:
                db = tables.open_file(db_path + "data_" + str(db_i) + ".hdf5", mode='a')

            for traj_i in range(trajs_per_env):
                # Generate a QMDP object, initial belief, initial state and goal
                # state also generates a random grid for the first iteration
                maqmdp, b0_comm_s, b0_priv_s, init_s, goal = domain.random_instance((traj_i == 0))
                # print("\ngrid map:")
                # print(self.domain.grid)
                # print("initial state of both agents:", init_s)
                # print("goal cell:", goal)
                # print("initial belief of agent i:")
                # print(b0_bin.reshape((2, self.domain.N, self.domain.M)))
                maqmdp.solve()
                nested_policy = np.transpose(
                    maqmdp.opponent_policy,
                    axes=(1, 0))

                comm_s = init_s.copy()
                lin_comm_s = np.ravel_multi_index(
                    comm_s, (domain.num_cell, domain.num_cell))

                b_i, b_j = b0_comm_s.copy(), b0_comm_s.copy()  # binary belief
                b_i_priv_s = b0_priv_s.copy()
                # print("i's initial belief over i:",
                #       b_i.reshape((domain.num_cell, domain.num_cell)).sum(1).
                #       reshape((domain.N, domain.M)))

                reward_sum_i = 0.0

                beliefs_i = [b_i_priv_s]
                common_states = list()
                actions_i = list()
                actions_j = list()
                observs_i = list()

                collisions = 0
                wrong_opens_i = 0
                wrong_opens_j = 0
                succ_opens_i = 0
                succ_opens_j = 0
                failed = False
                step_i = 0

                while True:
                    # print("\nSTEP", step_i)
                    common_states.append(lin_comm_s)

                    # stop if trajectory limit reached
                    if step_i >= params.max_traj_len \
                            or succ_opens_i >= 5 \
                            or succ_opens_j >= 5:
                        if succ_opens_i < 1:
                            failed = True
                        if collisions > 1:
                            failed = True
                        print("COLLISION TIMES:", collisions)
                        print("SUCCESSFUL OPEN TIMES:", succ_opens_i)
                        print("WRONG OPEN TIMES:", wrong_opens_i)
                        print("ACCUMULATED REWARD:", reward_sum_i)
                        print("failed:", failed)
                        print("traj_len:", step_i)
                        break

                    # choose action
                    if step_i == 0:
                        # dummy first action
                        act_i, act_j = params.staynlisten, params.staynlisten
                    else:
                        act_i = maqmdp.maqmdp_action(b_i, identity='subjective')
                        act_j = maqmdp.maqmdp_action(b_j, identity='objective')

                    # print("AGENT i CHOOSES ACTION [%d]" % act_i)
                    # print("AGENT j CHOOSES ACTION [%d]" % act_j)

                    joint_a = [act_i, act_j]
                    lin_joint_a = np.ravel_multi_index(
                        joint_a, (params.num_action, params.num_action))

                    if act_i == params.dooropen and comm_s[0] != goal:
                        wrong_opens_i += 1

                    if act_i == params.dooropen and comm_s[0] == goal:
                        succ_opens_i += 1

                    if act_j == params.dooropen and comm_s[1] != goal:
                        wrong_opens_j += 1

                    if act_j == params.dooropen and comm_s[1] == goal:
                        succ_opens_j += 1

                    # simulate action
                    comm_s, lin_comm_s, reward_i = maqmdp.transition(
                        comm_s, lin_comm_s, joint_a, lin_joint_a)
                    # subj_s, obj_s = comm_s
                    # print("AGENT i IS AT [%d], AGENT j IS AT [%d]" % (subj_s, obj_s))

                    obs_i = maqmdp.random_obs(
                        lin_common_s=lin_comm_s,
                        lin_joint_act=lin_joint_a,
                        identity='subjective')
                    obs_j = maqmdp.random_obs(
                        lin_common_s=lin_comm_s,
                        lin_joint_act=lin_joint_a,
                        identity='objective')

                    _, b_i = maqmdp.belief_update_with_obs(
                        belief=b_i,
                        self_priv_a=act_i,
                        obs=obs_i)
                    _, b_j = maqmdp.belief_update_with_obs(
                        belief=b_j,
                        self_priv_a=act_j,
                        obs=obs_j)

                    # print("AGENT i OBSERVES", np.unravel_index(obs_i, (2, 2, 2, 2)))
                    # print("AGENT j OBSERVES", np.unravel_index(obs_j, (2, 2, 2, 2)))
                    # print("b_i over i's locations:")
                    # print(b_i.toarray().squeeze().reshape(
                    #     (domain.num_cell, domain.num_cell)).sum(1).reshape((domain.N, domain.M)))
                    # print("b_i over j's locations:")
                    # print(b_i.toarray().squeeze().reshape(
                    #     (domain.num_cell, domain.num_cell)).sum(0).reshape((domain.N, domain.M)))

                    actions_i.append(act_i)
                    actions_j.append(act_j)
                    observs_i.append(obs_i)

                    if np.array_equal(actions_i[-2:], [5, 5]) or \
                            np.array_equal(actions_i[-2:], [1, 3]) or \
                            np.array_equal(actions_i[-2:], [3, 1]) or \
                            np.array_equal(actions_i[-2:], [4, 2]) or \
                            np.array_equal(actions_i[-2:], [2, 4]) or \
                            np.array_equal(actions_i[-3:], [0, 0]):
                        failed = True

                    reward_sum_i += reward_i

                    # count collisions
                    if np.isclose(reward_i, params.R_obst):
                        collisions += 1

                    step_i += 1

                # print("ACTION TRAJECTORY:", actions_i)

                # Fill into the database.
                if not failed:
                    db.root.valid_trajs.append([len(db.root.samples)])
                if wrong_opens_i:
                    self.num_wrongopen_trajs += 1
                if collisions:
                    self.num_collision_trajs += 1

                traj_len = step_i

                step = np.stack(
                    [common_states[:traj_len],
                     actions_i[:traj_len],
                     actions_j[:traj_len],
                     observs_i[:traj_len]], axis=1)

                # The "sample" stores all the indices pointing at envs and steps, along with
                # instance/task-corresponding (goal, trajectory length, collision times, and
                # flag of success or failure.)
                sample = np.array(
                    [len(db.root.envs), goal, len(db.root.steps), traj_len,
                     collisions, wrong_opens_i, failed], dtype='i')
                db.root.samples.append(sample[None])
                db.root.beliefs.append(np.array(beliefs_i[:1]))
                db.root.nested_policies.append(nested_policy[None])
                db.root.expRs.append([reward_sum_i])
                db.root.steps.append(step)
            db.root.envs.append(domain.grid[None])
            db.close()

    def create_db(self, db_path, total_env_count=None, trajs_per_env=None):
        """
        :param db_path: file name for database
        :param total_env_count: total number of environments in the dataset
        (helps to preallocate space)
        :param trajs_per_env: number of trajectories per environment
        """
        if total_env_count is not None and trajs_per_env is not None:
            total_traj_count = total_env_count * trajs_per_env * 10
        else:
            total_traj_count = 0

        db = tables.open_file(db_path, mode='w')

        db.create_earray(
            db.root,
            name='envs',
            atom=tables.Int32Atom(),
            shape=(0, self.domain.N, self.domain.M),
            expectedrows=total_env_count)

        db.create_earray(
            db.root,
            name='valid_trajs',
            atom=tables.Int32Atom(),
            shape=(0,),
            expectedrows=total_traj_count)

        db.create_earray(
            db.root,
            name='expRs',
            atom=tables.Float32Atom(),
            shape=(0,),
            expectedrows=total_traj_count)

        db.create_earray(
            db.root,
            name='beliefs',
            atom=tables.Float32Atom(),
            shape=(0, 2, self.domain.num_cell),
            expectedrows=total_traj_count)

        db.create_earray(
            db.root,
            name='nested_policies',
            atom=tables.Float32Atom(),
            shape=(0, self.domain.num_state, self.domain.num_action),
            expectedrows=total_traj_count)

        db.create_earray(
            db.root,
            name='steps',
            atom=tables.Int32Atom(),
            shape=(0, 4),
            expectedrows=total_traj_count * 10)  # rough estimate

        # env_id, goal_state, step_id, traj_length, collisions, failed
        db.create_earray(
            db.root,
            name='samples',
            atom=tables.Int32Atom(),
            shape=(0, 7),
            expectedrows=total_traj_count)
        return db


def generate_grid_data(db_path, grid_n, grid_m, envs, trajs_per_env,
                       pr_obst_high, pr_obst_low, random_pr_obst,
                       R_obst, R_move, R_listen, R_open_goal, R_open_wrong,
                       pr_move_succ, pr_obs_succ, discount, reason_level,
                       parallel, write_mode='append'):
    """
    :param db_path: path for data file. use separate folders for training and test data
    :param grid_n: grid rows
    :param grid_m: grid columns
    :param envs: number of environments in the dataset (grids)
    :param trajs_per_env: number of trajectories per environment (different initial state, goal, initial belief)
    :param pr_obst_high:
    :param pr_obst_low:
    :param random_pr_obst:
    :param R_obst:
    :param R_move:
    :param R_listen:
    :param R_open_goal:
    :param R_open_wrong:
    :param pr_move_succ: probability of transition succeeding, otherwise stays in place
    :param pr_obs_succ: probability of correct observation, independent in each direction
    :param discount:
    :param reason_level:
    """
    params = dotdict({
        'grid_n': grid_n,
        'grid_m': grid_m,
        'pr_obst_high': pr_obst_high,  # probability of obstacles in random grid
        'pr_obst_low': pr_obst_low,
        'random_Pobst': random_pr_obst,

        'R_obst': R_obst,
        'R_open_goal': R_open_goal,
        'R_open_wrong': R_open_wrong,
        'R_move': R_move,
        'R_listen': R_listen,
        'discount': discount,
        'pr_move_succ': pr_move_succ,
        'pr_obs_succ': pr_obs_succ,

        'num_action': 6,
        'moves': [[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]],
        'staynlisten': 0,
        'dooropen': 5,

        'cardinal_dirs': [[0, 1], [1, 0], [0, -1], [-1, 0]],
        'rim_dirs': [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]],

        'reason_level': reason_level,

        'parallel': parallel,
    })

    params['num_joint_action'] = params.num_action ** 2
    params['obs_dim'] = len(params.cardinal_dirs)
    params['num_obs'] = 2 ** params.obs_dim
    params['num_cell'] = params.grid_n * params.grid_m
    params['num_state'] = params.num_cell ** 2
    params['max_traj_len'] = 2 * (params.grid_n + params.grid_m)
    params['R_move'] = [params.R_move] * params.num_action

    # save params
    if not os.path.isdir(db_path):
        os.mkdir(db_path)
    pickle.dump(dict(params), open(db_path + "params.pickle", 'wb'), -1)

    # make database file
    if parallel:
        num_proc = multiprocessing.cpu_count()
        # db_list = list()
        # if write_mode == 'overwrite':
        #     for i in range(num_db):
        #         db_list.append(
        #             generator.create_db(
        #                 db_path=db_path + "data_" + str(i) + ".hdf5",
        #                 total_env_count=envs//num_db,
        #                 trajs_per_env=trajs_per_env))
        # else:
        #     for i in range(num_db):
        #         if os.path.isfile(db_path + "data_" + str(i) + ".hdf5"):
        #             db_list.append(tables.open_file(db_path + "data_" + str(i) + ".hdf5", mode='a'))
        #         else:
        #             db_list.append(
        #                 generator.create_db(
        #                     db_path=db_path + "data_" + str(i) + ".hdf5",
        #                     total_env_count=envs//num_db,
        #                     trajs_per_env=trajs_per_env))

        # iter_db_list = list()
        # for i in range(envs):
        #     idx = np.arange(num_db)[i % num_db]
        #     iter_db_list.append(db_list[idx])
        # proc_pool = multiprocessing.Pool()
        # proc_pool.map(
        #     func=generator.generate_trajectories,
        #     iterable=list(zip(iter_db_list, [trajs_per_env for _ in range(envs)])))
        # proc_pool.close()
        # proc_pool.join()

        ps = list()
        rand_seed = np.random.randint(1000)
        for i in range(num_proc):
            np.random.seed(rand_seed + i)
            domain = MATigerGridEnv(params)
            generator = ILExpertDemonstrationGenerator(domain=domain, params=params)

            if not envs % num_proc:
                envs_per_proc = envs / num_proc
            else:
                envs_per_proc = envs // num_proc + 1
            p = multiprocessing.Process(
                target=generator.generate_trajectories_parallel,
                args=(db_path, i, envs_per_proc, trajs_per_env, write_mode))
            ps.append(p)
            p.start()
        for p in ps:
            p.join()

    else:
        # randomize seeds, set to previous value to determine random numbers
        np.random.seed(73)

        # grid domain object
        domain = MATigerGridEnv(params)
        generator = ILExpertDemonstrationGenerator(domain=domain, params=params)

        if write_mode == 'overwrite':
            db = generator.create_db(
                db_path=db_path + "data.hdf5",
                total_env_count=envs,
                trajs_per_env=trajs_per_env)
        else:
            db = tables.open_file(db_path + "data.hdf5", mode='a')

        for env_i in range(envs):
            print("\nGenerating env %d with %d trajectories " % (env_i, trajs_per_env))
            # arg = list(zip(db, trajs_per_env))
            generator.generate_trajectories(db, trajs_per_env)
            print("Current succ rate:", len(db.root.valid_trajs) / len(db.root.samples))
            print("Current collision rate:", generator.num_collision_trajs / len(db.root.samples))
            print("Current wrong open rate:", generator.num_wrongopen_trajs / len(db.root.samples))

    # start_time_main = time.time()
    # end_time_main = time.time()
    # print("Total time consumed:", end_time_main - start_time_main)

    print("Done.")


def main():
    args = tiger_grid.get_args()
    write_mode = args.write_mode
    db_path = args.db_path
    if db_path[-1] != '/':
        db_path += '/'

    if os.path.isdir(db_path):
        if write_mode == 'overwrite':
            answer = str()
            while answer.lower() not in ['y', 'yes', 'n', 'no']:
                answer = input("You are going to overwrite the existing database at %s. Are you sure(y/n)?"
                               % db_path)
            if answer.lower() in ['y', 'yes']:
                shutil.rmtree(db_path)
                os.mkdir(db_path)
            else:
                ans = str()
                while ans.lower() not in ['y', 'yes', 'n', 'no']:
                    ans = input("Would you append new trajectories to the existing database(y/n)?")
                if ans.lower() in ['y', 'yes']:
                    write_mode = "append"
                else:
                    return
        elif write_mode == 'append':
            pass
        else:
            raise ValueError("The input argument for write mode is invalid! Only accept {'append', 'overwrite'}.")
    else:
        if write_mode == 'append':
            answer = str()
            while answer.lower() not in ['y', 'yes', 'n', 'no']:
                answer = input("%s does not exist. Would you like to create a new database(y/n)?" % db_path)
            if answer.lower() in ['y', 'yes']:
                write_mode = "overwrite"
                os.mkdir(db_path)
            else:
                return
        elif write_mode == 'overwrite':
            os.mkdir(db_path)
        else:
            raise ValueError("The input argument for write mode is invalid! Only accept {'append', 'overwrite'}.")

    # training data
    # start_time = time.time()
    generate_grid_data(
        db_path=db_path + 'train/',
        grid_n=args.grid_n,
        grid_m=args.grid_m,
        envs=args.train_envs,
        trajs_per_env=args.train_trajs_per_env,
        pr_obst_high=args.pr_obst_high,
        pr_obst_low=args.pr_obst_low,
        random_pr_obst=args.random_pr_obst,
        R_listen=args.R_listen,
        R_move=args.R_move,
        R_obst=args.R_obst,
        R_open_goal=args.R_open_goal,
        R_open_wrong=args.R_open_wrong,
        pr_move_succ=args.pr_move_succ,
        pr_obs_succ=args.pr_obs_succ,
        discount=args.discount,
        reason_level=args.reason_level,
        parallel=args.parallel,
        write_mode=write_mode)
    # end_time = time.time()
    # print("time taken for the whole process:", end_time - start_time)

    # evaluation data
    generate_grid_data(
        db_path=db_path + 'eval/',
        grid_n=args.grid_n,
        grid_m=args.grid_m,
        envs=args.eval_envs,
        trajs_per_env=args.eval_trajs_per_env,
        pr_obst_high=args.pr_obst_high,
        pr_obst_low=args.pr_obst_low,
        random_pr_obst=args.random_pr_obst,
        R_listen=args.R_listen,
        R_move=args.R_move,
        R_obst=args.R_obst,
        R_open_goal=args.R_open_goal,
        R_open_wrong=args.R_open_wrong,
        pr_move_succ=args.pr_move_succ,
        pr_obs_succ=args.pr_obs_succ,
        discount=args.discount,
        reason_level=args.reason_level,
        parallel=args.parallel,
        write_mode=write_mode)


# default
if __name__ == "__main__":
    main()
