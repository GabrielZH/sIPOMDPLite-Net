import sys

import numpy as np
from tensorpack import dataflow

sys.path.append('/home/gz67063/projects/sipomdplite-net')

from data_processing.database import Database
from domains.tiger_grid.single_agent.env_simulator import SATigerGridEnv
from domains.tiger_grid.single_agent.learner_traj_evaluation import simulate_policy


class Datafeed(object):
    def __init__(
            self,
            params,
            filename,
            mode="train",
            min_env=0,
            max_env=0):
        """
        Datafeed from filtered samples
        :param params: dotdict including both domain parameters and training parameters
        :param filename: database file path
        :param mode: "train" or "valid" or "eval"
        :param min_env: only include environments with id larger than this
        :param max_env: only include environments with id smaller than this. No limit if set to zero.
        """
        self.params = params
        self.mode = mode
        self.steps_in_epoch = None
        self.filename = filename
        self.min_env = min_env
        self.max_env = max_env

        # wrapper that returns a database object
        self.get_db = (lambda: Database(filename))
        self.domain = SATigerGridEnv(params)

        # preload filtered samples before forked
        only_valid = (mode != "eval" and not self.params.include_failed_trajs)
        self.filtered_samples = self.filter_samples(
            params=params,
            db=self.get_db(),
            only_valid=only_valid,
            min_env=min_env,
            max_env=max_env)

    @staticmethod
    def filter_samples(
            params,
            db,
            only_valid=True,
            min_env=0,
            max_env=0):
        """Preloads samples and produces filtered_samples filtered according to training parameters
        Sample filter format: (sample_id, env_id, goal_state, step_id, effective_traj_len)
        """
        db.open()

        # preload all samples, because random access is slow
        samples = db.samples[:]  # env_id, goal_state, step_id, traj_length, collisions, failed

        # filter valids
        if only_valid:
            sample_indices = db.valid_trajs[:]
        else:
            sample_indices = np.arange(len(db.samples))

        # filter env range
        if max_env > 0 or min_env > 0:
            env_indices = samples[sample_indices, 0]
            # transform percentage to index
            if not max_env:
                max_env_i = 9999999
            elif max_env <= 1.0:
                max_env_i = int(len(db.envs) * max_env)
            else:
                max_env_i = int(max_env)

            if min_env <= 1.0:
                min_env_i = int(len(db.envs) * min_env)
            else:
                min_env_i = int(min_env)

            sample_indices = sample_indices[
                np.nonzero(
                    np.logical_and(
                        env_indices >= min_env_i, env_indices < max_env_i))[0]]

            print("Envs limited to the range %d-%d from %d" %
                  (min_env_i, (max_env_i if max_env else 0), len(db.envs)))

        samples = samples[sample_indices]
        db.close()

        # effective traj lens
        effective_traj_lens = samples[:, 3] - 1  # exclude last step in the trajectory

        # limit effective traj lens to lim_traj_len
        if params.lim_traj_len > 0:
            print("Limiting trajectory length to %d" % params.lim_traj_len)
            effective_traj_lens = np.clip(
                effective_traj_lens,
                0,
                params.lim_traj_len)
        elif params.lim_traj_len < 0 and \
                -params.lim_traj_len <= np.min(effective_traj_lens):
            print("Limiting trajectory length reversely to %d" % -params.lim_traj_len)
            effective_traj_lens = np.clip(
                effective_traj_lens,
                np.min(effective_traj_lens) + params.lim_traj_len,
                np.min(effective_traj_lens))
        else:
            pass

        # sanity check
        assert np.all(effective_traj_lens >= 0)

        filtered_samples = np.stack(
            [sample_indices,  # original index
             samples[:, 0],  # env_i
             samples[:, 1],  # goal
             samples[:, 2],  # step_i
             effective_traj_lens,  # effective_traj_len
            ], axis=1)

        return filtered_samples

    def build_dataflow(
            self,
            batch_size,
            step_size,
            restart_limit=None,
            cache=None):
        """
        :param batch_size: batch size
        :param step_size: number of steps for BPTT
        :param restart_limit: restart after limit number of batches.
        Used for validation. If 0 (but not None) or larger
         than an epoch its set to one epoch.
        :param cache: preloaded cache
        :return: dataflow with input data
        """
        # update db wrapper function with shared cache
        if cache is not None:
            self.get_db = (lambda: Database(
                filename=self.filename,
                cache=cache))

        df = dataflow.DataFromList(
            self.filtered_samples,
            shuffle=(self.mode == 'train'))

        if restart_limit is None:
            df = dataflow.RepeatedData(df, 1000000)  # reshuffles on every repeat

        df = DynamicTrajBatch(
            ds=df,
            batch_size=batch_size,
            step_size=step_size,
            traj_lens=self.filtered_samples[:, 4])

        self.steps_in_epoch = df.steps_in_epoch()
        if restart_limit is not None:
            if restart_limit == 0 or restart_limit >= self.steps_in_epoch:
                restart_limit = self.steps_in_epoch-1
            self.steps_in_epoch = restart_limit
            df = OneShotData(df, size=restart_limit)

        df = TrajDataFeed(
            ds=df,
            get_db_func=self.get_db,
            domain=self.domain,
            batch_size=batch_size,
            step_size=step_size)

        # uncomment to test dataflow speed
        # dataflow.TestDataSpeed(df, size=1000).start()

        return df

    def build_eval_dataflow(
            self,
            policy=None,
            repeats=None):
        """
        :param policy: policy to evaluate when mode == eval
        :param repeats: repeat evaluation multiple times when mode == eval
        :return: dataflow with evaluation results
        """
        df = dataflow.DataFromList(
            self.filtered_samples,
            shuffle=False)
        df = dataflow.RepeatedData(df, 1000000)
        df = EvalDataFeed(
            ds=df,
            get_db_func=self.get_db,
            domain=self.domain,
            policy=policy,
            repeats=repeats)

        return df

    def build_cache(self):
        """Preload cache of the database
        For multiprocessing call this before fork.
        Input cache to all next instances of the database.
        """
        db = self.get_db()
        cache = db.build_cache(cache_nodes=self.params.cache)
        db.close()

        return cache


class DynamicTrajBatch(dataflow.BatchDataByShape):
    def __init__(self, ds, batch_size, step_size, traj_lens):
        """
        Breaks trajectories into trainings steps and collects batches. Assumes sequential input

        Makes batches for BPTT from trajectories of different length. Batch is divided into blocks where BPTT is
        performed. Trajectories are padded to block limits. New trajectory begins from the next block, even when
        other trajectories are not finished in the batch.

        Behaviour is similar to:
        https://blog.altoros.com/the-magic-behind-google-translate-sequence-to-sequence-models-and-tensorflow.html

        :param ds: sequential input dataflow. Expects samples in the form
            (sample_id, env_id, goal_state, step_id, traj_len)
        :param batch_size: batch size
        :param step_size: step size for BPTT
        :param traj_lens: list of numpy array of trajectory lengths for ALL samples in dataset. Used to compute size()
        :return batched data, each with shape [step_size, batch_size, ...].
            Adds is_traj_head field, a binary indicator: 1 if it is the first step of the trajectory, 0 otherwise
            Output: (sample_id, env_id, goal_state, step_id, traj_len, isstart)
        """
        super(DynamicTrajBatch, self).__init__(ds, batch_size, idx=0)
        self.batch_size = batch_size
        self.step_size = step_size

        self.batch_samples = None

        self.step_field = 3
        self.traj_len_field = 4
        self.sample_fields = 6  # including is_traj_head

        blocks = ((traj_lens - 1) // self.step_size) + 1
        self._steps_in_epoch = np.sum(blocks) // self.batch_size
        self._total_epochs = (None if ds.size() is None
                              else ((ds.size() - 1) // len(traj_lens)) + 1)

    def size(self):
        return self._steps_in_epoch * self._total_epochs

    def steps_in_epoch(self):
        return self._steps_in_epoch

    def reset_state(self):
        super(DynamicTrajBatch, self).reset_state()

    def get_data(self):
        with self._guard:
            self.batch_samples = np.zeros([self.batch_size, self.sample_fields], 'i')
            generator = self.ds.get_data()
            try:
                while True:
                    # collect which samples should be replaced by a new one
                    # for the non-zero indices the sample is still valid in the batch
                    self.batch_samples[:, self.step_field] += self.step_size
                    self.batch_samples[:, self.traj_len_field] -= self.step_size
                    self.batch_samples[:, -1] = 0  # is_traj_head

                    new_indices = np.nonzero(self.batch_samples[:, self.traj_len_field] <= 0)[0]
                    self.batch_samples[new_indices, -1] = 1

                    for idx in new_indices:
                        # replace these samples in batch
                        # get new datapoint, list of fields
                        self.batch_samples[idx, :-1] = next(generator)

                    yield self.batch_samples
            except StopIteration:
                return


class EvalDataFeed(dataflow.ProxyDataFlow):
    """

    """
    def __init__(
            self,
            ds,
            get_db_func,
            domain,
            policy,
            repeats=1):
        super(EvalDataFeed, self).__init__(ds)
        self.get_db = get_db_func
        self.domain = domain
        self.policy = policy
        self.repeats = repeats

        self.db = None

        self.task_param_processor = TaskParamPreprocessor(domain)

    def __del__(self):
        self.close()

    def reset_state(self):
        super(EvalDataFeed, self).reset_state()
        if self.db is not None:
            print ("WARNING: reopening database. This is not recommended.")
            self.db.close()
        self.db = self.get_db()
        self.db.open()

    def close(self):
        if self.db is not None: self.db.close()
        self.db = None

    def get_data(self):
        for dp in self.ds.get_data():
            yield self.eval_sample(dp)

    def eval_sample(
            self,
            sample):
        """
        :param sample: sample vector in the form
        (sample_id, env_id, goal_state, step_id, traj_len)
        :return result matrix, first row for expert policy,
        consecutive rows for evaluated policy.
        fields: success rate, trajectory length, collision
        rate, accumulated reward
        """
        sample_i, env_i, goal, step_i, _ = [
            np.atleast_1d(x.squeeze())
            for x in np.split(sample, sample.shape[0], axis=0)]

        env = self.db.envs[env_i[0]]
        init_belief = self.db.beliefs[sample_i[0]]
        db_step = self.db.steps[step_i[0]]

        init_state, init_action, _ = db_step

        # statistics: Success rate, trajectory length, collision rate, accumulated reward.
        # First row for expert, second row for evaluated policy.
        # results = np.zeros([self.repeats+1, 4], 'f')
        # results[0] = np.array([success, collided, wrong_opens, reward_sum], 'f')
        # print("expert performance:", results[0])
        results = np.zeros(
            [self.repeats * 2, 4],
            dtype=np.float32)
        processed_goal = self.task_param_processor.process_goals(goal)
        processed_init_belief = self.task_param_processor.process_beliefs(init_belief)

        for eval_i in range(self.repeats):
            expert_success, expert_collisions, \
            expert_wrong_open_times, expert_reward_sum, \
            learner_success, learner_collisions, \
            learner_wrong_open_times, \
            learner_reward_sum = simulate_policy(
                policy=self.policy,
                env_map=env,
                init_belief=init_belief,
                processed_init_belief=processed_init_belief,
                goal=goal,
                processed_goal=processed_goal,
                params=self.domain.params,
                init_state=init_state,
                init_action=init_action)
            expert_success = (1 if expert_success else 0)
            expert_collided = np.min([expert_collisions, 1])
            expert_wrong_open = np.min([expert_wrong_open_times, 1])
            learner_success = (1 if learner_success else 0)
            learner_collided = np.min([learner_collisions, 1])
            learner_wrong_open = np.min([learner_wrong_open_times, 1])

            results[eval_i] = np.array(
                [expert_success,
                 expert_collided,
                 expert_wrong_open,
                 expert_reward_sum], dtype=np.float32)
            results[eval_i + self.repeats] = np.array(
                [learner_success,
                 learner_collided,
                 learner_wrong_open,
                 learner_reward_sum], dtype=np.float32)
            # results[eval_i+1] = np.array([success, traj_len, collided, reward_sum], 'f')
            # print("network performance:", results[eval_i + 1])
        print("expert performance for %d repeats:" % self.repeats)
        print(results[:self.repeats])
        print("network performance for %d repeats:" % self.repeats)
        print(results[self.repeats:])

        return results  # success, traj_len, collided, reward_sum


class TrajDataFeed(dataflow.ProxyDataFlow):
    """
    Loads training data from database given batched samples.
    Inputs are batched samples of shape:
    [step_size, batch_size, 7]
    Each sample corresponds to:
    (sample_id, env_id, goal_state, step_id, traj_len)
    """

    def __init__(
            self,
            ds,
            get_db_func,
            domain,
            batch_size,
            step_size):
        super(TrajDataFeed, self).__init__(ds)
        self.get_db = get_db_func
        self.domain = domain
        self.batch_size = batch_size
        self.step_size = step_size

        self.traj_field_idx = 3

        self.db = None

        self.task_param_processer = TaskParamPreprocessor(self.domain)

    def __del__(self):
        self.close()

    def reset_state(self):
        super(TrajDataFeed, self).reset_state()
        if self.db is not None:
            print ("WARNING: reopening database. This is not recommended.")
            self.db.close()
        self.db = self.get_db()
        self.db.open()

    def close(self):
        if self.db is not None: self.db.close()
        self.db = None

    def get_data(self):
        for dp in self.ds.get_data():
            yield self.process_samples(dp)

    def process_samples(
            self,
            samples):
        """
        :param samples: numpy array, axis 0 for trajectory steps,
        axis 1 for batch, axis 2 for sample descriptor
        sample descriptor: (index (in original db), env_i,
        goal_states, step_i, b_index, traj_len, isstart)
        """
        sample_i, env_i, goals, step_i, traj_len, is_traj_head = [
            np.atleast_1d(x.squeeze())
            for x in np.split(samples, samples.shape[1], axis=1)]

        env_maps = self.db.envs[env_i]
        goal_maps = self.task_param_processer.process_goals(goals)

        # Initial belief
        init_belief = self.db.beliefs[:][sample_i]
        init_belief = self.task_param_processer.process_beliefs(init_belief)

        step_indices = step_i[None, :] + np.arange(self.step_size + 2)[:, None]
        step_indices = step_indices.clip(max=len(self.db.steps) - 1)

        # mask for valid steps vs zero padding
        valid_mask = np.nonzero(np.arange(self.step_size)[:, None] < traj_len[None, :])

        # actions
        step_idx_helper = step_indices[:self.step_size][valid_mask]
        label_idx_helper = step_indices[1:self.step_size + 1][valid_mask]

        input_action = np.zeros(
            (self.step_size, self.batch_size),
            dtype=np.int32)
        input_action[valid_mask] = self.db.steps[:][step_idx_helper, 1]

        label_action = np.zeros(
            input_action.shape,
            dtype=np.int32)
        label_action[valid_mask] = self.db.steps[:][label_idx_helper, 1]

        input_observation = np.zeros(
            (self.step_size, self.batch_size),
            dtype=np.int32)
        input_observation[valid_mask] = self.db.steps[:][step_idx_helper, 2]

        # Observation trajectories processed for soft indexing. Uncomment to activate.
        # linear_obs = self.db.steps[:][step_indices[:self.step_size][valid_mask], 2]
        # input_observation = np.zeros(
        #     (self.step_size, self.batch_size, self.domain.obs_dim), dtype=np.int32)
        # input_observation[valid_mask] = np.array(
        #     np.unravel_index(linear_obs, [2, 2, 2, 2])).transpose([1, 0])

        # set weights
        valid_step_mask = np.zeros(
            input_action.shape,
            dtype=np.float32)
        valid_step_mask[valid_mask] = 1.0

        return [env_maps, goal_maps, init_belief,
                is_traj_head, input_action,
                input_observation,
                valid_step_mask, label_action]


class OneShotData(dataflow.FixedSizeData):
    """
    Dataflow repeated after fixed number of samples
    """
    def size(self):
        return 1000000

    def get_data(self):
        with self._guard:
            while True:
                itr = self.ds.get_data()
                try:
                    for cnt in range(self._size):
                        yield next(itr)
                except StopIteration:
                    print ("End of dataset reached")
                    raise StopIteration


class TaskParamPreprocessor(object):
    def __init__(
            self,
            domain):
        self.domain = domain

    def process_goals(
            self,
            goals):
        """
        :param goals: linear goal state
        :return: goal map, same size as grid
        """
        goal_maps = np.zeros(
            [goals.shape[0], self.domain.N, self.domain.M],
            dtype=np.int32)
        idx = np.unravel_index(
            goals,
            [self.domain.N, self.domain.M])
        goal_maps[np.arange(goals.shape[0]), idx[0], idx[1]] = 1

        return goal_maps

    def process_beliefs(
            self,
            beliefs):
        """
        :param beliefs: belief in linear space
        :return: belief reshaped to num_cell x num_cell
        """
        batch = beliefs.shape[0] if beliefs.ndim > 1 else 1
        beliefs = beliefs.reshape(
            [batch, self.domain.N, self.domain.M])

        return beliefs.astype(np.float32)
