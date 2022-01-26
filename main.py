"""
Script to train SIPOMDPLite-net and evaluate the learned policy
"""

import _pickle as pickle
import time
import sys
import os

import numpy as np
import tensorflow as tf

sys.path.append('/home/gz67063/projects/sipomdplite-net')

from data_processing import ma_tiger_grid_data_feeding
from network.train import SIPOMDPLiteNet
from network.policy import SIPOMDPLiteNetPolicy
from configs.network import get_args
from utils.envs.dotdict import dotdict


def run_training(params):
    """
    Train sipomdp-lite-net.
    """
    # build dataflows
    datafile = os.path.join(params.db_path, "train/data.hdf5")
    train_feed = ma_tiger_grid_data_feeding.Datafeed(
        params,
        filename=datafile,
        mode="train",
        max_env=params.train_envs_proportion)
    valid_feed = ma_tiger_grid_data_feeding.Datafeed(
        params,
        filename=datafile,
        mode="valid",
        min_env=params.train_envs_proportion)

    # get cache for training data
    train_cache = train_feed.build_cache()

    df_train = train_feed.build_dataflow(
        params.batch_size, params.step_size,
        cache=train_cache)
    # restart after full validation set
    df_valid = valid_feed.build_dataflow(
        params.batch_size, params.step_size,
        cache=train_cache)

    df_train.reset_state()
    time.sleep(0.2)
    df_valid.reset_state()
    time.sleep(0.2)

    train_iterator = df_train.get_data()
    valid_iterator = df_valid.get_data()

    # built model into the default graph
    with tf.Graph().as_default():
        # build network for training
        network = SIPOMDPLiteNet(
            params,
            batch_size=params.batch_size,
            step_size=params.step_size)
        network.build_inference()  # build graph for inference including loss
        network.build_train(params.learning_rate)  # build training ops

        # build network for evaluation
        # network_pred = QMDPNet(params, batch_size=1, step_size=1)
        # network_pred.build_inference(reuse=True)

        # Create a saver for writing training checkpoints.
        saver = tf.compat.v1.train.Saver(
            var_list=tf.compat.v1.trainable_variables(),
            max_to_keep=100)

        # Get initialize Op
        init = tf.compat.v1.global_variables_initializer()

        # Create a TF session
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

        # Run the Op to initialize variables
        sess.run(init)

        # load previously saved model
        if params.load_model:
            print("Loading from " + params.load_model[0])
            loader = tf.compat.v1.train.Saver(
                var_list=tf.compat.v1.trainable_variables())
            loader.restore(sess, params.load_model[0])
        summary_writer = tf.compat.v1.summary.FileWriter(
            logdir=params.save_to_path,
            graph=sess.graph)
        summary_writer.flush()

    epoch = -1
    best_epoch = 0
    no_improvement_epochs = 0
    patience = params.init_patience_epochs  # initial patience
    decay_step = 0
    valid_losses = list()

    start_time = time.time()

    for epoch in range(params.epochs):
        training_loss = 0.0
        for step in range(train_feed.steps_in_epoch):
            data = train_iterator.__next__()
            feed_dict = {network.placeholders[i]: data[i]
                         for i in range(len(network.placeholders))}

            _, loss, _ = sess.run(
                [network.train_op, network.loss, network.update_belief_op],
                feed_dict=feed_dict)
            training_loss += loss

        # save belief and restore it after validation
        belief = sess.run([network.belief])[0]

        # accumulate loss over the enitre validation set
        valid_loss = 0.0
        for step in range(valid_feed.steps_in_epoch):
            data = valid_iterator.__next__()
            # assert step > 0 or np.isclose(data[3], 1.0).all()
            feed_dict = {network.placeholders[i]: data[i]
                         for i in range(len(network.placeholders))}
            loss, _ = sess.run(
                [network.loss, network.update_belief_op],
                feed_dict=feed_dict)
            valid_loss += loss

        tf.compat.v1.assign(network.belief, belief)

        training_loss /= train_feed.steps_in_epoch
        valid_loss /= valid_feed.steps_in_epoch

        # print status
        lr = sess.run([network.learning_rate])[0]
        print('Epoch %d, lr=%f, training loss=%.3f, valid loss=%.3f' %
              (epoch, lr, training_loss, valid_loss))

        valid_losses.append(valid_loss)
        best_epoch = np.array(valid_losses).argmin()

        # save a checkpoint if needed
        if best_epoch == epoch or epoch == 0:
            best_model = saver.save(
                sess,
                os.path.join(params.save_to_path, 'model.chk'),
                global_step=epoch)
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        # check for early stopping
        if no_improvement_epochs > patience:
            # finish training if learning rate decay steps reached
            if decay_step >= params.decay_steps:
                break
            decay_step += 1
            no_improvement_epochs = 0

            # restore best model found so far
            saver.restore(sess, best_model)

            # decay learning rate
            sess.run(tf.compat.v1.assign(network.decay_step, decay_step))
            learning_rate = network.learning_rate.eval(session=sess)
            print("Decay step %d, learning rate = %f" % (decay_step, learning_rate))

            # use smaller patience for future iterations
            patience = params.decay_patience_epochs

    # Training done
    end_time = time.time()
    duration = end_time - start_time
    epoch += 1
    print("Training loop over after %d epochs" % epoch)
    print("Total time: %.3f" % duration)

    # restore best model
    if best_epoch != epoch:
        print("Restoring %s from epoch %d" % (str(best_model), best_epoch))
        saver.restore(sess, best_model)

    # save best model
    checkpoint_file = os.path.join(params.save_to_path, 'final.chk')
    saver.save(sess, checkpoint_file)

    return checkpoint_file


def run_eval(params, modelfile):
    # built model into the default graph
    with tf.Graph().as_default():
        # build network for evaluation
        network = SIPOMDPLiteNet(params, batch_size=1, step_size=1)
        network.build_inference()

        # Create a saver for loading checkpoint
        saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.trainable_variables())

        # Create a TF session
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # use CPU
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto())

        # load model from file
        saver.restore(sess, modelfile)

        # policy
        policy = SIPOMDPLiteNetPolicy(network, sess)

    # Store the mean statistics of each trial
    expert_trials_succ_rate = list()
    expert_trials_colli_rate_succ = list()
    expert_trials_accum_reward_succ = list()
    expert_trials_traj_len_succ = list()
    expert_trials_colli_rate_all = list()
    expert_trials_accum_reward_all = list()
    expert_trials_traj_len_all = list()
    network_trials_succ_rate = list()
    network_trials_colli_rate_succ = list()
    network_trials_accum_reward_succ = list()
    network_trials_traj_len_succ = list()
    network_trials_colli_rate_all = list()
    network_trials_accum_reward_all = list()
    network_trials_traj_len_all = list()

    for _ in range(params.eval_trials):
        # build dataflows
        eval_feed = ma_tiger_grid_data_feeding.Datafeed(
            params,
            filename=os.path.join(params.db_path, "eval/data.hdf5"),
            mode="eval")
        df = eval_feed.build_eval_dataflow(
            policy=policy,
            repeats=params.eval_repeats)
        df.reset_state()
        time.sleep(0.1)
        eval_iterator = df.get_data()

        print("Evaluating %d samples, repeating simulation %d time(s)" %
              (params.eval_samples, params.eval_repeats))
        expert_results = list()
        network_results = list()
        for eval_i in range(params.eval_samples):
            res = eval_iterator.__next__()
            # expert_results.append(res[:1]) # success, traj_len, collided, reward_sum
            # network_results.append(res[1:])
            expert_results.append(res[:params.eval_repeats])
            network_results.append(res[params.eval_repeats:])

        def print_results(results):
            results = np.concatenate(results, axis=0)
            succ = results[:, 0]
            traj_len_succ = results[succ > 0, 1]
            traj_len_all = results[:, 1]
            collided_succ = results[succ > 0, 2]
            collided_all = results[:, 2]
            reward_succ = results[succ > 0, 3]
            reward_all = results[:, 3]
            mean_succ = np.mean(succ)
            mean_traj_len_succ = np.mean(traj_len_succ)
            mean_reward_succ = np.mean(reward_succ)
            mean_collided_succ = np.mean(collided_succ)
            mean_traj_len_all = np.mean(traj_len_all)
            mean_reward_all = np.mean(reward_all)
            mean_collided_all = np.mean(collided_all)
            print("Success rate: %.3f\n"
                  "- Successful Cases -\n"
                  "Collision rate: %.3f\n"
                  "Accumulated reward: %.1f\n"
                  "Trajectory length: %.1f\n"
                  "- All Test Data -\n"
                  "Collision rate: %.3f\n"
                  "Accumulated reward: %.1f\n"
                  "Trajectory length: %.1f\n" %
                  (mean_succ, mean_collided_succ, mean_reward_succ, mean_traj_len_succ,
                   mean_collided_all, mean_reward_all, mean_traj_len_all))

            return mean_succ, mean_collided_succ, mean_reward_succ, mean_traj_len_succ, \
                   mean_collided_all, mean_reward_all, mean_traj_len_all

        print("Expert")
        expert_succ_rate, expert_collided_succ, expert_reward_succ, expert_traj_len_succ, \
        expert_collided_all, expert_reward_all, expert_traj_len_all = print_results(expert_results)
        print("IPOMDPLite-Net")
        network_succ_rate, network_collided_succ, network_reward_succ, network_traj_len_succ, \
        network_collided_all, network_reward_all, network_traj_len_all = print_results(network_results)

        expert_trials_succ_rate.append(expert_succ_rate)
        expert_trials_colli_rate_succ.append(expert_collided_succ)
        expert_trials_accum_reward_succ.append(expert_reward_succ)
        expert_trials_traj_len_succ.append(expert_traj_len_succ)
        expert_trials_colli_rate_all.append(expert_collided_all)
        expert_trials_accum_reward_all.append(expert_reward_all)
        expert_trials_traj_len_all.append(expert_traj_len_all)
        network_trials_succ_rate.append(network_succ_rate)
        network_trials_colli_rate_succ.append(network_collided_succ)
        network_trials_accum_reward_succ.append(network_reward_succ)
        network_trials_traj_len_succ.append(network_traj_len_succ)
        network_trials_colli_rate_all.append(network_collided_all)
        network_trials_accum_reward_all.append(network_reward_all)
        network_trials_traj_len_all.append(network_traj_len_all)

    stddev_expert_succ_rate = np.std(expert_trials_succ_rate)
    stddev_expert_colli_rate_succ = np.std(expert_trials_colli_rate_succ)
    stddev_expert_accum_reward_succ = np.std(expert_trials_accum_reward_succ)
    stddev_expert_traj_len_succ = np.std(expert_trials_traj_len_succ)
    stddev_expert_colli_rate_all = np.std(expert_trials_colli_rate_all)
    stddev_expert_accum_reward_all = np.std(expert_trials_accum_reward_all)
    stddev_expert_traj_len_all = np.std(expert_trials_traj_len_all)
    stddev_network_succ_rate = np.std(network_trials_succ_rate)
    stddev_network_colli_rate_succ = np.std(network_trials_colli_rate_succ)
    stddev_network_accum_reward_succ = np.std(network_trials_accum_reward_succ)
    stddev_network_traj_len_succ = np.std(network_trials_traj_len_succ)
    stddev_network_colli_rate_all = np.std(network_trials_colli_rate_all)
    stddev_network_accum_reward_all = np.std(network_trials_accum_reward_all)
    stddev_network_traj_len_all = np.std(network_trials_traj_len_all)

    print("standard deviations for all stats:")
    print("- expert -")
    print("success rate: %.3f\n"
          "collision rate of successful cases: %.3f\n"
          "accumulated reward of successful cases: %.1f\n"
          "trajectory length of successful cases: %.1f\n"
          "collision rate of all cases: %.3f\n"
          "accumulated reward of all cases: %.1f\n"
          "trajectory length of all cases: %.1f\n" %
          (stddev_expert_succ_rate,
           stddev_expert_colli_rate_succ,
           stddev_expert_accum_reward_succ,
           stddev_expert_traj_len_succ,
           stddev_expert_colli_rate_all,
           stddev_expert_accum_reward_all,
           stddev_expert_traj_len_all))
    print("- IPOMDPLite-net -")
    print("success rate: %.3f\n"
          "collision rate of successful cases: %.3f\n"
          "accumulated reward of successful cases: %.1f\n"
          "trajectory length of successful cases: %.1f\n"
          "collision rate of all cases: %.3f\n"
          "accumulated reward of all cases: %.1f\n"
          "trajectory length of all cases: %.1f\n" %
          (stddev_network_succ_rate,
           stddev_network_colli_rate_succ,
           stddev_network_accum_reward_succ,
           stddev_network_traj_len_succ,
           stddev_network_colli_rate_all,
           stddev_network_accum_reward_all,
           stddev_network_traj_len_all))


def main():
    network_args = get_args()
    args = dotdict(
        pickle.load(
            open(
                os.path.join(
                    network_args.db_path, 'train/params.pickle'), 'rb')))
    if network_args.K < 0:
        network_args.K = args.num_cell

    for key in vars(network_args):
        args[key] = getattr(network_args, key)

    print(args)

    if args.epochs == 0:
        assert len(args.load_model) == 1
        model = args.load_model[0]
    else:
        model = run_training(args)

    run_eval(args, model)


if __name__ == '__main__':
    main()  # skip filename
