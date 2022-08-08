import argparse
from configs.config_utils import boolean_arg


def get_args():
    parser = argparse.ArgumentParser()

    # Read-and-write directories
    parser.add_argument('--db_path', help='The path to the database file storing domain data.')
    parser.add_argument('--save_to_path', help='The path to save the current model.')
    parser.add_argument('--load_model', nargs='*', help='The path to an existing trained model based on which the new '
                                                        'model is further trained.')

    # Training
    # General
    parser.add_argument('--batch_size', type=int, default=100, help='Literally.')
    parser.add_argument('--step_size', type=int, default=5, help='Number of consecutive steps for training RNN.')
    parser.add_argument('--train_envs_proportion', type=float, default=0.9, help='Proportion of the training data.')
    parser.add_argument('--include_failed_trajs', type=boolean_arg, default=False, help='If True, uses all generated expert trajectories'
                                                                                        'to train; otherwise, only uses the successful cases.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Literally.')
    parser.add_argument('--epochs', type=int, default=1000, help='Total number of epochs.')
    parser.add_argument('--init_patience_epochs', type=int, default=30, help='Number of epochs within which the loss'
                                                                             'no longer descends.')
    parser.add_argument('--decay_patience_epochs', type=int, default=10, help='Number of epochs per each decay step.')
    parser.add_argument('--decay_steps', type=int, default=20, help='Times that the learning rate is decayed after '
                                                                    'initial patience epochs.')

    # Models of sparse interaction framework
    parser.add_argument('--load_trained_sa_trans_func', type=boolean_arg, default=True, help='')
    parser.add_argument('--load_trained_sa_rwd_func', type=boolean_arg, default=True, help='')
    parser.add_argument('--load_trained_sa_obs_func', type=boolean_arg, default=True, help='')
    parser.add_argument('--use_prior_rwd_interaction_indicator', type=boolean_arg, default=False, help='')
    parser.add_argument('--use_prior_trans_interaction_indicator', type=boolean_arg, default=False, help='')

    # Domain dependent
    parser.add_argument('--K', type=int, default=-1, help='Number specified for the value iteration.')
    parser.add_argument('--lim_traj_len', type=int, default=100, help='Clip all trajectories to this length.')

    # Evaluation
    parser.add_argument('--eval_samples', type=int, default=100, help='Number of samples for evaluation.')
    parser.add_argument('--eval_repeats', type=int, default=1, help='Number of repeats for each task considering the '
                                                                    'stochasticity in simulations.')
    parser.add_argument('--eval_trials', type=int, default=1, help='Number of trials for simulations in evaluation, using'
                                                                   'to compute standard deviation.')

    # Others
    parser.add_argument('--cache', nargs='*', default=['steps', 'envs', 'beliefs'], help='Domain data to cache for better'
                                                                                         'efficiency.')

    return parser.parse_args()
