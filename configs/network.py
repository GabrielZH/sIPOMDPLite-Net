import argparse
from configs.config_utils import boolean_arg


def get_args():
    parser = argparse.ArgumentParser()

    # Read-and-write directories
    parser.add_argument('--db_path', help='')
    parser.add_argument('--save_to_path', help='')
    parser.add_argument('--load_model', nargs='*', help='')

    # Training
    # General
    parser.add_argument('--batch_size', type=int, default=100, help='')
    parser.add_argument('--step_size', type=int, default=5, help='')
    parser.add_argument('--train_envs_proportion', type=float, default=0.9, help='')
    parser.add_argument('--include_failed_trajs', type=boolean_arg, default=False, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--epochs', type=int, default=1000, help='')
    parser.add_argument('--init_patience_epochs', type=int, default=30, help='')
    parser.add_argument('--decay_patience_epochs', type=int, default=10, help='')
    parser.add_argument('--decay_steps', type=int, default=20, help='')

    # Models of sparse interaction framework
    parser.add_argument('--load_trained_sa_trans_func', type=boolean_arg, default=True, help='')
    parser.add_argument('--load_trained_sa_rwd_func', type=boolean_arg, default=True, help='')
    parser.add_argument('--load_trained_sa_obs_func', type=boolean_arg, default=True, help='')
    parser.add_argument('--use_prior_rwd_interaction_indicator', type=boolean_arg, default=False, help='')
    parser.add_argument('--use_prior_trans_interaction_indicator', type=boolean_arg, default=False, help='')

    # Domain dependent
    parser.add_argument('--K', type=int, default=-1, help='')
    parser.add_argument('--lim_traj_len', type=int, default=100, help='')

    # Evaluation
    parser.add_argument('--eval_samples', type=int, default=100, help='')
    parser.add_argument('--eval_repeats', type=int, default=1, help='')
    parser.add_argument('--eval_trials', type=int, default=1, help='')

    # Others
    parser.add_argument('--cache', nargs='*', default=['steps', 'envs', 'beliefs'], help='')

    return parser.parse_args()
