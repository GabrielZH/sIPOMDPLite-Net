import argparse
from configs.config_utils import boolean_arg


def get_args():
    parser = argparse.ArgumentParser()

    # Environment
    # -- Grid --
    parser.add_argument('--dsample_n', type=int, default=10, help='Target map length after resizing.')
    parser.add_argument('--dsample_m', type=int, default=10, help='Target map width after resizing.')

    # Agent
    # -- Models --
    parser.add_argument('--pr_move_succ', type=float, default=1.0, help='Probability of the agents successfully reaching'
                                                                        'the cell they intend to move. If not 1.0, then'
                                                                        'there is a (1.0 - pr_move_succ) chance that each'
                                                                        'of them remains in the previous cell.')
    parser.add_argument('--pr_obs_succ', type=float, default=1.0, help='Probability of the agents receiving a correct observation.')
    parser.add_argument('--R_open_goal', type=float, default=10.0, help='Reward of the agents successfully opening the '
                                                                        'door with gold.')
    parser.add_argument('--R_open_wrong', type=float, default=-5.0, help='Penalty of the agents choosing to open the door'
                                                                         'at a non-goal cell.')
    parser.add_argument('--R_obst', type=float, default=-10.0, help='Penalty of the agents colliding into an obstacle.')
    parser.add_argument('--R_move', type=float, default=-0.5, help='Cost of the agents moving without colliding into any obstacle.')
    parser.add_argument('--R_listen', type=float, default=-0.2, help='Cost of the agents listening.')
    parser.add_argument('--discount', type=float, default=0.99, help='The discounted factor in Bellman equation.')
    parser.add_argument('--reason_level', type=int, default=2, help='Reasoning level of the subjective agent.')

    #
    parser.add_argument('--db_path', type=str, help='The path to store the store generated training or test data.')
    parser.add_argument('--lidar_map_lib_path', type=str, help='The local path to LiDAR maps.')
    parser.add_argument('--lidar_map_name', type=str, help='Name of the LiDAR map.')
    parser.add_argument('--train_envs', type=int, default=10000, help='Number of grid environments for training.')
    parser.add_argument('--eval_envs', type=int, default=500, help='Number of grid environments for evaluation.')
    parser.add_argument('--train_trajs_per_env', type=int, default=5, help='Number of trajectories generated for each'
                                                                           'grid environment for training.')
    parser.add_argument('--eval_trajs_per_env', type=int, default=1, help='Number of trajectories generated for each grid'
                                                                          'environment for evaluation.')
    parser.add_argument('--parallel', type=boolean_arg, default=False, help='If true, generates data parallely.')
    parser.add_argument('--write_mode', type=str, default='append', help='If append, continues to generate data based on'
                                                                         'existing ones; otherwise, deletes old ones and '
                                                                         're-generates.')

    return parser.parse_args()
