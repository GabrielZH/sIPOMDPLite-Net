import argparse
from configs.config_utils import boolean_arg


def get_args():
    parser = argparse.ArgumentParser()

    # Environment
    # -- Grid --
    parser.add_argument('--dsample_n', type=int, default=10, help='')
    parser.add_argument('--dsample_m', type=int, default=10, help='')

    # Agent
    # -- Models --
    parser.add_argument('--pr_move_succ', type=float, default=1.0, help='')
    parser.add_argument('--pr_obs_succ', type=float, default=1.0, help='')
    parser.add_argument('--R_open_goal', type=float, default=10.0, help='')
    parser.add_argument('--R_open_wrong', type=float, default=-5.0, help='')
    parser.add_argument('--R_obst', type=float, default=-10.0, help='')
    parser.add_argument('--R_move', type=float, default=-0.5, help='')
    parser.add_argument('--R_listen', type=float, default=-0.2, help='')
    parser.add_argument('--discount', type=float, default=0.99, help='')
    parser.add_argument('--reason_level', type=int, default=2, help='')

    #
    parser.add_argument('--db_path', type=str, help='')
    parser.add_argument('--lidar_map_lib_path', type=str, help='')
    parser.add_argument('--lidar_map_name', type=str, help='')
    parser.add_argument('--train_envs', type=int, default=10000, help='')
    parser.add_argument('--eval_envs', type=int, default=500, help='')
    parser.add_argument('--train_trajs_per_env', type=int, default=5, help='')
    parser.add_argument('--eval_trajs_per_env', type=int, default=1, help='')
    parser.add_argument('--parallel', type=boolean_arg, default=False, help='')
    parser.add_argument('--write_mode', type=str, default='append', help='')

    return parser.parse_args()
