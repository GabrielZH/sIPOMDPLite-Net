import argparse
import tables
import os

import numpy as np
import h5py as h5


def create_db(grid_n=6, grid_m=6):
    """
    :param grid_n:
    :param grid_m:
    """
    num_cell = grid_n * grid_m

    db = tables.open_file('./data.hdf5', mode='w')

    db.create_earray(
        db.root,
        name='envs',
        atom=tables.Int32Atom(),
        shape=(0, grid_n, grid_m))

    db.create_earray(
        db.root,
        name='valid_trajs',
        atom=tables.Int32Atom(),
        shape=(0,))

    db.create_earray(
        db.root,
        name='expRs',
        atom=tables.Float32Atom(),
        shape=(0,))

    db.create_earray(
        db.root,
        name='beliefs',
        atom=tables.Float32Atom(),
        shape=(0, 2, num_cell))

    db.create_earray(
        db.root,
        name='steps',
        atom=tables.Int32Atom(),
        shape=(0, 3))

    # env_id, goal_state, step_id, traj_length, collisions, failed
    db.create_earray(
        db.root,
        name='samples',
        atom=tables.Int32Atom(),
        shape=(0, 7))
    return db


def db_concatenate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_n', type=int, default=6, help='')
    parser.add_argument('--grid_m', type=int, default=6, help='')
    args = parser.parse_args()

    db_names = [f for f in os.listdir() if os.path.isfile(f) and f.endswith('.hdf5')]

    concat_db = create_db(args.grid_n, args.grid_m)

    for name in db_names:
        db = h5.File(name, 'r')
        for traj in db['valid_trajs']:
            concat_db.root.valid_trajs.append([traj + len(concat_db.root.samples)])
        for sample in db['samples']:
            rearranged_sample = np.array(
                [sample[0] + len(concat_db.root.envs), sample[1],
                 sample[2] + len(concat_db.root.steps), sample[3],
                 sample[4], sample[5], sample[6]], dtype='i')
            concat_db.root.samples.append(rearranged_sample[None])
        for belief in db['beliefs']:
            concat_db.root.beliefs.append(belief[None])
        for reward in db['expRs']:
            concat_db.root.expRs.append([reward])
        for step in db['steps']:
            concat_db.root.steps.append(step[None])
        for env in db['envs'][:]:
            concat_db.root.envs.append(env[None])
        db.close()


if __name__ == '__main__':
    db_concatenate()
