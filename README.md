# sIPOMDPLite-Net
Lightweight, Self-Interested Learning and Planning in POSGs with Sparse Interactions.
# Data Generation
```
python domains/tiger_grid/multi_agent/single_interaction_condition/expert_traj_generator.py --db_path PATH_TO_GENERATED_DATA --train_envs NUM_TRAIN_ENVS 
--eval_envs NUM_TEST_ENVS --train_trajs_per_env NUM_TRAJECTORIES_PER_TRAIN_ENVS --eval_trajs_per_env NUM_TRAJECTORIES_PER_TEST_ENVS --grid_n GRID_LENGTH 
--grid_m GRID_WIDTH
```
Arguments listed in the above command are required. For all available arguements including optional ones, please refer to the domain configuration file [tiger_grid.py](./configs/envs/tiger_grid.py)
# Model Training
To train the network from scratch:
```
python main.py --db_path PARENT_DIR_TO_TRAINING_DATA --load_model PATH_TO_EXISTING_MODEL
```
To further train an existing model:
```
python main.py --db_path PARENT_DIR_TO_TRAINING_DATA --load_model PATH_TO_EXISTING_MODEL --save_to_path PATH_TO_NEW_MODEL
```
Arguments listed in the above command are required. For other optional ones, please refer to __Training__ in domain configuration file [network.py](./configs/network.py).
# Model Evaluation
```
python main.py --db_path PARENT_DIR_TO_TEST_DATA --load_model PATH_TO_EXISTING_MODEL --epochs 0
```
Arguments listed in the above command are required. For other optional ones, please refer to __Evaluation__ in domain configuration file [network.py](./configs/network.py).
