# AgentForge: A Flexible Low-Code Platform for Reinforcement Learning Agent Design

This repository contains **AgentForge**, a flexible low-code platform designed for Reinforcement Learning (RL) agent design, which was presented in a paper published at **ICAART 2025**. The platform implements a custom Particle Swarm Optimization (PSO) algorithm for hyperparameter optimization in RL agents using stable baselines3. Additionally, **OPTUNA** is included as a baseline for hyperparameter optimization, supporting Bayesian Optimization (Tree-structured Parzen Estimator) and Random Search strategies.

## Running AgentForge
To perform parameter optimization using AgentForge:

1. **Provide your environment**:
   - Put your custom environment inside the folder `environments/`. You should follow the Gymnasium API when creating your environments.
   - Provide a `run/env_configs/<name>_config.py` file containing the following functions:
      1. `user_train_environment(seed, n_train_envs, **environment_kwargs)`:
         - Sets up the training environment for reinforcement learning using user-specified parameters.
         - Should return a vectorized training environment with common wrappers for observation normalization, stacking frames, and monitoring.
      2. `user_evaluate_policy(seed, model, **environment_kwargs)`:
         - Evaluates the policy of the trained model in a separate evaluation environment.
         - Should return a mean objective value (such as average reward) and additional user-defined metrics.

      3. `user_record_video(seed, model_path, video_folder, environment_kwargs, num_episodes=20, video_length=10000)`:
         - Optionally records video of the agent's performance in the environment.
         - Videos can be recorded for a specified number of episodes, with each video lasting a defined number of frames.
      
2. **Configuration File**:
   - Modify the configuration file (`config_files/config.yaml`) to suit your project's specific requirements and environment setup.
   - Configure parameters such as optimization algorithms, number of trials, hyperparameter bounds, etc.
3. **Execute Optimization**:
   - Run the `run-agentforge.sh` script to start the main PSO process:
      ```bash
      ./run-agentforge.sh
      ```

## Monitoring and Evaluation:
   - Monitor progress and results in the `./results/logs/` directory.
   - Optimization results are saved on the specified database (`database_url` in `config_files/config.yaml`).

## Directory Structure
- `config_files/`: Configuration files for PSO and OPTUNA.
- `environments/`: User-defined custom RL environments.
- `optimizers/`: PSO and OPTUNA optimizer implementations.
- `run/`: Scripts for running parameter optimization.
   - `env_config/`: Scripts to set up the user training environment and evaluation function.
      - `PixelLunarLander_config.py`: Code for the pixel-based Lunar Lander agent.
      - `SampleEnv_config.py`: Sample code showing how to create the necessary functions for a custom environment.
   - `pso-individual-evaluator.py`: Python PSO evaluator process script.
   - `pso-main-loop.py`: Python main PSO process script.
   - `optuna-main.py`: Python script to run Random Search or Bayesian Optimization using OPTUNA.
   - `run-optuna.sh`: Bash script to run the OPTUNA optimization.
   - `run-pso-evaluators.sh`: Bash script to run PSO evaluators.
   - `run-pso-main.sh`: Bash script to run the main PSO process.
- `run-agentforge.sh`: **Main bash script to run AgentForge optimization using the configuration from config.yaml. It will call the scripts inside the `run/` folder as needed.**
- `conda-environment.yml`: Anaconda environment file.

## Notes
- Use the Anaconda package manager to set up and run the provided code.
- You can run individual optimization scripts by running the bash scripts inside the `run/` folder.
- By default, AgentForge will run in the background. To kill it use the following command: `pkill -f "AgentForge|python"`
