# AgentForge: A Flexible Low-Code Platform for Reinforcement Learning Agent Design

**AgentForge** is a flexible low-code platform for designing Reinforcement Learning (RL) agents. It enables rapid testing of ideas by making it remarkably easier to optimize parameters. A modern RL agent often has several interconnected variables that need to be optimized, such as those related to the agent (e.g., rewards, noise levels), environment, and policy.  The platform implements Bayesian Optimization and Random Search strategies using **OPTUNA** alongside a custom Particle Swarm Optimization (PSO) algorithm for parameter optimization with Stable-Baselines3.

AgentForge was accepted in the **ICAART 2025** conference as a short paper. A preprint is available at http://arxiv.org/abs/2410.19528.

## Benefits of AgentForge
- **Joint optimization:** Optimize multiple parameter sets at once across your RL system.
- **Low-code setup:** Define and optimize RL parameters in one place with minimal code.
- **RL-specific:** Built with RL in mind—no need for deep expertise in parameter tuning.
- **Automated mapping:** Automatically maps parameters to model components, reducing manual work in specifying connections by hand.
- **Flexible optimizers:** Easily switch between Bayesian, Random Search, and PSO methods.
- **Accessible:** Suitable for non-experts, extending RL design to broader fields like cognitive science.
has context menu

## How to cite this work
```bash
@misc{fernandes_junior_agentforge_2024,
	title = {{AgentForge}: {A} {Flexible} {Low}-{Code} {Platform} for {Reinforcement} {Learning} {Agent} {Design}},
	shorttitle = {{AgentForge}},
	url = {http://arxiv.org/abs/2410.19528},
	urldate = {2024-10-28},
	publisher = {arXiv},
	author = {Fernandes Junior, Francisco Erivaldo and Oulasvirta, Antti},
	month = oct,
	year = {2024},
	keywords = {Computer Science - Machine Learning, Computer Science - Software Engineering},
}
```

## Installing AgentForge
### Prerequisites
1. **Install Anaconda, Miniconda, or Miniforge**:
   - Download and install one of the Conda distributions from:
     - [Anaconda](https://www.anaconda.com/products/distribution)
     - [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
     - [Miniforge](https://github.com/conda-forge/miniforge)
   - Follow the instructions for your operating system (Windows, macOS, or Linux).
  
### Installation Steps
1. **Clone the AgentForge Repository**:
   ```bash
   git clone https://github.com/yourusername/AgentForge.git
   cd AgentForge
   ```
2. **Create and Activate the Conda Environment**:
   - Create a Conda environment using the provided environment.yml file:
        ```bash
        conda env create -f conda-environment.yml
        ```
   - Activate the environment:
        ```bash
        conda activate AgentForge
        ```

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
- You can run individual optimization scripts by running the bash scripts inside the `run/` folder.
- By default, AgentForge will run in the background. To kill it use the following command: `pkill -f "AgentForge|python"`
