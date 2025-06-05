# Mixed-Fidelity Air Traffic Control RL Training

This repository contains the code for a research project investigating the feasibility and effectiveness of Mixed-Fidelity (MiFi) training for Reinforcement Learning (RL) agents in Air Traffic Management (ATM) conflict resolution tasks. The project explores how pre-training RL agents in a low-fidelity (LoFi) environment can accelerate and improve performance when subsequently trained in a high-fidelity (HiFi) environment.

## Project Overview

The core idea is to leverage a computationally inexpensive LoFi simulation for initial learning and then transfer this knowledge to a more complex and realistic HiFi simulation. This approach aims to reduce the overall training time and computational resources required to develop effective RL-based conflict resolution strategies for air traffic control.

The project includes:
-   **LoFi Environment (`CR_LoFi/`)**: A simplified air traffic simulation environment.
-   **HiFi Environment (`CR_HiFi/`)**: A more detailed and realistic air traffic simulation environment.
-   **Experiment Orchestration (`run.py`)**: A script to manage and automate training runs across different configurations, RL algorithms, and MiFi strategies.
-   **Analysis Tools** (implied by `analysis.py` and `plots/`): Scripts and utilities for processing experiment results and generating visualizations.
-   **Common Utilities (`common/`)**: Shared code for callbacks, filters, metrics, and plotting used across the project.

## Directory Structure

```
.
├── README.md
├── requirements.txt        # Python dependencies
├── run.py                  # Main script to run experiments
├── analysis.py             # Script for analyzing results
├── common/                 # Shared utilities
│   ├── callbacks.py
│   ├── filters.py
│   ├── metrics.py
│   └── plot.py
├── CR_LoFi/                # Low-Fidelity environment and training code
│   ├── main.py             # Main script for LoFi component
│   ├── atcenv/             # LoFi ATC environment source
│   └── ...
├── CR_HiFi/                # High-Fidelity environment and training code
│   ├── main.py             # Main script for HiFi component
│   ├── bluesky_gym/        # HiFi ATC environment
│   └── ...
├── experiments/            # Stores raw results, logs, and model checkpoints
│   ├── ATC_RL.{seed}/      # Results for a specific base seed
│   │   ├── {ALGO}_ts.csv   # Timing information for algorithms
│   │   ├── LoFi-{ALGO}/    # LoFi training outputs
│   │   └── HiFi-{ALGO}/    # HiFi training outputs (with LoFi pre-training)
│   └── ...
├── plots/                  # Stores generated plots and visualizations
│   └── ...
└── data/                   # Stores aggregated/pickled results (optional)
```

**Note**: `CR_LoFi` and `CR_HiFi` are Git submodules. You will need to initialize them after cloning this repository.

## Methodology

The MiFi training approach involves two main stages:

1.  **LoFi Pre-training**: An RL agent is trained in the `CR_LoFi` environment for a certain number of steps or up to a specified performance percentage. This environment is designed to be computationally efficient, allowing for rapid initial learning.
2.  **HiFi Training/Fine-tuning**: The pre-trained agent is then transferred to the `CR_HiFi` environment. Training continues in this more realistic but computationally intensive environment.

The `run.py` script automates this process, allowing for experiments with:
-   Different RL algorithms (e.g., PPO, A2C, SAC, DDPG, TD3).
-   Varying levels of LoFi pre-training (controlled by percentages).
-   Baseline runs (training purely in HiFi or LoFi).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/amoec/ATC_FlexRL.git
    cd ATC-FlexRL
    ```

2.  **Initialize Submodules:**
    ```bash
    git submodule update --init --recursive
    ```

3.  **Install Dependencies:**
    It's recommended to use a virtual environment (e.g., `venv` or `conda`).
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
    The `CR_LoFi` and `CR_HiFi` directories might have their own `requirements.txt` file. Ensure their dependencies are also installed. For example:
    ```bash
    pip install -r CR_LoFi/requirements.txt
    ```

## Running Experiments

The primary script for running experiments is `run.py`. It orchestrates the training in both LoFi and HiFi environments based on the specified arguments.

**Key Arguments for `run.py`:**

*   `--algo <ALGORITHM_NAME>`: Specifies the RL algorithm to use (e.g., `PPO`, `A2C`, `SAC`). This is a required argument.
*   `--runtime <HOURS>`: Total allocated runtime for the experiment in hours. This is a required argument.
*   `--n_incr <NUMBER>`: Number of increments for LoFi pre-training percentages (e.g., if `n_incr=5`, it tests 0%, 20%, ..., 100% LoFi pre-training). This is a required argument.
*   `--window <SIZE>`: Window size for moving average calculations. This is a required argument.
*   `--seed <SEED_VALUE>`: Base seed for reproducibility (default: `42`).

**Example Usage:**

To run an experiment with the PPO algorithm, a 24-hour runtime, 10 pre-training increments, and a window size of 100:
```bash
python run.py --algo PPO --runtime 24 --n_incr 10 --window 100 --seed 123
```

### Experiment Output

-   **Logs and Models**: Training logs, saved models, and other artifacts are typically saved in the `experiments/` directory. The structure is usually `experiments/ATC_RL.{base_seed}/{LoFi|HiFi}-{algo}/{algo}_{percentage_or_run_id}/`.

## Analysis and Plotting

-   The `analysis.py` script is used to process the data from the `experiments/` directory.
-   Generated plots and visualizations are stored in the `plots/` directory.
