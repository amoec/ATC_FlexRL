# ATC-FlexRL: Mixed-Fidelity Reinforcement Learning for Air Traffic Conflict Resolution

Code for the paper [*Mixed-Fidelity Reinforcement Learning for Aircraft Conflict-Resolution*](https://www.sesarju.eu/sites/default/files/documents/sid/2025/papers/SIDs_2025_paper_17-final.pdf), presented at SESAR Innovation Days 2025.

Training reinforcement learning agents in a realistic air traffic simulator is slow and computationally expensive. This project tests whether that cost can be reduced by pre-training agents in a cheap, low-fidelity (LoFi) simulation and then transferring them to a realistic, high-fidelity (HiFi) one, a mixed-fidelity (MiFi) approach to closing the gap between fast simulation and realistic simulation.

## How it works

Training happens in two stages:

1. **LoFi pre-training.** An agent is trained in the `CR_LoFi` environment, either for a set number of steps or until it reaches a specified performance percentage. This environment is computationally cheap, so the agent learns the basics quickly.
2. **HiFi training.** The pre-trained agent is transferred to the `CR_HiFi` environment and training continues in the more realistic, more expensive simulation.

`run.py` automates the full pipeline and supports:

- multiple RL algorithms: PPO, A2C, SAC, DDPG and TD3;
- a sweep over LoFi pre-training levels, from 0% to 100%;
- baseline runs trained purely in LoFi or purely in HiFi, for comparison.

## Components

| Path | Purpose |
| --- | --- |
| `CR_LoFi/` | Low-fidelity conflict-resolution environment (Git submodule) |
| `CR_HiFi/` | High-fidelity conflict-resolution environment, built on BlueSky-Gym (Git submodule) |
| `run.py` | Orchestrates training runs across algorithms, pre-training shares and baselines |
| `analysis.py` | Aggregates results from `experiments/` and writes figures to `plots/` |
| `common/` | Shared callbacks, filters, metrics and plotting utilities |

## Quick start

Clone the repository together with its submodules:

```bash
git clone --recurse-submodules https://github.com/amoec/ATC_FlexRL.git
cd ATC_FlexRL
```

If you have already cloned it without submodules, run `git submodule update --init --recursive` instead.

Create a virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r CR_LoFi/requirements.txt
pip install -r CR_HiFi/requirements.txt
```

Run an experiment with PPO, a 24-hour budget, 10 pre-training increments and a moving-average window of 100:

```bash
python run.py --algo PPO --runtime 24 --n_incr 10 --window 100 --seed 123
```

## Arguments for `run.py`

| Argument | Required | Description |
| --- | --- | --- |
| `--algo` | yes | RL algorithm: `PPO`, `A2C`, `SAC`, `DDPG` or `TD3` |
| `--runtime` | yes | Total runtime budget for the experiment, in hours |
| `--n_incr` | yes | Number of increments for the LoFi pre-training percentage. With `5`, the levels tested are 0%, 20%, 40%, 60%, 80% and 100% |
| `--window` | yes | Window size for moving-average calculations |
| `--seed` | no | Base seed for reproducibility (default: `42`) |

## Output

Training logs, model checkpoints and timing data are written to:

```
experiments/ATC_RL.{seed}/
├── {ALGO}_ts.csv            # timing information per algorithm
├── LoFi-{ALGO}/             # LoFi training outputs
└── HiFi-{ALGO}/             # HiFi training outputs, with LoFi pre-training
```

`analysis.py` processes these results and saves figures to `plots/`. Aggregated results can be stored in `data/`.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{moec2025mifi,
  title     = {Mixed-Fidelity Reinforcement Learning for Aircraft Conflict-Resolution},
  author    = {Moec, Adam and Groot, Dieudonne Janthony and Ellerbroek, Joost},
  booktitle = {SESAR Innovation Days 2025},
  year      = {2025}
}
```

## Acknowledgements

The low-fidelity environment is adapted from [atcenv](https://github.com/jangroter/atcenv). The high-fidelity environment is built on [BlueSky-Gym](https://github.com/TUDelft-CNS-ATM/bluesky-gym) and the [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky) air traffic simulator. This work was carried out at the Control & Operations department of the Faculty of Aerospace Engineering, TU Delft.

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE). The submodules are separate repositories and carry their own licences.
