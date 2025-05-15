import os
import numpy as np
import csv
import sys

import datetime
from stable_baselines3.common.callbacks import BaseCallback

class PerformanceStoppingCallback(BaseCallback):
    """
    Stop training if the total episode reward (averaged over a moving window)
    crosses a threshold percentage, based on the min and max of episode rewards
    from a reference CSV log.
    """

    def __init__(self, training_amount, window, algo, type: str="LoFi", verbose=0):
        super(PerformanceStoppingCallback, self).__init__(verbose)
        self.training_amount = float(training_amount) if training_amount != 'full' else 'full'
        self.window = int(window)
        self.verbose = verbose

        if self.training_amount != 'full':
            # Load only the 'total_reward' column (index 2) from CSV
            self.reward_data_full = np.loadtxt(
                f'/scratch/amoec/ATC_RL/{type}-{algo}/{algo}_full/logs/results.csv',
                delimiter=',',
                skiprows=1,  # skip header
                usecols=2    # column index for total_reward
            )
            self.len_full = len(self.reward_data_full)
            # min_loss: average of last 5% of the full training logs
            self.min_loss = np.mean(np.abs(self.reward_data_full[-int(self.len_full * 0.05):]))
            # max_loss: average of first 5% of the full training logs
            self.max_loss = np.mean(np.abs(self.reward_data_full[:int(self.len_full * 0.05)]))
        else:
            self.min_loss = None
            self.max_loss = None
            
        self.check = max(self.window, self.len_full * 0.05)
        # List to store final episode rewards
        self.reward_history = []
        # Accumulator for the current episode’s rewards
        self.episode_reward = 0.0

    def _on_step(self) -> bool:
        """
        Called at every environment step. We add the step’s reward to
        self.episode_reward. Once the environment is 'done', we record
        the total for that episode and do our early-stopping check.
        """
        if self.training_amount == 'full':
            # User wants full training with no early stop
            return True

        # 'rewards' is a single-element list when n_envs = 1
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])

        # Since n_envs=1, just grab index 0
        reward = rewards[0]
        done = dones[0]

        # Accumulate the reward
        self.episode_reward += reward

        # If the episode ended
        if done:
            # Record final episode reward
            self.reward_history.append(self.episode_reward)
            # Reset for the next episode
            self.episode_reward = 0.0

            # Perform early-stopping checks if we have enough episodes
            if len(self.reward_history) > self.check:
                moving_avg = np.mean(self.reward_history[-self.check:])

                # Make sure we have valid min_loss/max_loss and a non-zero range
                if (self.min_loss is not None) and (self.max_loss is not None):
                    denom = self.max_loss - self.min_loss
                    if denom != 0:
                        pct_train = np.round((self.max_loss - abs(moving_avg)) / denom * 100, 2)

                        if self.verbose:
                            print(
                                f"Moving Avg of last {self.window} episodes: {moving_avg:.2f}, "
                                f"Min: {self.min_loss:.2f}, Max: {self.max_loss:.2f}, "
                                f"pct_train: {pct_train:.2f}%"
                            )

                        # Check threshold
                        if pct_train >= self.training_amount and abs(moving_avg) < self.max_loss:
                            if self.verbose:
                                print(f"Early stopping triggered at {pct_train:.2f}%")
                            return False

        # If we didn't stop, continue
        return True
    
class TimestepStoppingCallback(BaseCallback):
    """
    Stop training after a certain percentage of total timesteps have been reached."""
    def __init__(self, target, total_timesteps=3e6, verbose=0):
        super(TimestepStoppingCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.timesteps = 0
        self.target = target
        
    def _on_step(self) -> bool:
        if self.target == 'full':
            # User wants full training with no early stop
            return True
        
        self.timesteps += 1
        pct = self.timesteps / self.total_timesteps * 100
        
        if pct >= self.target:
            return False
        
        return True
class SafeLogCallback(BaseCallback):
    """
    Combined callback that logs per‐episode metrics to CSV and safely saves the model
    before timeout. Flushes every 25 episodes or at the end of training.
    """
    def __init__(self, model_path, log_dir, log_filename, timeout,
                 save_buffer=True, safety=15, flush_frequency=25, verbose=0):
        super().__init__(verbose)
        # SafeSave params
        self.model_path = model_path
        self.timeout = timeout
        self.safety = safety * 60  # sec
        self.save_buffer = save_buffer

        # CSV logger params
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_filename)
        self.headers = ["timesteps", "episode", "reward"]
        self.current_episode = 0
        self.buffer = []
        self.flush_frequency = flush_frequency
        self.episode_reward = 0.0
        self._write_header()

    def _write_header(self):
        if not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0:
            with open(self.log_path, "w") as f:
                csv.writer(f).writerow(self.headers)

    def _flush_buffer(self):
        if not self.buffer:
            return
        with open(self.log_path, "a") as f:
            writer = csv.writer(f)
            for m in self.buffer:
                writer.writerow([m[h] for h in self.headers])
        self.buffer = []

    def _early_stop(self) -> bool:
        return (self.timeout - datetime.datetime.now()).total_seconds() <= self.safety

    def _on_step(self) -> bool:
        # accumulate reward
        r = self.locals.get("rewards", [0])[0]
        done = self.locals.get("dones", [False])[0]
        self.episode_reward += r

        if not done:
            return True

        # episode ended → log, flush if needed, maybe early‐stop
        self.current_episode += 1
        self.buffer.append({
            "timesteps": self.num_timesteps,
            "episode": self.current_episode,
            "reward": self.episode_reward
        })

        if self.current_episode % self.flush_frequency == 0:
            self._flush_buffer()

        if self._early_stop():
            self._flush_buffer()
            self.model.save(self.model_path)
            if self.save_buffer:
                try:
                    self.model.save_replay_buffer(self.model_path + "_buffer")
                except AttributeError:
                    pass
            if self.verbose:
                print("Stopped: approaching timeout. Model & logs saved.")
            sys.exit(0)
            return False

        # reset for next episode
        self.episode_reward = 0.0
        return True

    def _on_training_end(self):
        # flush any remaining records
        self._flush_buffer()