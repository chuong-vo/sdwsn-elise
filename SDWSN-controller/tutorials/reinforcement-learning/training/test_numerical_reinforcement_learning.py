#!/usr/bin/python3
#
# Copyright (C) 2022  Fernando Jurado-Lasso <ffjla@dtu.dk>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import os
import sys
import csv
import time

import torch
import gymnasium as gym


from sdwsn_controller.config import SDWSNControllerConfig, CONTROLLERS
from sdwsn_controller.reinforcement_learning.wrappers\
    import SaveOnBestTrainingRewardCallback

from stable_baselines3.common.monitor import Monitor


from stable_baselines3 import PPO


CONFIG_FILE = "numerical_controller_rl.json"


class MetricsLoggerEnv(gym.Wrapper):
    """Wrap environment to log info metrics into a CSV file."""

    def __init__(self, env, csv_path, phase):
        super().__init__(env)
        self.csv_path = csv_path
        self.phase = phase
        self.step_idx = 0
        fieldnames = [
            "timestamp", "phase", "step",
            "alpha", "beta", "delta",
            "reward", "power_normalized", "delay_normalized",
            "pdr_mean", "current_sf_len", "last_ts_in_schedule"
        ]
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._log_file = open(csv_path, 'a', newline='')
        self._writer = csv.DictWriter(self._log_file, fieldnames=fieldnames)
        if self._log_file.tell() == 0:
            self._writer.writeheader()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_idx += 1
        controller = getattr(self.env.unwrapped, 'controller', None)
        row = {
            "timestamp": time.time(),
            "phase": self.phase,
            "step": self.step_idx,
            "alpha": getattr(controller, 'alpha', None),
            "beta": getattr(controller, 'beta', None),
            "delta": getattr(controller, 'delta', None),
            "reward": info.get('reward'),
            "power_normalized": info.get('power_normalized'),
            "delay_normalized": info.get('delay_normalized'),
            "pdr_mean": info.get('pdr_mean'),
            "current_sf_len": info.get('current_sf_len'),
            "last_ts_in_schedule": info.get('last_ts_in_schedule')
        }
        self._writer.writerow(row)
        self._log_file.flush()
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def close(self):
        try:
            self._log_file.close()
        finally:
            super().close()


def train(env, log_dir, callback):
    """
    Just use the PPO algorithm.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training PPO on device: {device}")
    model = PPO("MlpPolicy", env,
                tensorboard_log=log_dir, verbose=1,
                seed=123,
                device=device)

    model.learn(total_timesteps=int(5e6),
                tb_log_name='training', callback=callback)
    # Let's save the model
    base_path = os.path.join(log_dir, "ppo_sdwsn")
    model.save(base_path)

    del model  # remove to demonstrate saving and loading

    best_model_path = None
    callback_path = getattr(callback, 'save_path', None)
    if callback_path:
        candidate = callback_path + ".zip"
        if os.path.exists(candidate):
            best_model_path = candidate

    return base_path + ".zip", best_model_path


def evaluation(env, model_path):
    model = PPO.load(model_path)

    total_reward = 0

    # Test the trained agent
    for _ in range(50):
        obs, _ = env.reset()
        done = False
        acc_reward = 0
        # Get last observations non normalized
        observations = env.controller.get_state()
        assert 0 <= observations['alpha'] <= 1
        assert 0 <= observations['beta'] <= 1
        assert 0 <= observations['delta'] <= 1
        assert observations['last_ts_in_schedule'] > 1
        # get requirements
        user_req_type = env.controller.user_requirements_type
        match user_req_type:
            case 'energy':
                env.controller.current_slotframe_size = env.controller.last_tsch_link+5
            case 'delay':
                env.controller.current_slotframe_size = env.max_slotframe_size-5
            case 'pdr':
                env.controller.current_slotframe_size = env.max_slotframe_size-5
            case 'balanced':
                env.controller.current_slotframe_size = env.max_slotframe_size-5
            case _:
                print("Unknow user requirements.")
        while (not done):
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                obs, reward, done, _ = step_result
            # Get last observations non normalized
            observations = env.controller.get_state()
            acc_reward += reward
            # Get last observations non normalized
            observations = env.controller.get_state()
            assert 0 <= observations['alpha'] <= 1
            assert 0 <= observations['beta'] <= 1
            assert 0 <= observations['delta'] <= 1
            assert observations['last_ts_in_schedule'] > 1
            if done:
                total_reward += acc_reward

    # Total reward, for this scenario, should be above 65.
    assert total_reward/50 > 64


def main():
    """
    This test the training, loading and testing of RL env.
    We dont use DB to avoid reducing the processing speed
    """
    # ----------------- RL environment, setup --------------------
    # Create output folder
    output_folder = './output/'
    os.makedirs(output_folder, exist_ok=True)

    # Monitor the environment
    log_dir = './tensorlog/'
    os.makedirs(log_dir, exist_ok=True)
    # Monitor the environment
    monitor_log_dir = './trained_model/'
    os.makedirs(monitor_log_dir, exist_ok=True)
    # Metrics logging
    metrics_dir = './metrics/'
    os.makedirs(metrics_dir, exist_ok=True)
    train_metrics_file = os.path.join(metrics_dir, 'train_metrics.csv')
    eval_metrics_file = os.path.join(metrics_dir, 'eval_metrics.csv')
    for metrics_file in (train_metrics_file, eval_metrics_file):
        if os.path.exists(metrics_file):
            os.remove(metrics_file)
    # -------------------- setup controller ---------------------
    config = SDWSNControllerConfig.from_json_file(CONFIG_FILE)
    controller_class = CONTROLLERS[config.controller_type]
    controller = controller_class(config)
    # ----------------- RL environment ----------------------------
    train_env = controller.reinforcement_learning.env
    train_env = MetricsLoggerEnv(train_env, train_metrics_file, phase='train')
    train_env = Monitor(train_env, monitor_log_dir)
    # Train the agent
    best_model = SaveOnBestTrainingRewardCallback(
        check_freq=1000, log_dir=monitor_log_dir)
    model_path, best_model_path = train(train_env, log_dir, callback=best_model)
    train_env.close()
    # ----------------- Test environment ----------------------------
    test_env = controller.reinforcement_learning.env
    test_env = MetricsLoggerEnv(test_env, eval_metrics_file, phase='eval')
    evaluation_path = best_model_path if best_model_path else model_path
    evaluation(test_env, evaluation_path)
    test_env.close()
    controller.stop()
    # Delete folders
    # try:
    #     shutil.rmtree(output_folder)
    # except OSError as e:
    #     print("Error: %s - %s." % (e.filename, e.strerror))
    # try:
    #     shutil.rmtree(log_dir)
    # except OSError as e:
    #     print("Error: %s - %s." % (e.filename, e.strerror))


if __name__ == '__main__':
    main()
    sys.exit(0)
