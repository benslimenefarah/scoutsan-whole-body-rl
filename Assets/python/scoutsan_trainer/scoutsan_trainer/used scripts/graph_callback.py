

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

SUCCESS_RADIUS = 0.05

class GraphSnapshotCallback(BaseCallback):
    def __init__(self, save_dir: str, snapshot_freq: int = 50_000,
                 success_radius: float = SUCCESS_RADIUS,
                 gripper_slice=(0, 3), target_slice=(7, 10), verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.snapshot_freq = snapshot_freq
        self.success_radius = success_radius
        self.g0, self.g1 = gripper_slice
        self.t0, self.t1 = target_slice
        os.makedirs(self.save_dir, exist_ok=True)

        self.episode_rewards = []
        self.episode_lengths = []
        self.timestep_marks = []

        self.episode_start_dist = []
        self.episode_end_dist = []
        self.dist_timestep_marks = []

        self.current_episode_dist = []
        self.last_episode_dist = []

        self.last_snapshot = 0

    def _dist(self, o):
        return float(np.linalg.norm(o[self.t0:self.t1] - o[self.g0:self.g1]))

    def _classify(self, truncated, term_reward):
        if truncated:
            return "FAIL (timeout)"
        if term_reward >= 15.0:
            return "SUCCESS"
        if term_reward <= -40.0:
            return "FAIL (collision)"
        if term_reward <= -5.0:
            return "FAIL (fell)"
        return "FAIL (ended)"

    def _on_step(self) -> bool:
        new_obs = self.locals.get("new_obs")
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards")
        total = int(getattr(self.model, "_total_timesteps", 0) or 0)

        for i, info in enumerate(infos):
            if "episode" in info:
                ep_r = float(info["episode"]["r"])
                ep_l = int(info["episode"]["l"])
                self.episode_rewards.append(ep_r)
                self.episode_lengths.append(ep_l)
                self.timestep_marks.append(self.num_timesteps)

                terminal_obs = info.get("terminal_observation")
                if terminal_obs is not None:
                    self.current_episode_dist.append(self._dist(terminal_obs))

                if self.current_episode_dist:
                    self.episode_start_dist.append(self.current_episode_dist[0])
                    self.episode_end_dist.append(self.current_episode_dist[-1])
                    self.dist_timestep_marks.append(self.num_timesteps)
                    self.last_episode_dist = self.current_episode_dist

                term_reward = float(rewards[i]) if rewards is not None else ep_r
                truncated = bool(info.get("TimeLimit.truncated", False))
                outcome = self._classify(truncated, term_reward)
                left = max(total - self.num_timesteps, 0)
                recent = self.episode_rewards[-100:]
                mean_r = sum(recent) / max(1, len(recent))
                print(f"[ep {len(self.episode_rewards)}] {outcome}  |  reward: {ep_r:+8.1f}  |  "
                      f"mean{len(recent)}: {mean_r:+7.1f}  |  steps: {ep_l}  |  left: {left:,}", flush=True)

                self.current_episode_dist = [self._dist(new_obs[i])]
            else:
                self.current_episode_dist.append(self._dist(new_obs[i]))

        if self.num_timesteps - self.last_snapshot >= self.snapshot_freq:
            self.last_snapshot = self.num_timesteps
            self._render_snapshot(tag=f"step_{self.num_timesteps:08d}")
            self._render_snapshot(tag="latest")
        return True

    def render_final(self):
        self._render_snapshot(tag="final")

    def _rolling(self, values, window_cap=100):
        values = np.asarray(values, dtype=np.float64)
        if len(values) < 10:
            return None, None
        window = min(window_cap, len(values) // 5 + 1)
        avg = np.convolve(values, np.ones(window) / window, mode="valid")
        offset = len(values) - len(avg)
        return avg, offset

    def _render_snapshot(self, tag: str):
        if len(self.episode_rewards) < 2:
            return

        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle(f"ScoutsanV1 (whole-body confondu, warm-started from 6-DOF arm) PPO "
                     f"— {self.num_timesteps:,} timesteps", fontsize=14)

        ax = axes[0, 0]
        ax.plot(self.timestep_marks, self.episode_rewards, alpha=0.3, color="steelblue", label="raw")
        avg, offset = self._rolling(self.episode_rewards)
        if avg is not None:
            ax.plot(self.timestep_marks[offset:], avg, linewidth=2, color="darkblue", label="avg")
        ax.set_xlabel("Timesteps"); ax.set_ylabel("Episode reward")
        ax.set_title("Episode reward over time"); ax.grid(True, alpha=0.3); ax.legend()

        ax = axes[0, 1]
        ax.plot(self.timestep_marks, self.episode_lengths, alpha=0.3, color="orange", label="raw")
        avg, offset = self._rolling(self.episode_lengths)
        if avg is not None:
            ax.plot(self.timestep_marks[offset:], avg, linewidth=2, color="darkorange", label="avg")
        ax.set_xlabel("Timesteps"); ax.set_ylabel("Episode length (steps)")
        ax.set_title("Episode length over time"); ax.grid(True, alpha=0.3); ax.legend()

        ax = axes[0, 2]
        recent = self.episode_rewards[-200:] if len(self.episode_rewards) >= 200 else self.episode_rewards
        ax.hist(recent, bins=30, color="purple", alpha=0.7, edgecolor="black")
        ax.set_xlabel("Episode reward"); ax.set_ylabel("Count")
        ax.set_title(f"Reward distribution (last {len(recent)} episodes)"); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        if len(self.episode_start_dist) >= 2:
            ax.plot(self.dist_timestep_marks, self.episode_start_dist, alpha=0.25, color="firebrick")
            ax.plot(self.dist_timestep_marks, self.episode_end_dist, alpha=0.25, color="seagreen")
            avg_s, off_s = self._rolling(self.episode_start_dist)
            avg_e, off_e = self._rolling(self.episode_end_dist)
            if avg_s is not None:
                ax.plot(self.dist_timestep_marks[off_s:], avg_s, linewidth=2, color="firebrick", label="start dist (avg)")
            if avg_e is not None:
                ax.plot(self.dist_timestep_marks[off_e:], avg_e, linewidth=2, color="seagreen", label="end dist (avg)")
            ax.axhline(self.success_radius, color="black", linestyle="--", linewidth=1, label="success radius")
            ax.legend()
        ax.set_xlabel("Timesteps"); ax.set_ylabel("Distance (m)")
        ax.set_title("Gripper-target distance: episode start vs end"); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if len(self.last_episode_dist) >= 2:
            ax.plot(self.last_episode_dist, color="seagreen")
            ax.axhline(self.success_radius, color="black", linestyle="--", linewidth=1, label="success radius")
            ax.legend()
        ax.set_xlabel("Step within episode"); ax.set_ylabel("Distance (m)")
        ax.set_title("Latest episode: does distance shrink?"); ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        if len(self.episode_end_dist) >= 10:
            successes = (np.array(self.episode_end_dist) < self.success_radius).astype(float)
            avg, offset = self._rolling(successes)
            if avg is not None:
                ax.plot(self.dist_timestep_marks[offset:], avg, linewidth=2, color="green")
                ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Timesteps"); ax.set_ylabel("Success rate")
        ax.set_title(f"Success rate (end distance < {self.success_radius} m)"); ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(self.save_dir, f"snapshot_{tag}.png")
        plt.savefig(out, dpi=120)
        plt.close(fig)
