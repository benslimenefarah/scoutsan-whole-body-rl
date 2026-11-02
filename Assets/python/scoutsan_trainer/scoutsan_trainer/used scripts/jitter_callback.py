

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class JitterLogCallback(BaseCallback):
    def __init__(self, out_csv, w_r=0.05, norm=32.0, dead=0.05, log_every=2048,
                 target_frac=0.15, verbose=0):
        super().__init__(verbose)
        self.out_csv = out_csv
        self.w_r = float(w_r); self.norm = float(norm); self.dead = float(dead)
        self.log_every = int(max(log_every, 1)); self.target_frac = float(target_frac)
        self._prev = None; self._have = None
        self._arm = 0.0; self._tot = 0.0; self._rev = 0; self._rev_steps = 0
        self._pos = 0.0; self._n = 0
        self._f = None

    def _on_training_start(self):
        d = os.path.dirname(self.out_csv)
        if d:
            os.makedirs(d, exist_ok=True)
        self._f = open(self.out_csv, "w")
        self._f.write("timesteps,samples,mean_arm_jerk,arm_reversal_rate_pct,"
                      "mean_r8_bite,r8_pct_of_posreward,suggested_wr,verdict\n")
        self._f.flush()
        if self.verbose:
            print(f"[jitter-log] arm-jitter trend -> {self.out_csv}  (w_r={self.w_r}, one row / rollout)")

    def _on_step(self):
        acts = self.locals.get("clipped_actions", None)
        if acts is None:
            acts = self.locals.get("actions", None)
        if acts is None:
            return True
        a = np.asarray(acts, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        n = a.shape[0]
        if self._prev is None or self._prev.shape != a.shape:
            self._prev = np.zeros_like(a); self._have = np.zeros(n, dtype=bool)

        m = self._have
        if m.any():
            da = a - self._prev
            arm = np.sum(da[:, 0:6] ** 2, axis=1)
            tot = np.sum(da ** 2, axis=1)
            rev = np.sum((a[:, 0:6] * self._prev[:, 0:6] < 0)
                         & (np.abs(a[:, 0:6]) > self.dead)
                         & (np.abs(self._prev[:, 0:6]) > self.dead), axis=1)
            self._arm += float(np.sum(arm[m])); self._tot += float(np.sum(tot[m]))
            self._rev += int(np.sum(rev[m])); self._rev_steps += int(np.sum(rev[m] > 0))
            self._n += int(m.sum())
            rews = self.locals.get("rewards", None)
            if rews is not None:
                r = np.asarray(rews, dtype=np.float32)
                self._pos += float(np.sum(np.clip(r[m], 0.0, None)))

        self._prev = a.copy(); self._have[:] = True
        dones = self.locals.get("dones", None)
        if dones is not None:
            self._have[np.asarray(dones, dtype=bool)] = False

        if self.n_calls % self.log_every == 0 and self._n > 0:
            self._write()
        return True

    def _write(self):
        mean_arm = self._arm / self._n
        mean_tot = self._tot / self._n
        rev_rate = 100.0 * self._rev_steps / self._n
        mean_r8  = self.w_r * mean_tot / self.norm
        mean_pos = (self._pos / self._n) if self._n else 0.0
        pct = (100.0 * mean_r8 / mean_pos) if mean_pos > 1e-9 else float("nan")
        if np.isfinite(pct) and pct > 1e-9:
            sugg = float(np.clip(self.w_r * (self.target_frac * 100.0) / pct, self.w_r, 1.0))
        else:
            sugg = self.w_r
        verdict = "smooth" if rev_rate < 10 else ("mild" if rev_rate < 30 else "strong")
        self._f.write(f"{self.num_timesteps},{self._n},{mean_arm:.5f},{rev_rate:.1f},"
                      f"{mean_r8:.5f},{pct:.1f},{sugg:.2f},{verdict}\n")
        self._f.flush()
        self._arm = self._tot = self._pos = 0.0; self._rev = self._rev_steps = 0; self._n = 0

    def _on_training_end(self):
        if self._n > 0 and self._f is not None:
            self._write()
        if self._f is not None:
            self._f.close(); self._f = None
