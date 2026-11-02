# deliverables

One folder per model, each with its videos, evaluation data, figures and model card.

| folder | model | result |
|---|---|---|
| `base_model/` | base driver (4 obs / 2 act) | 6/6 reach the 0.6 m gate |
| `arm_6dof_model/` | 6-DOF arm (14 / 6) | videos only |
| `s9_np_model/fixed target/` | whole body (14 / 8), fixed target | 5/5 aligned, min `d` 26 mm, min `φ` 1.35° |
| `s9_tarc_model/` | whole body (14 / 8), arc curriculum, end-to-end | 5/5 within 2 cm / 1.15°, 8.3 s to the gate on average |
| `wb1am_model/lambda=1/` | hybrid model, λ = 1 (controller not included, see the main README) | videos only |

## Contents of a model folder

- **Videos:** `*_FollowCam.mp4`, `*_sidecam.mp4`, `*_upcam.mp4` (three cameras, RGB axes on the gripper
  and the target), or `base_model_demo.mp4`.
- **Evaluation data:** `*_eval.json` (summary per episode and target) and `*_eval_steps.csv` (per step).
- **`figures/`:** plots as PDF and 300 dpi PNG, including:
  - `policy_graph`: observation → MLP → action head, read from the checkpoint;
  - `reward_curve`: training reward vs steps;
  - `reward_graph`: the reward function as a diagram;
  - `trajectories`: the gripper path in the target frame.
- **`*_model_card.pdf`:** task, observations, actions, reward, gate, configuration and results.
- The scripts that regenerate them (`*_plots.py`, `*_notes.py`, `*_card.py`, `make_*_videos.sh`).

## `common/`

Task-level figures shared by the whole-body models:

- `target_spawn_area`: the arc curriculum's spawn area, with the three evaluation targets
  (on-axis, arc −20°, arc +20°, 2 m away, 0.25 m high);
- `frames_confondu`: the gripper and target frames, and the two errors of the gate,
  `d = ‖p_G − p_T‖` and `φ = 2·arccos|⟨q_G, q_T⟩|`.

## Regenerate

```bash
python deliverables/make_policy_graphs.py      # policy diagrams
python deliverables/make_reward_curves.py      # training reward curves (needs the TensorBoard logs)
python deliverables/make_reward_graphs.py      # reward-function diagrams
python deliverables/make_trajectory_graphs.py  # target-frame trajectories
python deliverables/make_common_figures.py     # figures in common/
```

`base_model` and `s9_np_model` log only `d` and `φ` per step, so they have no trajectory figure.
