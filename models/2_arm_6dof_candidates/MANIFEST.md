# 6-DOF arm candidates — rescued 2026-10-01

Rescued from `RL_Scoutsan/python/old models/` before that repository was deleted. These are the
checkpoints behind the arm curriculum; the observation/action sizes below were read by loading each
file with `PPO.load`, so they are measured, not assumed.

| directory | obs / act | stage |
|---|---|---|
| `model_arm_L_2dof` | 6 / 2 | curriculum, 2 joints |
| `model_arm_L_3dof` | 6 / 3 | curriculum, 3 joints |
| `model_arm_L_4dof` | 6 / 4 | curriculum, 4 joints |
| `model_arm_L_6dof` | 7 / 6 | 6 joints, reaching only (position) |
| `model_arm_L_6dof_confondu` | 14 / 6 | first full pose: two frames, position + orientation |
| `model_arm_L_6dof_confondu_orient` | 14 / 6 | orientation gate added |
| `model_arm_L_6dof_confondu_orient_gpu` | 14 / 6 | same, GPU run |
| `model_arm_L_6dof_confondu_orient_multi` | 14 / 6 | multi-agent; reached ~96% at gate 0.05/0.08 |
| `model_arm_L_6dof_confondu_resume` | 14 / 6 | gate tightened to 0.03/0.05 |
| `model_arm_L_6dof_confondu_traj_smooth` | 14 / 6 | r7 = -w*sum(a^2), w=0.01 — did NOT smooth |
| `model_arm_L_6dof_confondu_traj_smooth_w0p1` | 14 / 6 | same term at w=0.1 — still did not smooth |
| `model_16arm_ratepenalty` | 14 / 6 | rate penalty sum((a_t - a_{t-1})^2) instead of sum(a^2) |
| `model_16arm_actionscale` | 14 / 6 | adaptive action scale v1 (0.10/0.15/0.1) — still shook |
| `model_16arm_actionscale_v2` | 14 / 6 | v2 (0.20/0.30/0.04) **plus the r3/r6 deadband fix** |
| `model_16arm_actionscale_v2_predeadband` | 14 / 6 | the archived before-state of that deadband A/B |
| `model_1arm_editor` | 14 / 8 | mislabelled: 8 actions, so a whole-body run, not an arm run |

## Which one is the 6-DOF arm deliverable

`model_16arm_actionscale_v2/ppo_16arm_actionscale_v2_interrupted.zip` — 14 obs / 6 act. It is the end
of the lineage: adaptive action scale v2 with the deadband fix applied in place, which is why the
pre-deadband state sits beside it as its own archived baseline for the A/B.

Pair it with `Assets/scripts/Arm scripts/ArmAgentL.cs`. It expects an arm-only scene with Behavior
Parameters at 14 obs / 6 continuous actions — not one of the whole-body scenes, which are 14/8.
The arm gate for that series was 0.03 m / 0.06 rad.

Still missing for a finished deliverable: an evaluation run, figures and a model card. The videos are
already in `deliverables/arm_6dof_model/`.
