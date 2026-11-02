

import torch
from stable_baselines3 import PPO

def _padded_copy(old_t, new_t, zero_grown):
    result = torch.zeros_like(new_t) if zero_grown else new_t.clone()
    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_t.shape, new_t.shape))
    result[slices] = old_t[slices]
    return result

def transfer_weights_padded(pretrained_path, new_model, verbose=True):
    old_model = PPO.load(pretrained_path, device=new_model.device)

    old_sd = old_model.policy.state_dict()
    new_sd = new_model.policy.state_dict()

    transferred, padded, fresh = [], [], []
    for key, new_tensor in new_sd.items():
        old_tensor = old_sd.get(key)
        if old_tensor is None:
            fresh.append((key, None, tuple(new_tensor.shape)))
            continue
        if old_tensor.shape == new_tensor.shape:
            new_sd[key] = old_tensor.clone()
            transferred.append(key)
        elif old_tensor.ndim == new_tensor.ndim and all(n >= o for o, n in zip(old_tensor.shape, new_tensor.shape)):
            zero_grown = "log_std" not in key
            new_sd[key] = _padded_copy(old_tensor, new_tensor, zero_grown)
            padded.append((key, tuple(old_tensor.shape), tuple(new_tensor.shape), zero_grown))
        else:
            fresh.append((key, tuple(old_tensor.shape), tuple(new_tensor.shape)))

    new_model.policy.load_state_dict(new_sd)

    if verbose:
        print(f"[transfer] loaded pretrained arm model: {pretrained_path}")
        print(f"[transfer] {len(transferred)} tensor(s) copied verbatim (exact shape match):")
        for k in transferred:
            print(f"             {k}")
        if padded:
            print(f"[transfer] {len(padded)} tensor(s) GROWN + padded (trained block copied, new region "
                  f"zeroed for weights/biases, kept fresh for log_std):")
            for k, os_, ns_, zg in padded:
                how = "new region ZEROED (function-preserving)" if zg else "new region kept fresh"
                print(f"             {k}: arm={os_} -> ScoutsanV1={ns_}  [{how}]")
        if fresh:
            print(f"[transfer] {len(fresh)} tensor(s) left FRESH (could not reconcile shapes):")
            for k, os_, ns_ in fresh:
                print(f"             {k}: arm={os_} -> ScoutsanV1={ns_}")
            print("[transfer] NOTE: a fresh tensor here means a shape change beyond a simple grow. "
                  "Check the ScoutsanV1 agent sends exactly 14 observations and 8 actions.")
        else:
            print("[transfer] all weights reconciled: body + value head + input layer verbatim, "
                  "action head padded 6->8 (function-preserving).")

    return transferred, padded, fresh

def transfer_arm_keep_zero_base(source_path, new_model, n_arm=6, n_total=8, verbose=True):
    old_model = PPO.load(source_path, device=new_model.device)
    old_sd = old_model.policy.state_dict()
    new_sd = new_model.policy.state_dict()
    fresh_log_std = new_sd.get("log_std")

    kept, base_zeroed, mismatched = [], [], []
    for key, new_tensor in new_sd.items():
        old_tensor = old_sd.get(key)
        if old_tensor is None or old_tensor.shape != new_tensor.shape:
            mismatched.append((key, None if old_tensor is None else tuple(old_tensor.shape), tuple(new_tensor.shape)))
            continue
        t = old_tensor.clone()
        if key == "action_net.weight":
            t[n_arm:n_total, :] = 0.0; base_zeroed.append(key)
        elif key == "action_net.bias":
            t[n_arm:n_total] = 0.0; base_zeroed.append(key)
        elif key == "log_std" and fresh_log_std is not None:
            t[n_arm:n_total] = fresh_log_std[n_arm:n_total].clone(); base_zeroed.append(key)
        else:
            kept.append(key)
        new_sd[key] = t

    new_model.policy.load_state_dict(new_sd)

    if verbose:
        print(f"[arm-keep] whole-body source (14/8): {source_path}")
        print(f"[arm-keep] {len(kept)} tensor(s) copied verbatim (shared body + value head + arm-relevant params).")
        print(f"[arm-keep] BASE zeroed/reset in: {base_zeroed}  -> arm rows [0:{n_arm}] kept, "
              f"base rows [{n_arm}:{n_total}] passive, base log_std fresh.")
        if mismatched:
            print(f"[arm-keep] WARNING {len(mismatched)} tensor(s) shape-mismatch (is the source really 14/8?): {mismatched}")
    return kept, base_zeroed, mismatched
