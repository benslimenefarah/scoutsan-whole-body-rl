#!/usr/bin/env python

import argparse, glob, os, re, runpy, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("RL_SCOUTSAN_ROOT") or os.path.abspath(os.path.join(HERE, ".."))

S9_TRAINER    = "scoutsan_Baseline/training/train_scoutsan_s9.py"

RUNS = {
    "s9_np": dict(
        desc      = "whole body, direct velocity, ONE fixed target (starting point of the T-arc run)",
        agent     = "Assets/scoutsan_Baseline/scripts/ScoutsanV1_1b.cs",
        scene     = "Assets/scoutsan_Baseline/scenes/s9.unity",
        robots    = 8,
        build     = "s9_np",
        model_dir = "scoutsan_Baseline/models/model_scoutsan_s9_np",
        prefix    = "ppo_scoutsan_s9_np",
        tb_dir    = "scoutsan_Baseline/tb/tb_scoutsan_s9_np",
        trainer   = S9_TRAINER,
        worker    = 14,
        timesteps = 10_000_000,
        gpu       = "0",
        warm_from = None,
        scene_note= "randomize* OFF, target fixed at targetSpawnOffset (0, 0.25, 2)",
    ),
    "s9_tarc": dict(
        desc      = "whole body, direct velocity, T-arc target curriculum (the end-to-end result)",
        agent     = "Assets/scoutsan_Baseline/scripts/ScoutsanV1_1b.cs",
        scene     = "Assets/scoutsan_Baseline/scenes/s9.unity",
        robots    = 8,
        build     = "s9_tarc",
        model_dir = "scoutsan_Baseline/models/model_scoutsan_s9_tarc",
        prefix    = "ppo_scoutsan_s9_tarc",
        tb_dir    = "scoutsan_Baseline/tb/tb_scoutsan_s9_tarc",
        trainer   = S9_TRAINER,
        worker    = 20,
        timesteps = 4_000_000,
        gpu       = "0",
        warm_from = ("scoutsan_Baseline/models/model_scoutsan_s9_np", "ppo_scoutsan_s9_np"),
        scene_note= "useTArcCurriculum=1; tArc distanceFullSteps 500000, arcFullSteps 1000000, "
                    "minRadius 0.75, lineRadius 2, arcTotalDeg 40, height 0.25",
    ),
}

STEPS_RE = re.compile(r"_(\d+)_steps\.zip$")

def die(run, msg):
    sys.exit(f"\n[{run}] REFUSING TO START\n{msg}\n")

def latest_checkpoint(model_dir, prefix):
    cands = glob.glob(os.path.join(model_dir, "checkpoints", f"{prefix}_*_steps.zip"))
    if cands:
        return max(cands, key=lambda p: int(STEPS_RE.search(p).group(1)) if STEPS_RE.search(p) else -1)
    interrupted = os.path.join(model_dir, f"{prefix}_interrupted.zip")
    if os.path.exists(interrupted):
        return interrupted

    final = os.path.join(model_dir, f"{prefix}.zip")
    return final if os.path.exists(final) else None

def check_build(run, cfg, build_dir, skip_stale):
    exe = sorted(glob.glob(os.path.join(build_dir, "*.x86_64")))
    if not exe:
        die(run, f"  No headless build in {build_dir}\n\n"
                 f"  Build it from Unity:\n"
                 f"    1. open   {cfg['scene']}\n"
                 f"    2. duplicate the robot to {cfg['robots']}, then verify:\n"
                 f"         grep -c 'm_BehaviorName:' {cfg['scene']}   # must print {cfg['robots']}\n"
                 f"    3. check the Inspector: {cfg['scene_note']}\n"
                 f"    4. Build Settings: this scene only, Linux x86_64, headless -> {build_dir}")

    raw = subprocess.run(["pgrep", "-af", r"\.x86_64"], capture_output=True, text=True).stdout.splitlines()
    busy = [l for l in raw if len(l.split(maxsplit=2)) > 1 and l.split(maxsplit=2)[1].endswith(".x86_64")]
    if busy:
        die(run, "  Another headless build is already running, and it holds the mlagents port:\n    "
                 + "\n    ".join(busy)
                 + "\n\n  Stop it first (kill -INT <pid>), or use --worker to move this run to a free port.")

    dll = os.path.join(build_dir, os.path.basename(exe[0])[:-7] + "_Data", "Managed", "Assembly-CSharp.dll")
    agent = os.path.join(REPO, cfg["agent"])
    if not skip_stale and os.path.exists(dll) and os.path.exists(agent):
        if os.path.getmtime(dll) < os.path.getmtime(agent):
            die(run, f"  STALE BUILD. Unity bakes the C# into the build, so this one does not contain\n"
                     f"  your current agent:\n"
                     f"    build  {dll}\n           {os.path.getmtime(dll):.0f}\n"
                     f"    agent  {cfg['agent']}\n           {os.path.getmtime(agent):.0f}\n\n"
                     f"  Rebuild, or pass --skip-build-check if you are certain it does not matter.")
    return exe[0]

def resolve_warmstart(run, cfg, model_dir, explicit):
    if explicit:
        if not os.path.exists(explicit):
            die(run, f"  --warmstart file not found: {explicit}")
        return explicit, f"explicit: {os.path.basename(explicit)}"
    own = latest_checkpoint(model_dir, cfg["prefix"])
    if own:
        return own, f"resuming this run: {os.path.basename(own)}"
    if cfg["warm_from"]:
        d, p = cfg["warm_from"]
        donor = latest_checkpoint(os.path.join(REPO, d), p)
        if not donor:
            die(run, f"  Nothing to warm start from. Expected a checkpoint of '{p}' in\n"
                     f"    {os.path.join(REPO, d)}\n"
                     f"  Pass --warmstart <file.zip>, or train that run first.")
        return donor, f"warm start from {p}: {os.path.basename(donor)}"
    return "", "from scratch (no donor configured)"

def main():
    ap = argparse.ArgumentParser(description="One launcher for every training run.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Stop a run with  kill -INT <pid>  so it writes _interrupted.zip.")
    ap.add_argument("run", nargs="?", help="which run (see --list)")
    ap.add_argument("--list", action="store_true", help="show the runs and exit")
    ap.add_argument("--dry-run", action="store_true", help="run every check and print the plan, launch nothing")
    ap.add_argument("--timesteps", type=int, help="override the step budget")
    ap.add_argument("--gpu", help="override CUDA_VISIBLE_DEVICES")
    ap.add_argument("--time-scale", type=float, help="Unity time scale (default 10 for headless)")
    ap.add_argument("--worker", type=int, help="worker id; the mlagents port is 5005 + worker")
    ap.add_argument("--warmstart", help="explicit .zip to start from")
    ap.add_argument("--force", action="store_true", help="proceed even though a final .zip exists")
    ap.add_argument("--skip-build-check", action="store_true", help="do not compare build and source mtimes")
    a = ap.parse_args()

    if a.list or not a.run:
        print("\nRuns:\n")
        for k, c in RUNS.items():
            print(f"  {k:10} {c['desc']}")
            print(f"  {'':10} scene {os.path.basename(c['scene'])} ({c['robots']} robots), "
                  f"build {c['build']}/, port {5005 + c['worker']}, {c['timesteps']:,} steps\n")
        print("  python training/train.py <run> --dry-run     to check without launching\n")
        return

    if a.run not in RUNS:
        sys.exit(f"unknown run '{a.run}'. Use --list.")
    cfg = RUNS[a.run]

    model_dir = os.path.join(REPO, cfg["model_dir"])
    build_dir = os.path.join(REPO, cfg["build"])
    worker    = a.worker if a.worker is not None else cfg["worker"]

    final_zip = os.path.join(model_dir, cfg["prefix"] + ".zip")
    if os.path.exists(final_zip) and not a.force:
        die(a.run, f"  A finished model already exists:\n    {final_zip}\n\n"
                   f"  Training restarts checkpoint numbering at 0 and would overwrite it.\n"
                   f"  Rename {cfg['model_dir']} to keep it, or pass --force.")

    exe = check_build(a.run, cfg, build_dir, a.skip_build_check)
    warm, warm_note = resolve_warmstart(a.run, cfg, model_dir, a.warmstart)

    env = {
        "ARM_BUILD":        exe,
        "ARM_MODEL_DIR":    model_dir,
        "ARM_MODEL_PREFIX": cfg["prefix"],
        "ARM_TB_DIR":       os.path.join(REPO, cfg["tb_dir"]),
        "ARM_WARMSTART":    warm,
        "ARM_TRANSPLANT":   "0",
        "ARM_WORKER_ID":    str(worker),
        "ARM_TIMESTEPS":    str(a.timesteps or cfg["timesteps"]),
        "ARM_TIME_SCALE":   str(a.time_scale if a.time_scale is not None else 10),
        "CUDA_VISIBLE_DEVICES": a.gpu or cfg["gpu"],
    }

    print(f"\n  run        {a.run}  --  {cfg['desc']}")
    print(f"  agent      {cfg['agent']}")
    print(f"  scene      {cfg['scene']}  ({cfg['robots']} robots)")
    print(f"  build      {exe}")
    print(f"  model      {cfg['model_dir']}")
    print(f"  tensorboard{'':1}{cfg['tb_dir']}")
    print(f"  warm start {warm_note}")
    print(f"  budget     {int(env['ARM_TIMESTEPS']):,} steps, time scale {env['ARM_TIME_SCALE']}")
    print(f"  port       {5005 + worker}   GPU {env['CUDA_VISIBLE_DEVICES']}")
    print(f"  trainer    {cfg['trainer']}")
    print(f"\n  CHECK IN THE SCENE: {cfg['scene_note']}\n")

    if a.dry_run:
        print("  --dry-run: every check passed, nothing launched.\n")
        return

    os.environ.update(env)
    os.chdir(os.path.dirname(os.path.join(REPO, cfg["trainer"])))
    runpy.run_path(os.path.join(REPO, cfg["trainer"]), run_name="__main__")

if __name__ == "__main__":
    main()
