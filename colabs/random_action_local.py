#!/usr/bin/env python3
"""
Overnight run for merge-v0 PPO agent using random-action IS fuzzing.
Each worker process loads its own env + model and runs a batch of rollouts.

Usage:
    python colabs/random_action_local.py                           # defaults
    python colabs/random_action_local.py --num_workers 8 --num_rollouts 500
    python colabs/random_action_local.py --num_workers 1           # single-process (safe for testing)
    nohup python colabs/random_action_local.py --num_workers 12 --num_rollouts 100000 --output colabs/random_action_results_100k.json > colabs/random_action.log 2>&1 &
"""

import argparse
import json
import os
import time
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import sys

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "../merge/merge_ppo/PPO_1/final_model")
DEFAULT_RANDOM_PROBS = [0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0]  # Additional 1.0 to have fully random policy
NOMINAL_RANDOM_PROB  = 0.01
DEFAULT_NUM_ROLLOUTS = 100000
DEFAULT_NUM_WORKERS  = 12
DEFAULT_OUTPUT       = os.path.join(SCRIPT_DIR, "random_action_results.json")


# ── Worker (runs in a subprocess) ────────────────────────────────────────────
def _worker(args):
    random_prob, n_rollouts, model_path, nominal_random_prob, seed = args

    import sys, os, warnings
    import numpy as np

    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull
    warnings.filterwarnings("ignore")

    import gymnasium as gym
    import highway_env
    from stable_baselines3 import PPO

    np.random.seed(seed)
    gym.register_envs(highway_env)

    model = PPO.load(model_path, device="cpu")
    env   = gym.make("merge-v0", render_mode=None)

    n_actions = env.action_space.n

    is_weights             = []
    failure_logp_nominals  = []
    failure_logp_proposals = []
    mlf_logp               = -np.inf

    for _ in range(n_rollouts):
        failed, logp_nom, logp_prop = _run_rollout(
            env, model, random_prob, nominal_random_prob, n_actions
        )

        w = float(np.exp(logp_nom - logp_prop)) if failed else 0.0
        is_weights.append(w)

        if failed:
            failure_logp_nominals.append(float(logp_nom))
            failure_logp_proposals.append(float(logp_prop))
            if logp_nom > mlf_logp:
                mlf_logp = logp_nom

    env.close()

    return {
        "random_prob":            random_prob,
        "is_weights":             is_weights,
        "failure_logp_nominals":  failure_logp_nominals,
        "failure_logp_proposals": failure_logp_proposals,
        "mlf_logp":               float(mlf_logp) if mlf_logp != -np.inf else None,
        "num_rollouts":           n_rollouts,
        "num_failures":           len(failure_logp_nominals),
    }


def _run_rollout(env, model, random_prob, nominal_random_prob, n_actions):
    """Single rollout; returns (failed, logp_nominal, logp_proposal)."""
    import numpy as np

    obs, _ = env.reset()
    done = truncated = False
    logp_nominal  = 0.0
    logp_proposal = 0.0

    while not (done or truncated):
        intended_action, _ = model.predict(obs, deterministic=True)

        use_random = np.random.rand() < random_prob
        action = env.action_space.sample() if use_random else intended_action

        if use_random:
            logp_proposal += np.log(random_prob)         + np.log(1.0 / n_actions)
            logp_nominal  += np.log(nominal_random_prob) + np.log(1.0 / n_actions)
        else:
            logp_proposal += np.log(1.0 - random_prob)
            logp_nominal  += np.log(1.0 - nominal_random_prob)

        obs, _, done, truncated, _ = env.step(action)

    vehicle = env.unwrapped.vehicle
    failed  = bool(getattr(vehicle, "crashed", False))
    return failed, logp_nominal, logp_proposal


# ── Merge worker results for one random_prob ─────────────────────────────────
def _merge(batch_results):
    is_weights             = []
    failure_logp_nominals  = []
    failure_logp_proposals = []
    mlf_logp               = None
    total_rollouts         = 0
    total_failures         = 0

    for r in batch_results:
        is_weights.extend(r["is_weights"])
        failure_logp_nominals.extend(r["failure_logp_nominals"])
        failure_logp_proposals.extend(r["failure_logp_proposals"])
        total_rollouts += r["num_rollouts"]
        total_failures += r["num_failures"]
        if r["mlf_logp"] is not None:
            if mlf_logp is None or r["mlf_logp"] > mlf_logp:
                mlf_logp = r["mlf_logp"]

    return {
        "random_prob":            batch_results[0]["random_prob"],
        "p_fail_is":              float(np.mean(is_weights)),
        "mlf_logp":               mlf_logp,
        "num_rollouts":           total_rollouts,
        "num_failures":           total_failures,
        "is_weights":             is_weights,
        "failure_logp_nominals":  failure_logp_nominals,
        "failure_logp_proposals": failure_logp_proposals,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Overnight random-action IS fuzzing sweep for merge-v0 PPO")
    parser.add_argument("--num_workers",   type=int,   default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--num_rollouts",  type=int,   default=DEFAULT_NUM_ROLLOUTS)
    parser.add_argument("--model_path",    type=str,   default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output",        type=str,   default=DEFAULT_OUTPUT)
    parser.add_argument("--random_probs",  type=float, nargs="+", default=DEFAULT_RANDOM_PROBS)
    args = parser.parse_args()

    print(f"Random-action fuzzing sweep config:")
    print(f"  random_probs : {args.random_probs}")
    print(f"  num_rollouts : {args.num_rollouts} per random_prob")
    print(f"  num_workers  : {args.num_workers}")
    print(f"  model_path   : {args.model_path}")
    print(f"  output       : {args.output}")
    print()

    tasks = []
    for rp in args.random_probs:
        chunk_size = args.num_rollouts // args.num_workers
        remainder  = args.num_rollouts % args.num_workers
        for w in range(args.num_workers):
            n    = chunk_size + (1 if w < remainder else 0)
            seed = int(rp * 1e6) + w
            tasks.append((rp, n, args.model_path, NOMINAL_RANDOM_PROB, seed))

    t0 = time.time()
    batch_by_rp = {}

    if os.path.exists(args.output):
        with open(args.output) as f:
            all_results = json.load(f)
        print(f"Resuming — loaded {len(all_results)} existing random_prob(s) from {args.output}")
    else:
        all_results = {}

    chunks_expected = {rp: args.num_workers for rp in args.random_probs}

    total_rollouts_all = args.num_rollouts * len(args.random_probs)
    bar = tqdm(total=total_rollouts_all, unit="rollout", desc="Fuzzing",
               dynamic_ncols=True, miniters=1, file=sys.stdout)

    def _save_and_report(rp, merged):
        all_results[str(rp)] = merged
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[saved] random_prob={rp:.3f}  failures={merged['num_failures']}/{merged['num_rollouts']}  p_fail_is={merged['p_fail_is']:.5f}  mlf_logp={merged['mlf_logp']}")

    if args.num_workers == 1:
        for task in tasks:
            r = _worker(task)
            rp = r["random_prob"]
            batch_by_rp.setdefault(rp, []).append(r)
            bar.update(r["num_rollouts"])
            bar.set_postfix(rp=f"{rp:.3f}", fails=r["num_failures"])
            if len(batch_by_rp[rp]) == chunks_expected[rp]:
                _save_and_report(rp, _merge(batch_by_rp[rp]))
    else:
        with Pool(processes=args.num_workers) as pool:
            for r in pool.imap_unordered(_worker, tasks):
                rp = r["random_prob"]
                batch_by_rp.setdefault(rp, []).append(r)
                bar.update(r["num_rollouts"])
                bar.set_postfix(rp=f"{rp:.3f}", fails=r["num_failures"])
                if len(batch_by_rp[rp]) == chunks_expected[rp]:
                    _save_and_report(rp, _merge(batch_by_rp[rp]))

    bar.close()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

# nohup python colabs/random_action_local.py --num_workers 8 --random_probs 0.01 --num_rollouts 100000 --output colabs/random_action_100k_nominal.json > colabs/random_action.log 2>&1 &
