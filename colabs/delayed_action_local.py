#!/usr/bin/env python3
"""
Justin's overnight run for merge-v0 PPO agent using delayed-action IS fuzzing.
Each worker process loads its own env + model and runs a batch of rollouts.

Usage:
    python colabs/delayed_action_local.py                          # defaults
    python colabs/delayed_action_local.py --num_workers 8 --num_rollouts 500
    python colabs/delayed_action_local.py --num_workers 1          # single-process (safe for testing)
    nohup python colabs/delayed_action_local.py --num_workers 12 --num_rollouts 20000 --output colabs/fuzzing_results_20k.json > colabs/fuzzing.log 2>&1 &
"""

import argparse
import json
import os
import time
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import sys

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "../merge/merge_ppo/PPO_1/final_model")
DEFAULT_DELAY_PROBS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5]
NOMINAL_DELAY_PROB  = 0.01
DEFAULT_NUM_ROLLOUTS = 20000
DEFAULT_NUM_WORKERS  = 10
DEFAULT_OUTPUT       = os.path.join(SCRIPT_DIR, "fuzzing_results.json")


# ── Worker (runs in a subprocess) ───────────────────────────────────────────
def _worker(args):
    """
    Load env + model fresh in this subprocess, run a batch of rollouts,
    return serialisable results dict.
    """
    delay_prob, n_rollouts, model_path, nominal_delay_prob, seed = args

    # imports inside worker so multiprocessing spawn works cleanly
    import sys, os, warnings
    import numpy as np

    # Suppress all worker stdout/stderr so tqdm in main process isn't broken
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

    is_weights             = []
    failure_logp_nominals  = []
    failure_logp_proposals = []
    mlf_logp               = -np.inf

    for _ in range(n_rollouts):
        failed, logp_nom, logp_prop = _run_rollout(env, model, delay_prob, nominal_delay_prob)

        w = float(np.exp(logp_nom - logp_prop)) if failed else 0.0
        is_weights.append(w)

        if failed:
            failure_logp_nominals.append(float(logp_nom))
            failure_logp_proposals.append(float(logp_prop))
            if logp_nom > mlf_logp:
                mlf_logp = logp_nom

    env.close()

    return {
        "delay_prob":             delay_prob,
        "is_weights":             is_weights,
        "failure_logp_nominals":  failure_logp_nominals,
        "failure_logp_proposals": failure_logp_proposals,
        "mlf_logp":               float(mlf_logp) if mlf_logp != -np.inf else None,
        "num_rollouts":           n_rollouts,
        "num_failures":           len(failure_logp_nominals),
    }


def _run_rollout(env, model, delay_prob, nominal_delay_prob):
    """Single rollout; returns (failed, logp_nominal, logp_proposal)."""
    import numpy as np

    obs, _ = env.reset()
    done = truncated = False
    logp_nominal  = 0.0
    logp_proposal = 0.0
    prev_action   = None

    while not (done or truncated):
        intended_action, _ = model.predict(obs, deterministic=True)

        if prev_action is not None:
            delayed = np.random.rand() < delay_prob
            action  = prev_action if delayed else intended_action

            if delayed:
                logp_proposal += np.log(delay_prob)
                logp_nominal  += np.log(nominal_delay_prob)
            else:
                logp_proposal += np.log(1.0 - delay_prob)
                logp_nominal  += np.log(1.0 - nominal_delay_prob)
        else:
            action = intended_action  # first step: no delay, no log-prob contribution

        prev_action = action
        obs, _, done, truncated, _ = env.step(action)

    vehicle = env.unwrapped.vehicle
    failed  = bool(getattr(vehicle, "crashed", False))
    return failed, logp_nominal, logp_proposal


# ── Merge worker results for one delay_prob ──────────────────────────────────
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

    p_fail_is = float(np.mean(is_weights))

    return {
        "delay_prob":             batch_results[0]["delay_prob"],
        "p_fail_is":              p_fail_is,
        "mlf_logp":               mlf_logp,
        "num_rollouts":           total_rollouts,
        "num_failures":           total_failures,
        "is_weights":             is_weights,
        "failure_logp_nominals":  failure_logp_nominals,
        "failure_logp_proposals": failure_logp_proposals,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Overnight IS fuzzing sweep for merge-v0 PPO")
    parser.add_argument("--num_workers",  type=int,   default=DEFAULT_NUM_WORKERS,
                        help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--num_rollouts", type=int,   default=DEFAULT_NUM_ROLLOUTS,
                        help="Rollouts per delay_prob (default: 500)")
    parser.add_argument("--model_path",   type=str,   default=DEFAULT_MODEL_PATH,
                        help="Path to PPO model zip")
    parser.add_argument("--output",       type=str,   default=DEFAULT_OUTPUT,
                        help="Output JSON file path")
    parser.add_argument("--delay_probs",  type=float, nargs="+",
                        default=DEFAULT_DELAY_PROBS,
                        help="Delay probabilities to sweep")
    args = parser.parse_args()

    print(f"Fuzzing sweep config:")
    print(f"  delay_probs  : {args.delay_probs}")
    print(f"  num_rollouts : {args.num_rollouts} per delay_prob")
    print(f"  num_workers  : {args.num_workers}")
    print(f"  model_path   : {args.model_path}")
    print(f"  output       : {args.output}")
    print()

    # Split each delay_prob's rollouts across workers as equal chunks
    tasks = []
    for dp in args.delay_probs:
        chunk_size  = args.num_rollouts // args.num_workers
        remainder   = args.num_rollouts % args.num_workers
        for w in range(args.num_workers):
            n = chunk_size + (1 if w < remainder else 0)
            seed = int(dp * 1e6) + w  # reproducible but varied per (delay_prob, worker)
            tasks.append((dp, n, args.model_path, NOMINAL_DELAY_PROB, seed))

    t0 = time.time()
    all_results = {}  # delay_prob -> merged result

    total_rollouts_all = args.num_rollouts * len(args.delay_probs)
    batch_by_dp = {}

    bar = tqdm(total=total_rollouts_all, unit="rollout", desc="Fuzzing",
               dynamic_ncols=True, miniters=1, file=sys.stdout)

    # Load existing results if output file already exists (resume support)
    if os.path.exists(args.output):
        with open(args.output) as f:
            all_results = json.load(f)
        print(f"Resuming — loaded {len(all_results)} existing delay_prob(s) from {args.output}")
    else:
        all_results = {}

    chunks_expected = {dp: args.num_workers for dp in args.delay_probs}

    def _save_and_report(dp, merged):
        all_results[str(dp)] = merged
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[saved] delay_prob={dp:.3f}  failures={merged['num_failures']}/{merged['num_rollouts']}  p_fail_is={merged['p_fail_is']:.5f}  mlf_logp={merged['mlf_logp']}")

    if args.num_workers == 1:
        for task in tasks:
            r = _worker(task)
            dp = r["delay_prob"]
            batch_by_dp.setdefault(dp, []).append(r)
            bar.update(r["num_rollouts"])
            bar.set_postfix(delay=f"{dp:.3f}", fails=r["num_failures"])
            if len(batch_by_dp[dp]) == chunks_expected[dp]:
                _save_and_report(dp, _merge(batch_by_dp[dp]))
    else:
        with Pool(processes=args.num_workers) as pool:
            for r in pool.imap_unordered(_worker, tasks):
                dp = r["delay_prob"]
                batch_by_dp.setdefault(dp, []).append(r)
                bar.update(r["num_rollouts"])
                bar.set_postfix(delay=f"{dp:.3f}", fails=r["num_failures"])
                if len(batch_by_dp[dp]) == chunks_expected[dp]:
                    _save_and_report(dp, _merge(batch_by_dp[dp]))

    bar.close()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()


# nohup python colabs/delayed_action_local.py --num_workers 2 --num_rollouts 100000 --delay_probs 0.01 --output colabs/delayed_action_100k_nominal.json > colabs/delayed_action.log 2>&1 &

