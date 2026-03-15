
"""
IC Fuzzing parallel runner  merge-v0 PPO, Beta(\alpha,\alpha) IS proposal.

Fuzzes IC variables for each of the 3 highway cars
  - lane     : Bernoulli(lane_bias_q)    [nominal q=0.5]
  - pos_offset: Beta(beta_alpha, beta_alpha) scaled to [-5, 5]  [nominal alpha=1.0 = uniform]
  - spd_offset: Beta(beta_alpha, beta_alpha) scaled to [-1, 1]  [nominal alpha=1.0 = uniform]


Sweep: Cartesian product of lane_bias_qs x beta_alphas
JSON format: "q={lane_bias_q:.2f}_a={beta_alpha:.2f}"

Usage:
    # quick test (single worker, small N):
    python colabs/ic_fuzzing_local.py --num_workers 1 --num_rollouts 200 --lane_bias_qs 0.5 0.9 --beta_alphas 0.5 1.0 --output colabs/ic_fuzzing_test.json
    # test
    python colabs/ic_fuzzing_local.py --num_workers 4 --num_rollouts 500 --lane_bias_qs 0.5 --beta_alphas 1.0 --output colabs/test_4w.json
    # full overnight run:
    python colabs/ic_fuzzing_local.py --num_workers 12 --num_rollouts 20000 --output colabs/ic_fuzzing_results.json
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from itertools import product
from multiprocessing import Pool
from tqdm import tqdm

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR           = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH   = os.path.join(SCRIPT_DIR, "../merge/merge_ppo/PPO_1/final_model")
DEFAULT_LANE_BIAS_QS = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_BETA_ALPHAS  = [0.3, 0.5, 0.9, 1.0]
NOMINAL_LANE_Q       = 0.5
NOMINAL_BETA_ALPHA   = 1.0
DEFAULT_NUM_ROLLOUTS = 20000
DEFAULT_NUM_WORKERS  = 10
DEFAULT_OUTPUT       = os.path.join(SCRIPT_DIR, "ic_fuzzing_results.json")


def _config_key(q: float, alpha: float) -> str:
    return f"q={q:.2f}_a={alpha:.2f}"


# ── Worker (spawned in subprocess) ────────────────────────────────────────────
def _worker(args):
    """Load env + model fresh in subprocess, run a batch, return serialisable result dict."""
    lane_bias_q, beta_alpha, n_rollouts, model_path, seed = args

    import os, sys, warnings
    import numpy as np
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull
    warnings.filterwarnings("ignore")

    import gymnasium as gym
    import highway_env
    from stable_baselines3 import PPO
    from highway_env.envs.merge_env import MergeEnv
    from highway_env import utils

    np.random.seed(seed)
    gym.register_envs(highway_env)

    # ── ICFuzzMergeEnv defined inline for clean subprocess pickling ──────────
    class _ICFuzzMergeEnv(MergeEnv):
        """MergeEnv with Beta(alpha,alpha) IS proposal for lane + pos + spd."""
        _BASE_POSITIONS = [90.0, 70.0, 5.0]
        _BASE_SPEEDS    = [29.0, 31.0, 31.5]

        def __init__(self, lane_bias_q: float = 0.5, beta_alpha: float = 1.0,
                     config=None, render_mode=None):
            self._lane_bias_q = lane_bias_q
            self._beta_alpha  = beta_alpha
            super().__init__(config=config, render_mode=render_mode)

        def _make_vehicles(self):
            road = self.road
            ego = self.action_type.vehicle_class(
                road, road.network.get_lane(("a", "b", 1)).position(30.0, 0.0), speed=30.0
            )
            road.vehicles.append(ego)
            other_type = utils.class_from_path(self.config["other_vehicles_type"])

            self.last_lanes = []
            self.last_u_pos = []   # normalized pos offset in [0,1]
            self.last_u_spd = []   # normalized spd offset in [0,1]

            for base_pos, base_spd in zip(self._BASE_POSITIONS, self._BASE_SPEEDS):
                # Lane: Bernoulli(lane_bias_q)
                lane_idx = 1 if self.np_random.random() < self._lane_bias_q else 0

                # Position offset: Beta(alpha,alpha) scaled to [-5, 5]
                u_pos = float(self.np_random.beta(self._beta_alpha, self._beta_alpha))
                pos_offset = u_pos * 10.0 - 5.0

                # Speed offset: Beta(alpha,alpha) scaled to [-1, 1]
                u_spd = float(self.np_random.beta(self._beta_alpha, self._beta_alpha))
                spd_offset = u_spd * 2.0 - 1.0

                self.last_lanes.append(lane_idx)
                self.last_u_pos.append(u_pos)
                self.last_u_spd.append(u_spd)

                lane     = road.network.get_lane(("a", "b", lane_idx))
                position = lane.position(base_pos + pos_offset, 0.0)
                road.vehicles.append(other_type(road, position, speed=base_spd + spd_offset))

            # Merging vehicle nominal has fixed IC (lane=0, pos=110, spd=20)
            merging_v = other_type(
                road, road.network.get_lane(("j", "k", 0)).position(110.0, 0.0), speed=20.0
            )
            merging_v.target_speed = 30.0
            road.vehicles.append(merging_v)
            self.vehicle = ego

    model = PPO.load(model_path, device="cpu")
    env   = _ICFuzzMergeEnv(lane_bias_q=lane_bias_q, beta_alpha=beta_alpha)

    is_weights             = []
    failure_logp_nominals  = []
    failure_logp_proposals = []
    failure_types          = []
    mlf_logp               = -np.inf

    for _ in range(n_rollouts):
        failed, failure_type, logp_nom, logp_prop = _run_rollout(
            env, model, lane_bias_q, beta_alpha
        )
        w = float(np.exp(logp_nom - logp_prop)) if failed else 0.0
        is_weights.append(w)

        if failed:
            failure_logp_nominals.append(float(logp_nom))
            failure_logp_proposals.append(float(logp_prop))
            failure_types.append(failure_type)
            if logp_nom > mlf_logp:
                mlf_logp = logp_nom

    env.close()

    return {
        "lane_bias_q":            lane_bias_q,
        "beta_alpha":             beta_alpha,
        "is_weights":             is_weights,
        "failure_logp_nominals":  failure_logp_nominals,
        "failure_logp_proposals": failure_logp_proposals,
        "failure_types":          failure_types,
        "mlf_logp":               float(mlf_logp) if mlf_logp != -np.inf else None,
        "num_rollouts":           n_rollouts,
        "num_failures":           len(failure_logp_nominals),
    }


def _run_rollout(env, model, lane_bias_q: float, beta_alpha: float):
    """Single rollout. Returns (failed, failure_type, logp_nominal, logp_proposal)."""
    from scipy.stats import beta as beta_dist
    import numpy as np

    obs, _ = env.reset()
    done = truncated = False

    logp_nominal  = 0.0
    logp_proposal = 0.0

    # Lane terms
    for lane_idx in env.last_lanes:
        if lane_idx == 1:
            logp_nominal  += np.log(NOMINAL_LANE_Q)
            logp_proposal += np.log(lane_bias_q) if lane_bias_q > 0.0 else -np.inf
        else:
            logp_nominal  += np.log(1.0 - NOMINAL_LANE_Q)
            logp_proposal += np.log(1.0 - lane_bias_q) if lane_bias_q < 1.0 else -np.inf

    if beta_alpha != NOMINAL_BETA_ALPHA:
        for u_pos, u_spd in zip(env.last_u_pos, env.last_u_spd):
            lp_pos = beta_dist.logpdf(u_pos, beta_alpha, beta_alpha)
            lp_spd = beta_dist.logpdf(u_spd, beta_alpha, beta_alpha)
            # proposal log-prob (nominal = 0 for all valid u)
            logp_proposal += lp_pos + lp_spd
            # nominal log-prob stays 0 (uniform)

    merger = env.unwrapped.road.vehicles[-1]

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)

    vehicle      = env.unwrapped.vehicle
    failed       = bool(getattr(vehicle, "crashed", False))
    failure_type = None
    if failed:
        failure_type = "merger" if getattr(merger, "crashed", False) else "highway"

    return failed, failure_type, logp_nominal, logp_proposal


# ── Merge worker results for one (lane_bias_q, beta_alpha) config ─────────────
def _merge(batch_results):
    is_weights             = []
    failure_logp_nominals  = []
    failure_logp_proposals = []
    failure_types          = []
    mlf_logp               = None
    total_rollouts         = 0
    total_failures         = 0

    for r in batch_results:
        is_weights.extend(r["is_weights"])
        failure_logp_nominals.extend(r["failure_logp_nominals"])
        failure_logp_proposals.extend(r["failure_logp_proposals"])
        failure_types.extend(r["failure_types"])
        total_rollouts += r["num_rollouts"]
        total_failures += r["num_failures"]
        if r["mlf_logp"] is not None:
            if mlf_logp is None or r["mlf_logp"] > mlf_logp:
                mlf_logp = r["mlf_logp"]

    p_fail_is = float(np.mean(is_weights))

    return {
        "lane_bias_q":            batch_results[0]["lane_bias_q"],
        "beta_alpha":             batch_results[0]["beta_alpha"],
        "p_fail_is":              p_fail_is,
        "mlf_logp":               mlf_logp,
        "num_rollouts":           total_rollouts,
        "num_failures":           total_failures,
        "is_weights":             is_weights,
        "failure_logp_nominals":  failure_logp_nominals,
        "failure_logp_proposals": failure_logp_proposals,
        "failure_types":          failure_types,
    }


def main():
    parser = argparse.ArgumentParser(
        description="IC IS fuzzing sweep for merge-v0 PPO (Beta proposal for all IC vars)"
    )
    parser.add_argument("--num_workers",  type=int,   default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--num_rollouts", type=int,   default=DEFAULT_NUM_ROLLOUTS,
                        help="Rollouts per (lane_bias_q, beta_alpha) config")
    parser.add_argument("--model_path",   type=str,   default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output",       type=str,   default=DEFAULT_OUTPUT)
    parser.add_argument("--lane_bias_qs", type=float, nargs="+",
                        default=DEFAULT_LANE_BIAS_QS,
                        help="P(lane=1) values to sweep; must be strictly in (0,1) for IS validity (default: 0.5 0.7 0.9)")
    parser.add_argument("--beta_alphas",  type=float, nargs="+",
                        default=DEFAULT_BETA_ALPHAS,
                        help="Beta(alpha,alpha) shape for pos/spd (default: 0.3 0.5 1.0)")
    args = parser.parse_args()

    configs = list(product(args.lane_bias_qs, args.beta_alphas))

    print("IC Fuzzing sweep config:")
    print(f"  lane_bias_qs : {args.lane_bias_qs}")
    print(f"  beta_alphas  : {args.beta_alphas}")
    print(f"  configs      : {len(configs)} total ({len(args.lane_bias_qs)}×{len(args.beta_alphas)})")
    print(f"  num_rollouts : {args.num_rollouts} per config")
    print(f"  num_workers  : {args.num_workers}")
    print(f"  model_path   : {args.model_path}")
    print(f"  output       : {args.output}")
    print()

    # Resume: load previously completed configs
    if os.path.exists(args.output):
        with open(args.output) as f:
            all_results = json.load(f)
        print(f"Resuming  loaded {len(all_results)} existing config(s) from {args.output}")
    else:
        all_results = {}

    # Build worker task list, skipping already-completed configs
    tasks           = []
    chunks_expected = {}
    configs_to_run  = []

    for q, alpha in configs:
        key = _config_key(q, alpha)
        if key in all_results:
            print(f"  Skipping {key} (already complete)")
            continue
        configs_to_run.append((q, alpha))
        chunk_size = args.num_rollouts // args.num_workers
        remainder  = args.num_rollouts % args.num_workers
        chunks_expected[key] = args.num_workers
        for w in range(args.num_workers):
            n    = chunk_size + (1 if w < remainder else 0)
            seed = int(q * 1e6 + alpha * 1e3) + w
            tasks.append((q, alpha, n, args.model_path, seed))

    if not tasks:
        print("All configs already complete. Nothing to do.")
        return

    total_rollouts_all = args.num_rollouts * len(configs_to_run)
    batch_by_key       = {}

    def _save_and_report(key, merged):
        all_results[key] = merged
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[saved] {key}  "
              f"failures={merged['num_failures']}/{merged['num_rollouts']}  "
              f"p_fail_is={merged['p_fail_is']:.5f}  "
              f"mlf_logp={merged['mlf_logp']}")

    t0  = time.time()
    bar = tqdm(total=total_rollouts_all, unit="rollout", desc="IC Fuzzing",
               dynamic_ncols=True, miniters=1, file=sys.stdout)

    if args.num_workers == 1:
        for task in tasks:
            r   = _worker(task)
            key = _config_key(r["lane_bias_q"], r["beta_alpha"])
            batch_by_key.setdefault(key, []).append(r)
            bar.update(r["num_rollouts"])
            bar.set_postfix(key=key, fails=r["num_failures"])
            if len(batch_by_key[key]) == chunks_expected[key]:
                _save_and_report(key, _merge(batch_by_key[key]))
    else:
        with Pool(processes=args.num_workers) as pool:
            for r in pool.imap_unordered(_worker, tasks):
                key = _config_key(r["lane_bias_q"], r["beta_alpha"])
                batch_by_key.setdefault(key, []).append(r)
                bar.update(r["num_rollouts"])
                bar.set_postfix(key=key, fails=r["num_failures"])
                if len(batch_by_key[key]) == chunks_expected[key]:
                    _save_and_report(key, _merge(batch_by_key[key]))

    bar.close()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} min")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
