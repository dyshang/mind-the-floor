"""ALFWorld degenerate policies: fixed legal action / uniform random legal action.

Requires: pip install alfworld && alfworld-download
  (https://github.com/alfworld/alfworld; text-only TWEnv, no GPU)

Policies (paper Table 2):
  fixed  : always emit the same admissible-command slot (first "go to" if
           present, else first admissible command)
  random : uniform draw from the admissible commands each step

Run:
  python alfworld_degenerate.py --policy random --episodes 134 --seed 0
  python alfworld_degenerate.py --policy random --seeds 0-19   # checklist item 4
Output: results/alfworld_<policy>.jsonl (per-episode success, steps, seed).

Checklist item 4 asks for >= 20 replicates of a stochastic policy, so --seeds
takes a range and records every episode of every replicate in one file. The
deterministic `fixed` policy needs exactly one. Harness versions are recorded
beside the results: an ALFWorld floor is only comparable across runs that share
an engine, and nothing in an earlier draft pinned one.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"
MAX_STEPS = 50  # standard ALFWorld horizon


def pick(policy: str, admissible: list[str], rng: random.Random) -> str:
    if policy == "random":
        return rng.choice(admissible)
    for cmd in admissible:  # fixed: deterministic preference order
        if cmd.startswith("go to"):
            return cmd
    return admissible[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["fixed", "random"], required=True)
    parser.add_argument("--episodes", type=int, default=134,
                        help="eval-out-of-distribution split size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default=None,
                        help="inclusive range 'a-b' of replicate seeds; overrides --seed")
    parser.add_argument("--out", type=Path, default=None,
                        help="write here instead of results/alfworld_<policy>.jsonl")
    args = parser.parse_args()
    seeds = [args.seed]
    if args.seeds:
        lo, hi = (int(x) for x in args.seeds.split("-"))
        seeds = list(range(lo, hi + 1))

    import alfworld.agents.environment as environment
    import yaml
    config = yaml.safe_load(open("base_config.yaml"))  # from alfworld repo
    env_type = config["env"]["type"]
    get_env = getattr(environment, "get_environment", None)
    env_cls = get_env(env_type) if get_env else getattr(environment, env_type)
    env = env_cls(config, train_eval="eval_out_of_distribution")
    env = env.init_env(batch_size=1)

    RESULTS.mkdir(exist_ok=True)
    out = args.out or RESULTS / f"alfworld_{args.policy}.jsonl"
    per_seed = {}
    with open(out, "w") as f:
        for seed in seeds:
            rng = random.Random(seed)
            successes = 0
            for ep in range(args.episodes):
                obs, info = env.reset()
                done, won, steps = False, False, 0
                while not done and steps < MAX_STEPS:
                    cmd = pick(args.policy, info["admissible_commands"][0], rng)
                    obs, _, dones, info = env.step([cmd])
                    done = dones[0]
                    won = bool(info.get("won", [False])[0])
                    steps += 1
                successes += won
                f.write(json.dumps({"episode": ep, "success": won, "steps": steps,
                                    "policy": args.policy, "seed": seed}) + "\n")
            per_seed[seed] = successes
            print(f"  seed {seed}: {successes}/{args.episodes} = "
                  f"{successes/args.episodes:.4f}")
    total = sum(per_seed.values())
    n = args.episodes * len(seeds)
    print(f"{args.policy}: {total}/{n} = {total/n:.4f} over {len(seeds)} replicate(s)")
    print(f"wrote {out}")
    if args.out is None:
        import importlib.metadata as md
        def ver(pkg):
            try:
                return md.version(pkg)
            except Exception:
                return "unknown"
        meta = RESULTS / f"alfworld_{args.policy}_meta.json"
        meta.write_text(json.dumps({
            "policy": args.policy, "episodes_per_seed": args.episodes,
            "seeds": seeds, "successes_per_seed": per_seed,
            "successes": total, "episodes_total": n, "rate": total / n,
            "max_steps": MAX_STEPS, "split": "eval_out_of_distribution",
            "alfworld_version": ver("alfworld"),
            "textworld_version": ver("textworld"),
        }, indent=1) + "\n")
        print(f"wrote {meta}")


if __name__ == "__main__":
    main()
