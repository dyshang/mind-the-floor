"""tau-bench degenerate agents: constant transfer_to_human / immediate stop.

Requires: pip install tau-bench  (https://github.com/sierra-research/tau-bench)
User simulator uses a small API model; budget cap enforced by --max-episodes.
Anthropic works too: --user-model claude-haiku-4-5 --user-provider anthropic
(needs ANTHROPIC_API_KEY; OpenAI needs OPENAI_API_KEY).

Two policies (paper Table 2):
  transfer : always call transfer_to_human_agents with a fixed summary
  stop     : respond once with a canned message and end the episode

Run:
  python tau_bench_degenerate.py --env retail --policy transfer \
      --user-model gpt-4o-mini --max-episodes 50
Output: results/tau_<env>_<policy>.jsonl (per-episode reward, seed, params).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"
SEED = 0

CANNED = "I am unable to help with this request."


class DegenerateAgent:
    """Input-oblivious agent for tau-bench's agent interface."""

    def __init__(self, policy: str):
        self.policy = policy

    def solve(self, env, task_index: int) -> dict:
        env.reset(task_index=task_index)
        if self.policy == "transfer":
            action = {"name": "transfer_to_human_agents",
                      "kwargs": {"summary": CANNED}}
        else:  # stop
            action = {"name": "respond", "kwargs": {"content": CANNED}}
        # One constant action, then terminate the episode.
        from tau_bench.types import Action
        result = env.step(Action(name=action["name"], kwargs=action["kwargs"]))
        return {"task": task_index, "reward": result.reward,
                "policy": self.policy, "seed": SEED}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["retail", "airline"], required=True)
    parser.add_argument("--policy", choices=["transfer", "stop"], required=True)
    parser.add_argument("--user-model", default="gpt-4o-mini")
    parser.add_argument("--user-provider", default="openai",
                        help="litellm provider for the user simulator "
                             "(openai, anthropic, ...)")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="cap for API budget; None = full split")
    args = parser.parse_args()

    from tau_bench.envs import get_env
    env = get_env(args.env, user_strategy="llm", user_model=args.user_model,
                  user_provider=args.user_provider, task_split="test")
    n = len(env.tasks) if args.max_episodes is None else min(args.max_episodes,
                                                             len(env.tasks))
    agent = DegenerateAgent(args.policy)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"tau_{args.env}_{args.policy}.jsonl"
    rewards = []
    with open(out, "w") as f:
        for i in range(n):
            episode = agent.solve(env, i)
            rewards.append(episode["reward"])
            f.write(json.dumps(episode) + "\n")
    print(f"{args.env}/{args.policy}: pass^1 = {sum(rewards)/n:.4f} over {n} episodes")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
