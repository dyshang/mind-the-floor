"""Label-aware constant floor on tau-bench (Section 5, Appendix C.3).

Policy: one constant respond message that contains every expected output of
the env's test tasks, then transfer_to_human_agents. It never writes, so it
passes a task iff replaying the ground-truth actions leaves the database
unchanged and every expected output is a substring of the message (lowercased,
commas removed: the rule in Env.calculate_reward).

Two independent counts, which must agree:
  (1) the predicate above, computed directly;
  (2) the installed package's own Env.calculate_reward, fed that action history.
Writes results/tau_label_aware.json (writes nothing else); check_appendix_numbers.py
checks it offline. Needs the tau-bench package:
  .venv/bin/python scripts/executed/tau_label_aware.py
"""
import importlib
import json
import sys
from pathlib import Path

from tau_bench.envs.base import consistent_hash, to_hashable
from tau_bench.types import RESPOND_ACTION_NAME, Action

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tau_static_floor import task_set_digest  # noqa: E402  (pure function, writes nothing)

recorded = json.loads((REPO / "results" / "tau_static_floor.json").read_text())["envs"]
out = {}
for envname in ("airline", "retail"):
    load_data = importlib.import_module(f"tau_bench.envs.{envname}.data").load_data
    tools_mod = importlib.import_module(f"tau_bench.envs.{envname}.tools")
    tool_map = {t.get_info()["function"]["name"]: t for t in tools_mod.ALL_TOOLS}
    tasks_mod = importlib.import_module(f"tau_bench.envs.{envname}.tasks_test")
    tasks = getattr(tasks_mod, next(a for a in ("TASKS_TEST", "TASKS") if hasattr(tasks_mod, a)))
    digest = task_set_digest(tasks)
    assert digest == recorded[envname]["task_set_sha256"], f"{envname}: task set differs from the recorded one"

    message = " ".join(o for t in tasks for o in t.outputs)
    init_hash = consistent_hash(to_hashable(load_data()))
    rows = []
    for i, task in enumerate(tasks):
        data = load_data()
        for a in task.actions:
            if a.name != RESPOND_ACTION_NAME:
                try:
                    tool_map[a.name].invoke(data=data, **a.kwargs)
                except Exception:
                    pass
        unchanged = consistent_hash(to_hashable(data)) == init_hash
        found = [o.lower() in message.lower().replace(",", "") for o in task.outputs]
        rows.append({"task_index": i, "db_unchanged": unchanged, "n_outputs": len(task.outputs),
                     "outputs_found": all(found), "outputs_with_comma": [o for o in task.outputs if "," in o]})

    never_write = sum(r["db_unchanged"] and r["n_outputs"] == 0 for r in rows)
    label_aware = sum(r["db_unchanged"] and r["outputs_found"] for r in rows)

    # (2) the package's own reward function
    import tau_bench.envs.base as base
    base.load_user = lambda **kw: None  # no user simulator: calculate_reward never calls it
    from tau_bench.envs import get_env
    env = get_env(envname, user_strategy="llm", user_model="unused", task_split="test",
                  user_provider="openai", task_index=0)
    pkg = 0
    for i, task in enumerate(tasks):
        env.task = task
        env.data = load_data()  # the policy never writes, so this is its final state
        env.actions = [Action(name=RESPOND_ACTION_NAME, kwargs={"content": message}),
                       Action(name="transfer_to_human_agents", kwargs={"summary": "constant"})]
        pkg += env.calculate_reward().reward > 0

    out[envname] = {"n": len(tasks), "task_set_sha256": digest,
                    "passes_task_indices": [r["task_index"] for r in rows if r["db_unchanged"] and r["outputs_found"]],
                    "never_write_no_output (paper's floor)": never_write,
                    "label_aware_constant (predicate)": label_aware,
                    "label_aware_constant (package reward)": int(pkg),
                    "message_chars": len(message),
                    "db_unchanged_but_needs_output": [r["task_index"] for r in rows if r["db_unchanged"] and r["n_outputs"] > 0],
                    "outputs_with_comma": [r for r in rows if r["outputs_with_comma"]]}
    print(envname, json.dumps({k: v for k, v in out[envname].items() if k != "outputs_with_comma"}))

out["generator"] = "scripts/executed/tau_label_aware.py"
(REPO / "results" / "tau_label_aware.json").write_text(json.dumps(out, indent=2) + "\n")
