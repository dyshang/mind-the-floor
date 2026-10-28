"""Execution-free replication of the tau-bench never-write floor (appendix C.3).

A task passes under any never-writing degenerate policy iff replaying its
ground-truth actions leaves the database hash unchanged AND the task requires
no communicated outputs. This counts that predicate directly over the released
task files -- no user simulator, no API key -- and cross-checks the executed
runs in results/tau_<env>_transfer.jsonl per task.

Requires the tau-bench package (same venv as tau_bench_degenerate.py):
  ~/venvs/tau/bin/python scripts/executed/tau_static_floor.py
"""

from __future__ import annotations

import datetime
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path

from tau_bench.envs.base import consistent_hash, to_hashable
from tau_bench.types import RESPOND_ACTION_NAME

RESULTS = Path(__file__).resolve().parents[2] / "results"


def task_set_digest(tasks) -> str:
    """SHA-256 over the task definitions the predicate is computed from.

    tau-bench ships as version 0.1.0 from a plain sdist, so the version string
    is not a drift signal -- it has never changed. The task set is what the
    predicate actually depends on, so hash that instead: if upstream edits an
    action or an expected output, this digest moves and the recorded floor is
    known to be stale rather than silently wrong. Same role as the per-file
    sha256 in results/bfcl_slice_counts.json.
    """
    canonical = [
        {
            "actions": [{"name": a.name, "kwargs": a.kwargs} for a in t.actions],
            "outputs": list(t.outputs),
        }
        for t in tasks
    ]
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def static_floor(envname: str) -> tuple[list[bool], str, int]:
    load_data = importlib.import_module(f"tau_bench.envs.{envname}.data").load_data
    tools_mod = importlib.import_module(f"tau_bench.envs.{envname}.tools")
    tool_map = {t.get_info()["function"]["name"]: t for t in tools_mod.ALL_TOOLS}
    tasks_mod = importlib.import_module(f"tau_bench.envs.{envname}.tasks_test")
    tasks_attr = next(a for a in ("TASKS_TEST", "TASKS") if hasattr(tasks_mod, a))
    tasks = getattr(tasks_mod, tasks_attr)

    init_hash = consistent_hash(to_hashable(load_data()))
    pred = []
    raised = 0
    for task in tasks:
        data = load_data()
        for action in task.actions:
            if action.name == RESPOND_ACTION_NAME:
                continue
            try:
                tool_map[action.name].invoke(data=data, **action.kwargs)
            except Exception:
                # Assumption: an errored ground-truth action leaves `data`
                # unchanged, so continuing past it matches harness behaviour. A
                # tool that mutates before raising would break it.
                #
                # We count the raises rather than arguing the assumption is
                # safe. Citing the agreement with the executed runs would be
                # circular -- that agreement is what this predicate exists to
                # establish -- whereas a recorded count of zero makes the
                # assumption vacuous and the identity assumption-free.
                raised += 1
        unchanged = consistent_hash(to_hashable(data)) == init_hash
        pred.append(unchanged and len(task.outputs) == 0)
    return pred, task_set_digest(tasks), raised


def main() -> None:
    try:
        version = importlib.metadata.version("tau_bench")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"

    provenance = {
        "_note": "Provenance for the execution-free tau-bench floor of appendix C.3. "
                 "The per-task predicates in tau_<env>_static_floor.jsonl are derived "
                 "artifacts: regenerating them needs tau-bench installed, but checking "
                 "the floor and the agreement figure against them does not. "
                 "task_set_sha256 is the drift signal -- if upstream edits the task "
                 "set it moves, and the recorded counts are stale rather than wrong. "
                 "scripts/check_appendix_numbers.py verifies the counts offline.",
        "generated": datetime.date.today().isoformat(),
        "tau_bench_version": version,
        "generator": "scripts/executed/tau_static_floor.py",
        "envs": {},
    }

    for envname in ("airline", "retail"):
        pred, digest, raised = static_floor(envname)
        print(f"{envname}: static never-write floor = {sum(pred)}/{len(pred)}")

        # Persist the per-task predicate. Without it the agreement figure the
        # appendix quotes (50/50, 115/115) can only be rechecked by installing
        # tau-bench and re-running this script; with it, anyone can recompute
        # the agreement offline from the released JSONL.
        out = RESULTS / f"tau_{envname}_static_floor.jsonl"
        with open(out, "w") as f:
            for i, p in enumerate(pred):
                f.write(json.dumps({"env": envname, "task_index": i,
                                    "never_write_passes": p}) + "\n")
        print(f"  wrote {out.name}")

        env_record = {
            "n_tasks": len(pred),
            "static_floor": sum(pred),
            "task_set_sha256": digest,
            "per_task": f"results/tau_{envname}_static_floor.jsonl",
            "replay_actions_raised": raised,
        }

        jl = RESULTS / f"tau_{envname}_transfer.jsonl"
        if jl.exists():
            measured = [json.loads(line)["reward"] for line in open(jl)]
            agree = sum((r > 0) == p for r, p in zip(measured, pred))
            print(f"  executed transfer = {sum(r > 0 for r in measured)}/{len(measured)}; "
                  f"per-task agreement = {agree}/{len(measured)}")
            env_record["executed_transfer"] = sum(r > 0 for r in measured)
            env_record["agreement"] = agree
        provenance["envs"][envname] = env_record

    meta = RESULTS / "tau_static_floor.json"
    meta.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {meta.name}")


if __name__ == "__main__":
    main()
