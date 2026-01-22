from langsmith import Client
from datetime import datetime, timedelta
import csv

import config

# ==== CONFIGURE THESE ====
PROJECT_NAME = "default"
OUTPUT_CSV = f"./data/{PROJECT_NAME}_steps_duration_cost.csv"

# Optional: limit to recent runs, e.g. last 7 days;
# set to None to fetch all time
DAYS_BACK = 1
# =========================


def iso_to_dt(ts: datetime):
    if ts is None:
        return None
    return datetime.fromisoformat(ts.isoformat().replace("Z", "+00:00"))


def get_trace_ids_for_project(client: Client, project_name: str):
    """Return IDs of root runs (traces) for the given project."""
    kwargs = {
        "project_name": project_name,
        "is_root": True,
        "id_only": False,
    }
    if DAYS_BACK is not None:
        since = datetime.utcnow() - timedelta(days=DAYS_BACK)
        kwargs["start_time"] = since

    traces = client.list_runs(**kwargs)
    return [r.id for r in traces]


def get_traces_for_project(client: Client, project_name: str):
    """Return IDs of root runs (traces) for the given project."""
    kwargs = {
        "project_name": project_name,
        "is_root": True,
        "id_only": False,
    }
    if DAYS_BACK is not None:
        since = datetime.utcnow() - timedelta(days=DAYS_BACK)
        kwargs["start_time"] = since

    traces = client.list_runs(**kwargs)
    return list(traces)

def get_all_runs_for_trace(client: Client, trace_id: str):
    """Return all runs (parent + children) for a trace."""
    return list(client.list_runs(trace_id=trace_id))


def main():
    client = Client()

    #trace_ids = get_trace_ids_for_project(client, PROJECT_NAME)
    traces = get_traces_for_project(client, PROJECT_NAME)
    print(f"Found {len(traces)} traces in project '{PROJECT_NAME}'")

    fieldnames = [
        "trace_id",
        "content",
        "run_id",
        "parent_run_id",
        "name",
        "run_type",
        "start_time",
        "end_time",
        "duration_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "total_cost",
        "project_name",
    ]
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trace in traces:
            trace_id = trace.id 
            content = trace.inputs["messages"][0]["content"][0]["text"]
            runs = get_all_runs_for_trace(client, trace_id)

            for run in runs:
                start = iso_to_dt(run.start_time)
                end = iso_to_dt(run.end_time)
                duration_s = (
                    (end - start).total_seconds() if start and end else None
                )

                row = {
                    "trace_id": run.trace_id,
                    "content": content,
                    "run_id": run.id,
                    "parent_run_id": run.parent_run_id,
                    "name": run.name,
                    "run_type": run.run_type,
                    "start_time": run.start_time,
                    "end_time": run.end_time,
                    "duration_ms": round(duration_s, 2),
                    "prompt_tokens": run.prompt_tokens,
                    "completion_tokens": run.completion_tokens,
                    "total_tokens": run.total_tokens,
                    "total_cost": run.total_cost,
                    "project_name": PROJECT_NAME,
                }
                writer.writerow(row)

    print(f"Wrote CSV to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()