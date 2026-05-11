"""CLI for the lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


def _checkpointer_settings(cfg: dict) -> tuple[str, str | None]:
    persistence = cfg.get("persistence") or {}
    kind = persistence.get("checkpointer") or cfg.get("checkpointer", "memory")
    database_url = (
        persistence.get("sqlite_path")
        or persistence.get("database_url")
        or cfg.get("sqlite_path")
        or cfg.get("database_url")
    )
    return str(kind), database_url


def _state_history_available(graph: object, config: dict) -> bool:
    get_state_history = getattr(graph, "get_state_history", None)
    if not callable(get_state_history):
        return False
    try:
        history = get_state_history(config)
        return next(iter(history), None) is not None
    except Exception as exc:
        typer.echo(f"State history evidence unavailable: {exc}", err=True)
        return False


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer_kind, database_url = _checkpointer_settings(cfg)
    checkpointer = build_checkpointer(checkpointer_kind, database_url)
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    history_config = None
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        final_state = graph.invoke(state, config=run_config)
        metrics.append(metric_from_state(final_state, scenario.expected_route.value, scenario.requires_approval))
        history_config = history_config or run_config
    report = summarize_metrics(metrics)
    report.resume_success = bool(history_config) and _state_history_available(graph, history_config)
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(f"Wrote metrics to {output}")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


if __name__ == "__main__":
    app()
