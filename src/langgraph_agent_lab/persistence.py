"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    Memory is the zero-infrastructure default. SQLite uses an explicit sqlite3
    connection because recent langgraph-checkpoint-sqlite versions may expose
    from_conn_string() as a context manager rather than a checkpointer instance.
    """
    normalized_kind = (kind or "memory").lower()
    if normalized_kind == "none":
        return None
    if normalized_kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if normalized_kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError("SQLite checkpointer requires: pip install langgraph-checkpoint-sqlite") from exc

        path = Path(database_url or "outputs/checkpoints.sqlite")
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        checkpointer = SqliteSaver(conn=conn)
        setup = getattr(checkpointer, "setup", None)
        if callable(setup):
            setup()
        return checkpointer
    if normalized_kind == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise RuntimeError("Postgres checkpointer requires: pip install langgraph-checkpoint-postgres") from exc
        return PostgresSaver.from_conn_string(database_url or "")
    raise ValueError(f"Unknown checkpointer kind: {kind}")
