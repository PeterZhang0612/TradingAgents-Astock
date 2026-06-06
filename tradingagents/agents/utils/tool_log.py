"""Tool call logging: records which agent called which tool, params, timing, errors."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler


class ToolLogCallbackHandler(BaseCallbackHandler):
    """Callback that logs every tool invocation to a JSONL file."""

    def __init__(self, output_dir: str = "") -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._entries: List[Dict[str, Any]] = []
        self._pending: Dict[str, float] = {}  # run_id → start time
        self._pending_params: Dict[str, str] = {}  # run_id → tool name + params

        # Resolve log path
        base = Path(output_dir) if output_dir else Path.home() / ".tradingagents"
        base.mkdir(parents=True, exist_ok=True)
        self._log_path = base / "tool_calls.jsonl"

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Record tool start time."""
        tool_name = serialized.get("name", "unknown")

        # Try to infer which analyst is calling from metadata/tags
        agent = "unknown"
        if tags:
            agent_tags = [t for t in tags if "graph" in t.lower() or "tool" in t.lower()]
            if agent_tags:
                agent = agent_tags[0]
        # Fallback: check metadata for langgraph node info
        if metadata and agent == "unknown":
            agent = metadata.get("langgraph_node", metadata.get("checkpoint_ns", "unknown"))

        with self._lock:
            self._pending[run_id] = time.time()
            self._pending_params[run_id] = f"{agent}|{tool_name}|{input_str[:250]}"

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Record tool completion with success."""
        self._flush(run_id, error=None)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Record tool failure."""
        self._flush(run_id, error=str(error)[:200])

    def _flush(self, run_id: str, error: Optional[str]) -> None:
        with self._lock:
            start = self._pending.pop(run_id, None)
            params = self._pending_params.pop(run_id, "")
            if start is None:
                return

            elapsed_ms = round((time.time() - start) * 1000)
            agent, _, rest = params.partition("|")
            tool_name, _, input_preview = rest.partition("|")

            entry = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "agent": agent.strip() or "unknown",
                "tool": tool_name.strip() or "unknown",
                "params": input_preview.strip()[:300],
                "elapsed_ms": elapsed_ms,
            }
            if error:
                entry["error"] = error
            else:
                entry["error"] = ""

            self._entries.append(entry)
            self._write(entry)

    def _write(self, entry: Dict[str, Any]) -> None:
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @property
    def log_path(self) -> str:
        return str(self._log_path)
