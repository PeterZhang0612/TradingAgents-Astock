"""Regression checks for analyst/tool-node synchronization.

LLMs bind tools inside each analyst node, while LangGraph executes tool calls
through the matching ToolNode. Both lists must contain the same externally
prompted tools, otherwise the model can request a tool that the graph cannot
execute.
"""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _names_in_assigned_list(source_path: Path, function_name: str, target_name: str) -> set[str]:
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for child in ast.walk(node):
                if not isinstance(child, ast.Assign):
                    continue
                if not any(isinstance(t, ast.Name) and t.id == target_name for t in child.targets):
                    continue
                if isinstance(child.value, ast.List):
                    return {elt.id for elt in child.value.elts if isinstance(elt, ast.Name)}
    raise AssertionError(f"Could not find {target_name} list in {source_path}:{function_name}")


def _toolnode_names(source_path: Path, toolnode_key: str) -> set[str]:
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(module):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and key.value == toolnode_key):
                continue
            if not (isinstance(value, ast.Call) and getattr(value.func, "id", None) == "ToolNode"):
                continue
            if not value.args or not isinstance(value.args[0], ast.List):
                continue
            return {elt.id for elt in value.args[0].elts if isinstance(elt, ast.Name)}
    raise AssertionError(f"Could not find ToolNode for {toolnode_key} in {source_path}")


def test_hot_money_smart_search_is_bound_and_executable():
    analyst_tools = _names_in_assigned_list(
        ROOT / "tradingagents" / "agents" / "analysts" / "hot_money_tracker.py",
        "hot_money_tracker_node",
        "tools",
    )
    graph_tools = _toolnode_names(
        ROOT / "tradingagents" / "graph" / "trading_graph.py",
        "hot_money",
    )

    assert "smart_search_cli" in analyst_tools
    assert "smart_search_cli" in graph_tools
