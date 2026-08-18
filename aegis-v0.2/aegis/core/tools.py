"""
Tool Registry (architecture doc section 15, v0.2 scope).

Each tool declares a risk_category up front. Guardian Lite (see
core/guardian.py) uses this field to decide whether a tool call
executes automatically or needs user confirmation -- it never has
to inspect what the tool actually does.

v0.2 keeps this deliberately small: four tools, each with an
obviously-correct risk category, so we can prove out the "tools +
Guardian Lite" wiring before growing the registry. A richer schema
(permissions, timeouts, rollback strategy -- section 15) gets added
once real usage shows what's actually needed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

RISK_LOW = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH = "HIGH"


@dataclass
class Tool:
    name: str
    description: str
    risk_category: str
    parameters: dict  # JSON schema for input, in Anthropic tool-use format
    handler: Callable[..., str]


def _list_files(path: str = ".") -> str:
    target = Path(path).expanduser()
    if not target.exists():
        return f"Error: path does not exist: {path}"
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    return "\n".join(entries) if entries else "(empty directory)"


def _read_file(path: str) -> str:
    target = Path(path).expanduser()
    if not target.exists():
        return f"Error: file does not exist: {path}"
    if not target.is_file():
        return f"Error: not a file: {path}"
    try:
        return target.read_text(errors="replace")[:8000]
    except Exception as exc:  # noqa: BLE001 -- surfaced to the caller, not swallowed
        return f"Error reading file: {exc}"


def _run_command(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout + result.stderr
        return output[:4000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 15s"
    except Exception as exc:  # noqa: BLE001
        return f"Error running command: {exc}"


def _system_info() -> str:
    import platform

    return (
        f"OS: {platform.system()} {platform.release()}\n"
        f"Python: {platform.python_version()}\n"
        f"Machine: {platform.machine()}"
    )


TOOLS: dict[str, Tool] = {
    "list_files": Tool(
        name="list_files",
        description="List files and directories at a given path (defaults to current directory).",
        risk_category=RISK_LOW,
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path to list."}},
        },
        handler=_list_files,
    ),
    "read_file": Tool(
        name="read_file",
        description="Read and return the text contents of a file.",
        risk_category=RISK_LOW,
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path of the file to read."}},
            "required": ["path"],
        },
        handler=_read_file,
    ),
    "system_info": Tool(
        name="system_info",
        description="Report basic information about the host operating system.",
        risk_category=RISK_LOW,
        parameters={"type": "object", "properties": {}},
        handler=_system_info,
    ),
    "run_command": Tool(
        name="run_command",
        description="Execute a shell command on the host machine and return its output.",
        risk_category=RISK_HIGH,
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command to execute."}},
            "required": ["command"],
        },
        handler=_run_command,
    ),
}


def anthropic_tool_schemas() -> list[dict]:
    """Format the registry as Anthropic API tool definitions."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in TOOLS.values()
    ]
