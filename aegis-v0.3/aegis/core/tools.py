"""
Tool Registry (architecture doc section 15).

Each tool declares a risk_category up front. Guardian Lite (see
core/guardian.py) uses this field to decide whether a tool call
executes automatically or needs user confirmation -- it never has
to inspect what the tool actually does.

v0.3 adds the Computer Agent's tools: process listing, open-window
listing, launching/closing applications, and keyboard/mouse control.
GUI-dependent tools (list_open_windows, type_text, click_mouse) import
their backing library lazily, inside the handler, and catch broad
Exception rather than just ImportError -- on Linux without a display,
importing pyautogui raises a KeyError looking up the DISPLAY env var,
not an ImportError, so a narrower except would let that crash through
uncaught. This also means the registry itself, and every risk_category
declared in it, can be imported and tested with no display attached.
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


# ---------------------------------------------------------------------------
# v0.3: Computer Agent tools
# ---------------------------------------------------------------------------


def _list_processes(limit: int = 40) -> str:
    import psutil

    rows = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            rows.append(f"{proc.info['pid']}\t{proc.info['name']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    rows.sort(key=lambda r: r.split("\t")[1].lower())
    if limit:
        rows = rows[:limit]
    return "\n".join(rows) if rows else "(no processes found)"


def _list_open_windows() -> str:
    try:
        import pygetwindow as gw
    except Exception as exc:  # noqa: BLE001 -- covers ImportError and display-init errors alike
        return f"Error: window listing isn't available in this environment ({exc})."

    try:
        titles = [t for t in gw.getAllTitles() if t.strip()]
    except Exception as exc:  # noqa: BLE001
        return f"Error listing windows: {exc}"

    return "\n".join(titles) if titles else "(no open windows found)"


def _open_application(name: str) -> str:
    import os
    import platform

    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(name)  # noqa: S606 -- Windows-only, launches by registered name/path
        elif system == "Darwin":
            subprocess.Popen(["open", "-a", name])
        else:
            subprocess.Popen([name])
        return f"Launched '{name}'."
    except Exception as exc:  # noqa: BLE001
        return f"Error launching '{name}': {exc}"


def _close_application(name: str) -> str:
    import psutil

    terminated = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            proc_name = proc.info["name"] or ""
            if name.lower() in proc_name.lower():
                proc.terminate()
                terminated.append(f"{proc_name} (pid {proc.info['pid']})")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not terminated:
        return f"No running process matching '{name}' was found."
    return "Terminated: " + ", ".join(terminated)


def _type_text(text: str) -> str:
    try:
        import pyautogui
    except Exception as exc:  # noqa: BLE001
        return f"Error: keyboard control isn't available in this environment ({exc})."

    try:
        pyautogui.typewrite(text, interval=0.02)
        return f"Typed: {text!r}"
    except Exception as exc:  # noqa: BLE001
        return f"Error while typing: {exc}"


def _click_mouse(x: int, y: int) -> str:
    try:
        import pyautogui
    except Exception as exc:  # noqa: BLE001
        return f"Error: mouse control isn't available in this environment ({exc})."

    try:
        pyautogui.moveTo(x, y)
        pyautogui.click()
        return f"Clicked at ({x}, {y})."
    except Exception as exc:  # noqa: BLE001
        return f"Error while clicking: {exc}"


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
    "list_processes": Tool(
        name="list_processes",
        description="List currently running processes (pid and name).",
        risk_category=RISK_LOW,
        parameters={
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "Max number of processes to return."}},
        },
        handler=_list_processes,
    ),
    "list_open_windows": Tool(
        name="list_open_windows",
        description=(
            "List the titles of currently open windows. This is a text-only stand-in for "
            "'reading the screen' -- actual visual/OCR screen reading is v0.9 (Vision) scope."
        ),
        risk_category=RISK_LOW,
        parameters={"type": "object", "properties": {}},
        handler=_list_open_windows,
    ),
    "open_application": Tool(
        name="open_application",
        description="Launch an application by name.",
        risk_category=RISK_LOW,  # matches architecture doc section 5.3's own LOW-risk example list
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Name of the application to launch."}},
            "required": ["name"],
        },
        handler=_open_application,
    ),
    "close_application": Tool(
        name="close_application",
        description="Terminate all running processes whose name matches the given text.",
        risk_category=RISK_MEDIUM,  # can lose unsaved work; not a LOW no-consequence read
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Process name (or substring) to terminate."}},
            "required": ["name"],
        },
        handler=_close_application,
    ),
    "type_text": Tool(
        name="type_text",
        description="Type text via the keyboard into whichever window currently has focus.",
        risk_category=RISK_HIGH,  # can type into any field, including ones AEGIS can't see the contents of
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to type."}},
            "required": ["text"],
        },
        handler=_type_text,
    ),
    "click_mouse": Tool(
        name="click_mouse",
        description="Move the mouse to (x, y) screen coordinates and click.",
        risk_category=RISK_HIGH,  # can trigger any on-screen action, including destructive ones
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate."},
                "y": {"type": "integer", "description": "Y coordinate."},
            },
            "required": ["x", "y"],
        },
        handler=_click_mouse,
    ),
}


def anthropic_tool_schemas() -> list[dict]:
    """Format the registry as Anthropic API tool definitions."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in TOOLS.values()
    ]
