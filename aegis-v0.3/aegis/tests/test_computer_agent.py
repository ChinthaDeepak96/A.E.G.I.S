"""
v0.3 Computer Agent tests.

list_processes and close_application are tested for real -- psutil
doesn't need a display. open_application's cross-platform dispatch is
tested by mocking platform.system() and the underlying OS call.
list_open_windows / type_text / click_mouse depend on a real display
(pygetwindow / pyautogui), so those are tested by injecting a fake
module into sys.modules rather than requiring an actual screen.

These tests are designed to run on Windows, macOS, and Linux.
"""

import os
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

from core.tools import (
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    TOOLS,
)


def test_new_tools_have_expected_risk_categories():
    assert TOOLS["list_processes"].risk_category == RISK_LOW
    assert TOOLS["list_open_windows"].risk_category == RISK_LOW
    assert TOOLS["open_application"].risk_category == RISK_LOW
    assert TOOLS["close_application"].risk_category == RISK_MEDIUM
    assert TOOLS["type_text"].risk_category == RISK_HIGH
    assert TOOLS["click_mouse"].risk_category == RISK_HIGH


# ---------------------------------------------------------------------------
# Real, no-display-needed: process listing and termination via psutil
# ---------------------------------------------------------------------------


def test_list_processes_returns_real_processes():
    result = TOOLS["list_processes"].handler()

    assert "\t" in result  # pid<TAB>name rows
    assert len(result.splitlines()) > 0


def test_list_processes_respects_limit():
    result = TOOLS["list_processes"].handler(limit=2)

    assert len(result.splitlines()) <= 2


def test_close_application_terminates_matching_process():
    """
    Verify that close_application() terminates a matching process.

    psutil.process_iter() is mocked so the test cannot accidentally
    terminate pytest, A.E.G.I.S., or another real process.
    """

    fake_process = MagicMock()
    fake_process.info = {
        "pid": 98765,
        "name": "aegis_test_process.exe",
    }

    with patch(
        "psutil.process_iter",
        return_value=[fake_process],
    ):
        result = TOOLS["close_application"].handler(
            name="aegis_test_process"
        )

    fake_process.terminate.assert_called_once()

    assert "Terminated:" in result
    assert "aegis_test_process.exe" in result
    assert "98765" in result

def test_close_application_reports_no_match_cleanly():
    result = TOOLS["close_application"].handler(
        name="definitely_not_a_real_process_xyz"
    )

    assert "No running process matching" in result


# ---------------------------------------------------------------------------
# Cross-platform dispatch logic for open_application
# ---------------------------------------------------------------------------


def test_open_application_dispatches_on_windows():
    with (
        patch("platform.system", return_value="Windows"),
        patch("os.startfile", create=True) as mock_start,
    ):
        result = TOOLS["open_application"].handler(
            name="notepad"
        )

        mock_start.assert_called_once_with("notepad")
        assert "Launched" in result


def test_open_application_dispatches_on_macos():
    with (
        patch("platform.system", return_value="Darwin"),
        patch("subprocess.Popen") as mock_popen,
    ):
        result = TOOLS["open_application"].handler(
            name="TextEdit"
        )

        mock_popen.assert_called_once_with(
            ["open", "-a", "TextEdit"]
        )

        assert "Launched" in result


def test_open_application_dispatches_on_linux():
    with (
        patch("platform.system", return_value="Linux"),
        patch("subprocess.Popen") as mock_popen,
    ):
        result = TOOLS["open_application"].handler(
            name="gedit"
        )

        mock_popen.assert_called_once_with(
            ["gedit"]
        )

        assert "Launched" in result


def test_open_application_reports_error_cleanly():
    with (
        patch("platform.system", return_value="Linux"),
        patch(
            "subprocess.Popen",
            side_effect=FileNotFoundError("no such app"),
        ),
    ):
        result = TOOLS["open_application"].handler(
            name="not_a_real_app"
        )

        assert "Error" in result


# ---------------------------------------------------------------------------
# GUI-dependent tools
#
# These tests use fake modules rather than requiring a physical display.
# ---------------------------------------------------------------------------


def test_list_open_windows_uses_pygetwindow(monkeypatch):
    fake_gw = MagicMock()

    fake_gw.getAllTitles.return_value = [
        "Terminal",
        "",
        "Browser",
    ]

    monkeypatch.setitem(
        sys.modules,
        "pygetwindow",
        fake_gw,
    )

    result = TOOLS["list_open_windows"].handler()

    assert "Terminal" in result
    assert "Browser" in result


def test_list_open_windows_fails_gracefully_without_display(
    monkeypatch,
):
    monkeypatch.delitem(
        sys.modules,
        "pygetwindow",
        raising=False,
    )

    def _raise_import(name, *args, **kwargs):
        if name == "pygetwindow":
            raise KeyError("DISPLAY")

        return __import__(
            name,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        "builtins.__import__",
        _raise_import,
    )

    result = TOOLS["list_open_windows"].handler()

    assert "Error" in result


def test_type_text_calls_pyautogui(monkeypatch):
    fake_pyautogui = MagicMock()

    monkeypatch.setitem(
        sys.modules,
        "pyautogui",
        fake_pyautogui,
    )

    result = TOOLS["type_text"].handler(
        text="hello world"
    )

    fake_pyautogui.typewrite.assert_called_once_with(
        "hello world",
        interval=0.02,
    )

    assert "Typed" in result


def test_click_mouse_calls_pyautogui(monkeypatch):
    fake_pyautogui = MagicMock()

    monkeypatch.setitem(
        sys.modules,
        "pyautogui",
        fake_pyautogui,
    )

    result = TOOLS["click_mouse"].handler(
        x=100,
        y=200,
    )

    fake_pyautogui.moveTo.assert_called_once_with(
        100,
        200,
    )

    fake_pyautogui.click.assert_called_once()

    assert "Clicked at (100, 200)" in result