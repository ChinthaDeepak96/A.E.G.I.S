from datetime import datetime, timezone

import pytest

from core.tool_audit import (
    ToolAuditRecord,
)


def make_record(
    *,
    tool_name="test_tool",
    risk_category="low",
    decision="ALLOW",
    confirmed=False,
    executed=True,
    success=True,
    arguments=None,
    result=None,
    error=None,
    timestamp=None,
):
    return ToolAuditRecord(
        tool_name=tool_name,
        risk_category=risk_category,
        decision=decision,
        confirmed=confirmed,
        executed=executed,
        success=success,
        arguments=(
            {}
            if arguments is None
            else arguments
        ),
        result=result,
        error=error,
        timestamp=timestamp,
    )


# =========================================================
# BASIC RECORD CONTRACT
# =========================================================


def test_audit_record_can_be_created():
    record = make_record()

    assert (
        record.tool_name
        == "test_tool"
    )

    assert (
        record.risk_category
        == "low"
    )

    assert (
        record.decision
        == "ALLOW"
    )

    assert (
        record.confirmed
        is False
    )

    assert (
        record.executed
        is True
    )

    assert (
        record.success
        is True
    )


def test_audit_record_stores_arguments():
    arguments = {
        "path": "test.txt",
        "limit": 10,
    }

    record = make_record(
        arguments=arguments
    )

    assert (
        record.arguments
        == arguments
    )


def test_audit_record_stores_result():
    record = make_record(
        result="command completed"
    )

    assert (
        record.result
        == "command completed"
    )


def test_audit_record_stores_error():
    record = make_record(
        executed=False,
        success=False,
        error="permission denied",
    )

    assert (
        record.error
        == "permission denied"
    )


# =========================================================
# CONFIRMATION STATE
# =========================================================


def test_unconfirmed_execution_is_recorded():
    record = make_record(
        decision="CONFIRM",
        confirmed=False,
        executed=False,
        success=False,
    )

    assert (
        record.decision
        == "CONFIRM"
    )

    assert (
        record.confirmed
        is False
    )

    assert (
        record.executed
        is False
    )

    assert (
        record.success
        is False
    )


def test_confirmed_execution_is_recorded():
    record = make_record(
        decision="CONFIRM",
        confirmed=True,
        executed=True,
        success=True,
        result="executed",
    )

    assert (
        record.confirmed
        is True
    )

    assert (
        record.executed
        is True
    )

    assert (
        record.success
        is True
    )


# =========================================================
# FAILURE STATES
# =========================================================


def test_failed_execution_can_be_recorded():
    record = make_record(
        executed=False,
        success=False,
        error="handler failure",
    )

    assert (
        record.executed
        is False
    )

    assert (
        record.success
        is False
    )

    assert (
        record.error
        == "handler failure"
    )


def test_successful_execution_can_have_result():
    record = make_record(
        executed=True,
        success=True,
        result="success",
    )

    assert (
        record.result
        == "success"
    )

    assert (
        record.error
        is None
    )


# =========================================================
# TIMESTAMP
# =========================================================


def test_audit_record_accepts_timestamp():
    timestamp = datetime(
        2026,
        8,
        21,
        15,
        30,
        tzinfo=timezone.utc,
    )

    record = make_record(
        timestamp=timestamp
    )

    assert (
        record.timestamp
        == timestamp
    )


def test_audit_record_creates_timestamp_when_missing():
    before = datetime.now(
        timezone.utc
    )

    record = make_record()

    after = datetime.now(
        timezone.utc
    )

    assert (
        record.timestamp
        is not None
    )

    assert (
        before
        <= record.timestamp
        <= after
    )


def test_timestamp_is_timezone_aware():
    record = make_record()

    assert (
        record.timestamp.tzinfo
        is not None
    )


# =========================================================
# SERIALIZATION
# =========================================================


def test_audit_record_to_dict():
    record = make_record(
        tool_name="read_file",
        risk_category="low",
        decision="ALLOW",
        confirmed=False,
        executed=True,
        success=True,
        arguments={
            "path": "hello.txt"
        },
        result="hello",
    )

    data = record.to_dict()

    assert (
        data["tool_name"]
        == "read_file"
    )

    assert (
        data["risk_category"]
        == "low"
    )

    assert (
        data["decision"]
        == "ALLOW"
    )

    assert (
        data["confirmed"]
        is False
    )

    assert (
        data["executed"]
        is True
    )

    assert (
        data["success"]
        is True
    )

    assert (
        data["arguments"]
        == {
            "path": "hello.txt"
        }
    )

    assert (
        data["result"]
        == "hello"
    )

    assert (
        data["error"]
        is None
    )

    assert (
        "timestamp"
        in data
    )


def test_audit_record_from_dict():
    data = {
        "tool_name": "run_command",
        "risk_category": "high",
        "decision": "CONFIRM",
        "confirmed": True,
        "executed": True,
        "success": True,
        "arguments": {
            "command": "dir"
        },
        "result": "completed",
        "error": None,
        "timestamp": (
            "2026-08-21T15:30:00+00:00"
        ),
    }

    record = (
        ToolAuditRecord.from_dict(
            data
        )
    )

    assert (
        record.tool_name
        == "run_command"
    )

    assert (
        record.risk_category
        == "high"
    )

    assert (
        record.decision
        == "CONFIRM"
    )

    assert (
        record.confirmed
        is True
    )

    assert (
        record.executed
        is True
    )

    assert (
        record.success
        is True
    )

    assert (
        record.arguments
        == {
            "command": "dir"
        }
    )

    assert (
        record.result
        == "completed"
    )


def test_audit_record_round_trip():
    original = make_record(
        tool_name="system_info",
        risk_category="low",
        decision="ALLOW",
        arguments={
            "detail": True
        },
        result="system data",
    )

    restored = (
        ToolAuditRecord.from_dict(
            original.to_dict()
        )
    )

    assert (
        restored.tool_name
        == original.tool_name
    )

    assert (
        restored.risk_category
        == original.risk_category
    )

    assert (
        restored.decision
        == original.decision
    )

    assert (
        restored.confirmed
        == original.confirmed
    )

    assert (
        restored.executed
        == original.executed
    )

    assert (
        restored.success
        == original.success
    )

    assert (
        restored.arguments
        == original.arguments
    )

    assert (
        restored.result
        == original.result
    )

    assert (
        restored.error
        == original.error
    )


# =========================================================
# INPUT SAFETY
# =========================================================


def test_audit_record_does_not_modify_arguments():
    arguments = {
        "command": "dir",
        "value": 42,
    }

    original = dict(
        arguments
    )

    record = make_record(
        arguments=arguments
    )

    assert (
        record.arguments
        == original
    )


def test_empty_arguments_are_supported():
    record = make_record(
        arguments={}
    )

    assert (
        record.arguments
        == {}
    )


def test_none_result_is_supported():
    record = make_record(
        result=None
    )

    assert (
        record.result
        is None
    )


def test_none_error_is_supported():
    record = make_record(
        error=None
    )

    assert (
        record.error
        is None
    )


# =========================================================
# DISTINCT RECORDS
# =========================================================


def test_each_audit_record_is_independent():
    first = make_record(
        tool_name="first"
    )

    second = make_record(
        tool_name="second"
    )

    assert (
        first.tool_name
        == "first"
    )

    assert (
        second.tool_name
        == "second"
    )

    assert (
        first is not second
    )


# =========================================================
# FAILED AND CONFIRMED EXECUTION
# =========================================================


def test_confirmed_failed_execution_contains_all_evidence():
    record = make_record(
        tool_name="delete_file",
        risk_category="high",
        decision="CONFIRM",
        confirmed=True,
        executed=True,
        success=False,
        arguments={
            "path": "important.txt"
        },
        error="operation failed",
    )

    assert (
        record.tool_name
        == "delete_file"
    )

    assert (
        record.risk_category
        == "high"
    )

    assert (
        record.decision
        == "CONFIRM"
    )

    assert (
        record.confirmed
        is True
    )

    assert (
        record.executed
        is True
    )

    assert (
        record.success
        is False
    )

    assert (
        record.error
        == "operation failed"
    )


def test_denied_tool_contains_no_execution_result():
    record = make_record(
        tool_name="dangerous_tool",
        risk_category="high",
        decision="CONFIRM",
        confirmed=False,
        executed=False,
        success=False,
    )

    assert (
        record.executed
        is False
    )

    assert (
        record.success
        is False
    )

    assert (
        record.result
        is None
    )