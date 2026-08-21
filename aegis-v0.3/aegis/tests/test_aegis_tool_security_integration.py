from core.aegis import AEGIS
from core.llm_client import (
    LLMResponse,
    MockClient,
    ToolUseBlock,
)
from core.tool_audit_security import (
    ToolAuditSecurityAnalyzer,
)
from core.tool_audit_store import (
    SQLiteToolAuditStore,
)


def make_response(
    *blocks,
    stop_reason="end_turn",
):
    return LLMResponse(
        content=list(blocks),
        stop_reason=stop_reason,
    )


def make_tool_use(
    *,
    tool_id="tool-1",
    name="system_info",
    input_data=None,
):
    return ToolUseBlock(
        id=tool_id,
        name=name,
        input=(
            {}
            if input_data is None
            else input_data
        ),
    )


def make_aegis(
    tmp_path,
    responses,
):
    store = SQLiteToolAuditStore(
        str(
            tmp_path
            / "aegis_security.db"
        )
    )

    client = MockClient(
        responses=responses
    )

    aegis = AEGIS(
        client,
        tool_audit_store=store,
    )

    return aegis, store


def test_aegis_exposes_tool_audit_store(
    tmp_path,
):
    aegis, store = make_aegis(
        tmp_path,
        [
            make_response()
        ],
    )

    try:
        assert (
            aegis.tool_audit
            is store
        )

    finally:
        store.close()


def test_aegis_exposes_tool_security_analyzer(
    tmp_path,
):
    aegis, store = make_aegis(
        tmp_path,
        [
            make_response()
        ],
    )

    try:
        assert isinstance(
            aegis.tool_security,
            ToolAuditSecurityAnalyzer,
        )

    finally:
        store.close()


def test_aegis_security_analyzer_sees_tool_execution(
    tmp_path,
):
    aegis, store = make_aegis(
        tmp_path,
        [
            make_response(
                make_tool_use(
                    name="system_info"
                ),
                stop_reason="tool_use",
            ),
            make_response(
                stop_reason="end_turn"
            ),
        ],
    )

    try:
        aegis.respond(
            "Get system information."
        )

        records = (
            aegis.tool_security
            ._analyzer
            .records()
        )

        assert len(records) == 1

        assert (
            records[0].tool_name
            == "system_info"
        )

    finally:
        store.close()


def test_aegis_security_analysis_detects_high_risk_activity(
    tmp_path,
):
    aegis, store = make_aegis(
        tmp_path,
        [
            make_response(
                make_tool_use(
                    name="run_command",
                    input_data={
                        "command": "echo test"
                    },
                ),
                stop_reason="tool_use",
            ),
            make_response(
                stop_reason="end_turn"
            ),
        ],
    )

    try:
        aegis.respond(
            "Run a command."
        )

        findings = (
            aegis.tool_security
            .high_risk_activity()
        )

        assert len(findings) == 1

        assert (
            findings[0].tool_name
            == "run_command"
        )

        assert (
            findings[0].finding_type
            == "HIGH_RISK_ACTIVITY"
        )

    finally:
        store.close()


def test_aegis_security_analysis_is_read_only(
    tmp_path,
):
    aegis, store = make_aegis(
        tmp_path,
        [
            make_response(
                make_tool_use(
                    name="system_info"
                ),
                stop_reason="tool_use",
            ),
            make_response(
                stop_reason="end_turn"
            ),
        ],
    )

    try:
        aegis.respond(
            "Get system information."
        )

        before = store.list()

        findings = (
            aegis.tool_security
            .findings()
        )

        after = store.list()

        assert before == after

        assert isinstance(
            findings,
            list,
        )

    finally:
        store.close()