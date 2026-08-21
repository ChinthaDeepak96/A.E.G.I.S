from core.aegis import AEGIS
from core.llm_client import (
    LLMResponse,
    MockClient,
    ToolUseBlock,
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
            / "tool_audit.db"
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


# =========================================================
# AUDIT STORE INTEGRATION
# =========================================================


def test_aegis_creates_audit_store(
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
            aegis._tool_audit_store
            is store
        )

        assert isinstance(
            aegis._tool_audit_store,
            SQLiteToolAuditStore,
        )

    finally:
        store.close()


def test_aegis_gateway_uses_audit_store(
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
            aegis._tool_gateway._audit_store
            is store
        )

    finally:
        store.close()


# =========================================================
# REAL TOOL EXECUTION
# =========================================================


def test_aegis_tool_execution_creates_audit_record(
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
        response = aegis.respond(
            "Get system information."
        )

        assert response is not None

        records = store.list()

        assert len(records) == 1

        record = records[0]

        assert (
            record.tool_name
            == "system_info"
        )

        assert (
            record.executed
            is True
        )

        assert (
            record.success
            is True
        )

    finally:
        store.close()


# =========================================================
# UNKNOWN TOOL
# =========================================================


def test_aegis_unknown_tool_does_not_create_execution_audit(
    tmp_path,
):
    aegis, store = make_aegis(
        tmp_path,
        [
            make_response(
                make_tool_use(
                    name="does_not_exist"
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
            "Use the unknown tool."
        )

        # Unknown tools are rejected before reaching
        # ToolGateway, so there must be no audit record.
        records = store.list()

        assert records == []

        # The error is returned as a tool_result message
        # to the LLM rather than directly as the final
        # AEGIS response.
        assert any(
            "unknown tool"
            in str(message).lower()
            for message in aegis._history
        )

    finally:
        store.close()


# =========================================================
# MULTIPLE TOOL CALLS
# =========================================================


def test_aegis_multiple_tool_calls_create_multiple_audits(
    tmp_path,
):
    aegis, store = make_aegis(
        tmp_path,
        [
            make_response(
                make_tool_use(
                    tool_id="tool-1",
                    name="system_info",
                ),
                make_tool_use(
                    tool_id="tool-2",
                    name="list_processes",
                    input_data={
                        "limit": 5
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
            "Check the system."
        )

        records = store.list()

        assert len(records) == 2

        tool_names = {
            record.tool_name
            for record in records
        }

        assert tool_names == {
            "system_info",
            "list_processes",
        }

    finally:
        store.close()


# =========================================================
# AUDIT EVIDENCE
# =========================================================


def test_aegis_audit_contains_tool_arguments(
    tmp_path,
):
    aegis, store = make_aegis(
        tmp_path,
        [
            make_response(
                make_tool_use(
                    name="list_processes",
                    input_data={
                        "limit": 7
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
            "Show processes."
        )

        records = store.list()

        assert len(records) == 1

        record = records[0]

        assert (
            record.tool_name
            == "list_processes"
        )

        assert (
            record.arguments
            == {
                "limit": 7
            }
        )

    finally:
        store.close()


# =========================================================
# AUDIT PERSISTENCE
# =========================================================


def test_aegis_audit_store_persists_records(
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

    database_path = (
        tmp_path
        / "tool_audit.db"
    )

    try:
        aegis.respond(
            "Get system information."
        )

        records = store.list()

        assert len(records) == 1

        record_id = records[0].id

        store.close()

        reopened = SQLiteToolAuditStore(
            str(database_path)
        )

        try:
            loaded = reopened.get(
                record_id
            )

            assert loaded is not None

            assert (
                loaded.id
                == record_id
            )

            assert (
                loaded.tool_name
                == "system_info"
            )

        finally:
            reopened.close()

    finally:
        # The store may already be closed by
        # the persistence test.
        try:
            store.close()
        except Exception:
            pass