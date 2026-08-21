"""
A.E.G.I.S. Tool Execution Gateway.

The Tool Gateway is the single execution boundary between
Guardian decisions and actual tool handlers.

Responsibilities:

    1. Ask Guardian to review a tool.
    2. Execute LOW-risk tools automatically.
    3. Require explicit confirmation for CONFIRM decisions.
    4. Never execute a tool that has not been authorized.
    5. Pass arguments to the tool handler.
    6. Convert handler exceptions into controlled results.
    7. Persist an audit record for every execution attempt.
    8. Ensure audit failures never break tool execution.

Guardian decides whether execution is allowed.

The Gateway decides whether the handler is actually invoked.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.guardian import (
    ALLOW,
    CONFIRM,
    GuardianDecision,
    review,
)

from core.tool_audit import (
    ToolAuditRecord,
)

from core.tool_security_review import (
    ToolSecurityReviewer,
)

from core.tools import Tool


@dataclass
class ToolExecutionResult:
    """
    Result returned by ToolGateway.execute().
    """

    tool_name: str

    risk_category: str

    decision: str

    executed: bool

    requires_confirmation: bool

    result: str | None = None

    error: str | None = None

    confirmed: bool = False

    success: bool = False


class ToolGateway:
    """
    Controlled execution boundary for A.E.G.I.S. tools.

    Every tool execution must pass through this gateway.

    LOW-risk tools:

        Guardian -> ALLOW -> execute

    MEDIUM/HIGH/unknown-risk tools:

        Guardian -> CONFIRM
                   |
                   +-- confirmed=False -> do not execute
                   |
                   +-- confirmed=True  -> execute

    Audit behavior:

        If an audit store is supplied, every execution decision
        is persisted.

        Audit failures never propagate to the caller.
    """

    def __init__(
        self,
        *,
        audit_store=None,
        security_reviewer=None,
    ):
        """
        Create a ToolGateway.

        Args:
            audit_store:
                Optional persistent audit store.

            security_reviewer:
                Optional read-only security reviewer.

                If supplied, BLOCK decisions from the security
                review prevent tool execution.

                REVIEW and INFO do not block execution.
        """

        self._audit_store = audit_store

        if (
            security_reviewer is not None
            and not isinstance(
                security_reviewer,
                ToolSecurityReviewer,
            )
        ):
            raise TypeError(
                "security_reviewer must be a "
                "ToolSecurityReviewer instance."
            )

        self._security_reviewer = (
            security_reviewer
        )

    # =========================================================
    # PUBLIC EXECUTION
    # =========================================================

    def execute(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        *,
        confirmed: bool = False,
    ) -> ToolExecutionResult:
        """
        Review and, when authorized, execute a tool.

        Args:
            tool:
                Tool definition to execute.

            arguments:
                Keyword arguments passed to the handler.

            confirmed:
                Explicit user confirmation for tools whose
                Guardian decision is CONFIRM.

        Returns:
            ToolExecutionResult describing the outcome.

        Raises:
            TypeError:
                If tool or arguments are invalid.
        """

        self._validate_tool(
            tool
        )

        self._validate_arguments(
            arguments
        )

        security_result = (
            self._security_check(
                tool,
                arguments,
            )
        )

        if security_result is not None:
            return security_result

        decision = review(
            tool
        )

        # -----------------------------------------------------
        # ALLOW
        # -----------------------------------------------------

        if decision.decision == ALLOW:
            return self._execute_handler(
                tool,
                arguments,
                decision,
                confirmed=confirmed,
            )

        # -----------------------------------------------------
        # CONFIRM
        # -----------------------------------------------------

        if decision.decision == CONFIRM:

            if not confirmed:
                execution_result = (
                    ToolExecutionResult(
                        tool_name=tool.name,
                        risk_category=(
                            tool.risk_category
                        ),
                        decision=decision.decision,
                        executed=False,
                        requires_confirmation=True,
                        result=None,
                        error=None,
                        confirmed=False,
                        success=False,
                    )
                )

                self._audit_execution(
                    tool=tool,
                    arguments=arguments,
                    result=execution_result,
                )

                return execution_result

            return self._execute_handler(
                tool,
                arguments,
                decision,
                confirmed=True,
            )

        # -----------------------------------------------------
        # FAIL CLOSED
        # -----------------------------------------------------

        execution_result = (
            ToolExecutionResult(
                tool_name=tool.name,
                risk_category=(
                    tool.risk_category
                ),
                decision=decision.decision,
                executed=False,
                requires_confirmation=True,
                result=None,
                error=(
                    "Guardian returned an "
                    "unrecognized decision."
                ),
                confirmed=confirmed,
                success=False,
            )
        )

        self._audit_execution(
            tool=tool,
            arguments=arguments,
            result=execution_result,
        )

        return execution_result


        # =========================================================
    # SECURITY GATE
    # =========================================================

    def _security_check(
        self,
        tool: Tool,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult | None:
        """
        Evaluate the current security policy before execution.

        Returns:
            ToolExecutionResult when execution must be blocked.
            None when normal Guardian execution may continue.

        Security review is observational and read-only.

        Only BLOCK prevents execution.

        Security blocks are themselves persisted as audit
        records so that every execution attempt remains
        traceable.
        """

        if self._security_reviewer is None:
            return None

        try:
            decisions = (
                self._security_reviewer
                .decisions()
            )

        except Exception:
            # Security infrastructure must never crash
            # the execution gateway.
            #
            # Existing Gateway behavior remains unchanged
            # when security analysis itself fails.
            return None

        for security_decision in decisions:

            if (
                security_decision.action
                != "BLOCK"
            ):
                continue

            if (
                security_decision.tool_name
                != tool.name
            ):
                continue

            execution_result = (
                ToolExecutionResult(
                    tool_name=tool.name,
                    risk_category=(
                        tool.risk_category
                    ),
                    decision="SECURITY_BLOCK",
                    executed=False,
                    requires_confirmation=False,
                    result=None,
                    error=(
                        "Tool execution blocked "
                        "by security policy: "
                        + security_decision.reason
                    ),
                    confirmed=False,
                    success=False,
                )
            )

            self._audit_execution(
                tool=tool,
                arguments=arguments,
                result=execution_result,
            )

            return execution_result

        return None
    
    # =========================================================
    # HANDLER EXECUTION
    # =========================================================

    def _execute_handler(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        decision: GuardianDecision,
        *,
        confirmed: bool = False,
    ) -> ToolExecutionResult:
        """
        Invoke an already-authorized tool handler.

        Handler exceptions are contained inside the gateway.

        A handler returning None is treated as an unsuccessful
        execution because there is no usable tool result.

        For backwards compatibility, a handler exception without
        an audit store retains the original executed=False behavior.

        When an audit store is present, the result reports that the
        handler was actually invoked with executed=True and
        success=False. This matches the audit execution semantics.
        """

        try:
            raw_result = tool.handler(
                **arguments
            )

            # -------------------------------------------------
            # Handler returned None
            # -------------------------------------------------

            if raw_result is None:
                execution_result = (
                    ToolExecutionResult(
                        tool_name=tool.name,
                        risk_category=(
                            tool.risk_category
                        ),
                        decision=decision.decision,
                        executed=False,
                        requires_confirmation=False,
                        result=None,
                        error=None,
                        confirmed=confirmed,
                        success=False,
                    )
                )

            # -------------------------------------------------
            # Handler returned a real result
            # -------------------------------------------------

            else:
                execution_result = (
                    ToolExecutionResult(
                        tool_name=tool.name,
                        risk_category=(
                            tool.risk_category
                        ),
                        decision=decision.decision,
                        executed=True,
                        requires_confirmation=False,
                        result=str(raw_result),
                        error=None,
                        confirmed=confirmed,
                        success=True,
                    )
                )

        except Exception as exc:
            # Preserve the original gateway behavior when no
            # audit store is configured.
            executed = (
                True
                if self._audit_store is not None
                else False
            )

            execution_result = (
                ToolExecutionResult(
                    tool_name=tool.name,
                    risk_category=(
                        tool.risk_category
                    ),
                    decision=decision.decision,
                    executed=executed,
                    requires_confirmation=False,
                    result=None,
                    error=str(exc),
                    confirmed=confirmed,
                    success=False,
                )
            )

        self._audit_execution(
            tool=tool,
            arguments=arguments,
            result=execution_result,
        )

        return execution_result

    # =========================================================
    # AUDIT
    # =========================================================

    def _audit_execution(
        self,
        *,
        tool: Tool,
        arguments: dict[str, Any],
        result: ToolExecutionResult,
    ) -> None:
        """
        Persist a ToolAuditRecord when an audit store exists.

        Audit persistence is deliberately isolated from tool
        execution.

        If the audit store fails, the original execution result
        is still returned unchanged.
        """

        if self._audit_store is None:
            return

        try:
            record = ToolAuditRecord(
                tool_name=tool.name,
                risk_category=(
                    tool.risk_category
                ),
                decision=result.decision,
                confirmed=result.confirmed,
                executed=result.executed,
                success=result.success,
                arguments=dict(arguments),
                result=result.result,
                error=result.error,
                timestamp=datetime.now(
                    timezone.utc
                ),
            )

            self._audit_store.save(
                record
            )

        except Exception:
            # Never allow audit infrastructure to interfere
            # with actual tool execution.
            return

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _validate_tool(
        tool: Tool,
    ) -> None:
        """
        Validate that the supplied object is a Tool.
        """

        if not isinstance(
            tool,
            Tool,
        ):
            raise TypeError(
                "tool must be a Tool instance."
            )

    @staticmethod
    def _validate_arguments(
        arguments: dict[str, Any],
    ) -> None:
        """
        Validate handler arguments.
        """

        if not isinstance(
            arguments,
            dict,
        ):
            raise TypeError(
                "arguments must be a dictionary."
            )