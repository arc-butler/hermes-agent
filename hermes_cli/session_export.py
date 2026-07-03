"""Prompt-only and Markdown session export renderers.

``filter_user_prompts`` extracts only user-authored messages from a
session dict.  ``render_prompt_only_jsonl`` and ``render_prompt_only_md``
serialise those prompts as JSONL or Markdown respectively.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def filter_user_prompts(
    session_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return only user-authored messages from *session_data*.

    Each returned message dict includes the original row columns from
    the ``messages`` table (*id*, *session_id*, *role*, *content*,
    *timestamp*, *platform_message_id*, …).
    """
    messages: list[dict[str, Any]] = session_data.get("messages", [])
    return [m for m in messages if m.get("role") == "user"]


def _ts_to_iso(timestamp: float) -> str:
    """Convert a Unix-epoch *timestamp* to ISO-8601 with Z suffix."""
    try:
        return (
            datetime.fromtimestamp(timestamp, tz=timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
    except (OSError, ValueError, OverflowError):
        return ""


def render_prompt_only_jsonl(
    session_data: dict[str, Any],
) -> str:
    """Render user prompts from *session_data* as newline-delimited JSON.

    Each line is a JSON object with::

        {
            "session_id": "<session-id>",
            "prompt_index": <1-based-int>,
            "timestamp": "<ISO-8601>",
            "prompt": "<raw-user-text>",
            "message_id": <int-or-null>,
            "platform_message_id": "<str-or-null>",
            "event_id": "<str-or-null>"
        }
    """
    prompts = filter_user_prompts(session_data)
    lines: list[str] = []
    for idx, msg in enumerate(prompts, start=1):
        record = {
            "session_id": session_data.get("id"),
            "prompt_index": idx,
            "timestamp": _ts_to_iso(msg.get("timestamp", 0)),
            "prompt": msg.get("content") or "",
            "message_id": msg.get("id"),
            "platform_message_id": msg.get("platform_message_id"),
            "event_id": None,  # reserved for future gateway event linkage
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def render_prompt_only_md(
    session_data: dict[str, Any],
) -> str:
    """Render user prompts from *session_data* as Markdown.

    Output format::

        # User prompts for session <id>

        ## 1. <ISO-timestamp>

        <prompt-text>

        ## 2. <ISO-timestamp>

        …
    """
    prompts = filter_user_prompts(session_data)
    sid = session_data.get("id", "unknown")
    lines: list[str] = [f"# User prompts for session {sid}", ""]
    for idx, msg in enumerate(prompts, start=1):
        ts = _ts_to_iso(msg.get("timestamp", 0))
        lines.append(f"## {idx}. {ts}")
        lines.append("")
        lines.append((msg.get("content") or "").strip())
        lines.append("")
    return "\n".join(lines)


def render_full_session_md(
    session_data: dict[str, Any],
) -> str:
    """Render a full session as Markdown (all messages in order).

    Suitable as a shared full-session Markdown renderer that other
    export surfaces can reuse in the future.

    Output::

        # Session: <id>

        **User** <timestamp>

        <content>

        **Assistant** <timestamp>

        <content>

        …
    """
    messages: list[dict[str, Any]] = session_data.get("messages", [])
    sid = session_data.get("id", "unknown")
    lines: list[str] = [f"# Session: {sid}", ""]
    for msg in messages:
        role = msg.get("role", "unknown").capitalize()
        ts = _ts_to_iso(msg.get("timestamp", 0))
        lines.append(f"**{role}** {ts}")
        lines.append("")
        content = (msg.get("content") or "").strip()
        if content:
            lines.append(content)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            lines.append("")
            lines.append(f"  *tool_calls:* `{json.dumps(tool_calls, ensure_ascii=False)}`")
        lines.append("")
    return "\n".join(lines)
