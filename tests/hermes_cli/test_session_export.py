"""Tests for ``hermes_cli.session_export`` — prompt-only and Markdown renderers."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from hermes_cli.session_export import (
    filter_user_prompts,
    render_full_session_md,
    render_prompt_only_jsonl,
    render_prompt_only_md,
)

_FAKE_NOW = 1800000000.0  # Arbitrary stable timestamp for deterministic tests.


def _make_session(
    session_id: str = "test-session-1",
    messages: list[dict] | None = None,
) -> dict:
    return {
        "id": session_id,
        "title": "Test session",
        "source": "cli",
        "created_at": _FAKE_NOW,
        "updated_at": _FAKE_NOW,
        "messages": messages or [],
    }


def _user_msg(
    content: str,
    ts: float = _FAKE_NOW,
    msg_id: int = 1,
    platform_id: str | None = "pl-1",
) -> dict:
    return {
        "id": msg_id,
        "role": "user",
        "content": content,
        "timestamp": ts,
        "platform_message_id": platform_id,
    }


def _assistant_msg(
    content: str,
    ts: float = _FAKE_NOW,
    msg_id: int = 999,
) -> dict:
    return {
        "id": msg_id,
        "role": "assistant",
        "content": content,
        "timestamp": ts,
    }


def _tool_msg(
    content: str,
    ts: float = _FAKE_NOW,
    msg_id: int = 888,
) -> dict:
    return {
        "id": msg_id,
        "role": "tool",
        "content": content,
        "timestamp": ts,
    }


def _system_msg(
    content: str,
    ts: float = _FAKE_NOW,
    msg_id: int = 777,
) -> dict:
    return {
        "id": msg_id,
        "role": "system",
        "content": content,
        "timestamp": ts,
    }


# ── filter_user_prompts ──────────────────────────────────────────────


class TestFilterUserPrompts:
    def test_empty_session(self):
        assert filter_user_prompts(_make_session()) == []

    def test_only_user_messages(self):
        msgs = [_user_msg("hello"), _user_msg("world")]
        result = filter_user_prompts(_make_session(messages=msgs))
        assert len(result) == 2
        assert all(m["role"] == "user" for m in result)

    def test_filters_out_non_user_roles(self):
        msgs = [
            _user_msg("hi"),
            _assistant_msg("response"),
            _user_msg("follow-up"),
            _tool_msg('{"result": "ok"}'),
            _system_msg("system prompt"),
        ]
        result = filter_user_prompts(_make_session(messages=msgs))
        assert len(result) == 2
        assert [m["content"] for m in result] == ["hi", "follow-up"]

    def test_no_messages_key(self):
        session = {"id": "s1"}
        assert filter_user_prompts(session) == []


# ── render_prompt_only_jsonl ──────────────────────────────────────────


class TestRenderPromptOnlyJsonl:
    def test_empty_session(self):
        result = render_prompt_only_jsonl(_make_session())
        assert result == ""

    def test_single_prompt(self):
        msgs = [_user_msg("What is AI?", msg_id=1)]
        result = render_prompt_only_jsonl(_make_session(messages=msgs))
        records = [json.loads(line) for line in result.strip().split("\n")]
        assert len(records) == 1
        rec = records[0]
        assert rec["session_id"] == "test-session-1"
        assert rec["prompt_index"] == 1
        assert rec["prompt"] == "What is AI?"
        assert rec["message_id"] == 1
        assert rec["platform_message_id"] == "pl-1"

    def test_multiple_prompts_sequential_index(self):
        msgs = [
            _user_msg("first", msg_id=1),
            _assistant_msg("ok"),
            _user_msg("second", msg_id=2),
            _user_msg("third", msg_id=3),
        ]
        result = render_prompt_only_jsonl(_make_session(messages=msgs))
        records = [json.loads(line) for line in result.strip().split("\n")]
        assert len(records) == 3
        assert [r["prompt_index"] for r in records] == [1, 2, 3]
        assert [r["prompt"] for r in records] == ["first", "second", "third"]

    def test_timestamp_iso_format(self):
        ts = 1800000000.0
        expected_iso = (
            datetime.fromtimestamp(ts, tz=timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        msgs = [_user_msg("prompt", ts=ts)]
        result = render_prompt_only_jsonl(_make_session(messages=msgs))
        rec = json.loads(result.strip())
        assert rec["timestamp"] == expected_iso

    def test_jsonl_trailing_newline(self):
        msgs = [_user_msg("a"), _user_msg("b")]
        result = render_prompt_only_jsonl(_make_session(messages=msgs))
        assert result.endswith("\n")
        assert result.count("\n") == 2  # two lines + trailing

    def test_null_fields_for_missing_data(self):
        msgs = [{"id": None, "role": "user", "content": "test", "timestamp": 0}]
        result = render_prompt_only_jsonl(_make_session(messages=msgs))
        rec = json.loads(result.strip())
        assert rec["message_id"] is None
        assert rec["platform_message_id"] is None


# ── render_prompt_only_md ──────────────────────────────────────────


class TestRenderPromptOnlyMd:
    def test_empty_session(self):
        result = render_prompt_only_md(_make_session())
        assert result.startswith("# User prompts for session test-session-1")

    def test_single_prompt(self):
        msgs = [_user_msg("Hello world", msg_id=1)]
        result = render_prompt_only_md(_make_session(messages=msgs))
        assert "## 1." in result
        assert "Hello world" in result

    def test_multiple_prompts(self):
        msgs = [_user_msg("first", msg_id=1), _user_msg("second", msg_id=2)]
        result = render_prompt_only_md(_make_session(messages=msgs))
        assert "## 1." in result
        assert "## 2." in result
        assert "first" in result
        assert "second" in result

    def test_excludes_non_user_messages(self):
        msgs = [
            _user_msg("user text"),
            _assistant_msg("assistant text"),
            _tool_msg("tool output"),
        ]
        result = render_prompt_only_md(_make_session(messages=msgs))
        assert "user text" in result
        assert "assistant text" not in result
        assert "tool output" not in result


# ── render_full_session_md ──────────────────────────────────────────


class TestRenderFullSessionMd:
    def test_empty_session(self):
        result = render_full_session_md(_make_session())
        assert result.startswith("# Session: test-session-1")

    def test_shows_all_roles(self):
        msgs = [
            _user_msg("hello"),
            _assistant_msg("hi there"),
            _tool_msg("result: ok"),
        ]
        result = render_full_session_md(_make_session(messages=msgs))
        assert "hello" in result
        assert "hi there" in result
        assert "result: ok" in result
        assert "**User**" in result
        assert "**Assistant**" in result
        assert "**Tool**" in result

    def test_labels_role_properly(self):
        msgs = [_user_msg("test")]
        result = render_full_session_md(_make_session(messages=msgs))
        assert "User" in result

    def test_handles_tool_calls_field(self):
        msgs = [
            {
                "id": 1,
                "role": "assistant",
                "content": "Let me search",
                "timestamp": _FAKE_NOW,
                "tool_calls": [{"name": "web_search", "args": {"q": "test"}}],
            }
        ]
        result = render_full_session_md(_make_session(messages=msgs))
        assert "tool_calls" in result
        assert "web_search" in result
