# A.E.G.I.S. v0.3 — Core Brain + Tools + Guardian Lite + Computer Agent

Text conversation, LLM integration, AEGIS's personality, basic command
handling (v0.1), a tool registry gated by Guardian Lite (v0.2), and now
the Computer Agent (v0.3): process listing, launching/closing
applications, and direct keyboard/mouse control. Still no persistent
memory beyond the session, no full Guardian, no real visual screen
reading, no voice — those arrive in later versions, per the project
roadmap.

## What this is (and isn't)

**Is:** a conversation loop that talks to AEGIS (backed by either Claude
or a free local model — your choice), holds context across a session,
and can list/read files, check system info, or run a shell command —
with every tool call reviewed by Guardian Lite first.

**Isn't:** anything with memory across sessions, full audit logging,
a real risk classifier, or actual visual screen reading (that's OCR/
vision, v0.9 scope — `list_open_windows` just returns window titles).
Guardian Lite is deliberately a hardcoded lookup table (risk category
→ auto / confirm), not the real Guardian that arrives in v0.7.

## How Guardian Lite works

Every tool declares a `risk_category` (`core/tools.py`):

| Risk | Tools | Behavior |
|---|---|---|
| LOW | `list_files`, `read_file`, `system_info`, `list_processes`, `list_open_windows`, `open_application` | Executes automatically |
| MEDIUM | `close_application` | Blocked unless a `confirm` callback approves it |
| HIGH | `run_command`, `type_text`, `click_mouse` | Blocked unless a `confirm` callback approves it |

In the CLI, `confirm_action()` in `apps/cli.py` is the only place a
HIGH-risk call can be approved — it prints what AEGIS wants to run and
waits for an explicit `y`. If you run AEGIS programmatically (e.g. in
tests) without passing a `confirm` callback, HIGH-risk calls are
denied by default — Guardian fails closed, not open.

## Two ways to run AEGIS

| | Anthropic API | Local (Ollama) |
|---|---|---|
| Cost | Pay-per-token | Free, always |
| Setup | API key from console.anthropic.com | Install [Ollama](https://ollama.com), pull a model |
| Quality | Best — Claude Sonnet | Depends on your hardware and chosen model |
| Tool-calling reliability | Strong | Varies by model (llama3.1, qwen2.5 recommended) |

### Option A: Anthropic API (paid, best quality)

```bash
cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY
python -m apps.cli
```

### Option B: Local model via Ollama (free, no account needed)

```bash
# 1. Install Ollama: https://ollama.com/download
# 2. Pull a model (llama3.1 handles tool-calling reasonably well)
ollama pull llama3.1

# 3. Set the provider and run — no .env, no API key, needed at all
export AEGIS_PROVIDER=local
python -m apps.cli
```

Optional env vars for the local path: `AEGIS_OLLAMA_MODEL` (default
`llama3.1`), `AEGIS_OLLAMA_HOST` (default `http://localhost:11434`).

This is the first real piece of the future Model Router (architecture
doc section 51) — `core/llm_client.py` is still the only file that
knows how to talk to any provider; `core/aegis.py` doesn't know or
care which one is behind `self._llm`.

## Test

```bash
pytest
```

Tests run against a `MockClient` (see `core/llm_client.py`) — no API
key or network access required to verify the conversation loop,
history trimming, and command handling all work correctly.

## Definition of done for this version

v0.2's done-when (AEGIS lists tools, picks the right one, Guardian
blocks MEDIUM/HIGH risk without a code change) still holds. v0.3 adds:
AEGIS can open a named application, list running processes and open
windows, and Guardian correctly blocks `close_application`, `type_text`,
and `click_mouse` without a confirm callback. Covered by
`tests/test_computer_agent.py` — process listing and termination are
tested against real processes (psutil needs no display); keyboard/mouse
and window listing are tested by injecting a fake module in place of
pyautogui/pygetwindow, since this environment (and likely your CI) has
no real screen to test against. **You should still do a quick manual
smoke test on your own machine** — see below.

## Project layout

```
aegis/
├── core/
│   ├── config.py       # settings from env / .env
│   ├── llm_client.py   # talks to a provider — Anthropic API or local Ollama
│   ├── aegis.py         # personality, conversation state, tool loop
│   ├── tools.py         # tool registry (name, risk_category, handler; now incl. Computer Agent)
│   └── guardian.py      # Guardian Lite: risk_category -> ALLOW / CONFIRM
├── apps/
│   └── cli.py            # text entrypoint + the human behind CONFIRM
└── tests/
    ├── test_aegis.py
    ├── test_guardian.py
    ├── test_tools.py
    ├── test_ollama_client.py
    └── test_computer_agent.py
```

Deliberately not building out the full repository structure from the
architecture doc (agents/, memory/, devices/, etc.) yet — those
directories get created when the version that needs them actually
gets built, not before.

## Known limitation

`AEGIS._trim_history()` bounds context by message count, not by
complete conversational turns. With the default `history_limit` of
40 this is a non-issue, but a very low limit combined with an
in-progress multi-step tool exchange could in principle split a
`tool_use` from its matching `tool_result`. Proper turn-aware memory
management is v0.4 scope, not v0.2.

## Manual smoke test for the Computer Agent (do this once)

The automated tests can't exercise a real screen, so before you trust
this on your machine, try each of these once through the CLI and
confirm it does what you expect:

1. Ask it to list running processes — should run with no prompt.
2. Ask it to open a real application by name (e.g. "open notepad" on
   Windows) — should run with no prompt, and the app should actually
   open.
3. Ask it to close that application — Guardian should ask you to
   confirm first.
4. Ask it to type something — Guardian should ask first, and it types
   into whatever window currently has focus, so click into a scratch
   text editor before approving.

If `list_open_windows`, `type_text`, or `click_mouse` report an error
instead of doing anything, that's `pyautogui`/`pygetwindow` failing to
find a display or the right OS backend — check that you're running
this directly on your machine (not over SSH without X forwarding, not
in a headless container).

## What's next (v0.4)

**Persistent Memory.** Working memory, long-term memory, semantic
retrieval, personal preferences and project memory that survive
between sessions — AEGIS currently forgets everything the moment you
close it. This is also where `_trim_history()`'s known limitation
above gets a real, turn-aware fix instead of a documented caveat.
