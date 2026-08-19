# A.E.G.I.S. v0.2 — Core Brain + Tools + Guardian Lite

Text conversation, LLM integration, AEGIS's personality, basic command
handling (v0.1), plus a small tool registry gated by Guardian Lite
(v0.2). Still no persistent memory beyond the session, no full
Guardian, no Computer Agent, no voice — those arrive in later
versions, per the project roadmap.

## What this is (and isn't)

**Is:** a conversation loop that talks to AEGIS (backed by either Claude
or a free local model — your choice), holds context across a session,
and can list/read files, check system info, or run a shell command —
with every tool call reviewed by Guardian Lite first.

**Isn't:** anything with memory across sessions, full audit logging,
a real risk classifier, or open-ended computer control. Guardian
Lite is deliberately a hardcoded lookup table (risk category → auto
/ confirm), not the real Guardian that arrives in v0.7.

## How Guardian Lite works

Every tool declares a `risk_category` (`core/tools.py`):

| Risk | Tools | Behavior |
|---|---|---|
| LOW | `list_files`, `read_file`, `system_info` | Executes automatically |
| HIGH | `run_command` | Blocked unless a `confirm` callback approves it |

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

Per the roadmap: AEGIS can list its available tools, pick the right
one for a stated goal, and Guardian correctly blocks a MEDIUM/HIGH
risk tool call without needing a code change to do it. Covered by
`tests/test_aegis.py`'s v0.2 section, plus `tests/test_guardian.py`
and `tests/test_tools.py`. v0.1's original done-when (20-turn
conversation, no crashes) still holds and is still tested.

## Project layout

```
aegis/
├── core/
│   ├── config.py       # settings from env / .env
│   ├── llm_client.py   # talks to a provider — Anthropic API or local Ollama
│   ├── aegis.py         # personality, conversation state, tool loop
│   ├── tools.py         # tool registry (name, risk_category, handler)
│   └── guardian.py      # Guardian Lite: risk_category -> ALLOW / CONFIRM
├── apps/
│   └── cli.py            # text entrypoint + the human behind CONFIRM
└── tests/
    ├── test_aegis.py
    ├── test_guardian.py
    ├── test_tools.py
    └── test_ollama_client.py
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

## What's next (v0.3)

**Computer Agent.** Open/close applications, read the screen,
keyboard/mouse, process management — all routed through Guardian
Lite from day one, since that gate already exists. This is the first
version where "the AI can act on your machine" becomes literally
true, so it's also the version where Guardian actually earns its
keep instead of just gating file reads.
