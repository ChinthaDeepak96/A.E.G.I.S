# A.E.G.I.S. v0.1 — Core Brain

The first working slice of the A.E.G.I.S. personal AI platform: text
conversation, LLM integration, MAX's personality, and basic command
handling. No tools, no memory beyond the current session, no
Guardian, no voice yet — those arrive in v0.2 onward, per the
project roadmap.

## What this is (and isn't)

**Is:** a conversation loop you can run right now that talks to
Claude as "MAX" and holds context across a session.

**Isn't:** anything that can act on your computer, browse the web,
remember you tomorrow, or do anything Guardian would need to gate.
Those get layered on top of this loop deliberately, one version at a
time — see the roadmap for why the sequencing matters (Guardian in
particular arrives *before* the Computer Agent, not after).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY
```

## Run

```bash
python -m apps.cli
```

## Test

```bash
pytest
```

Tests run against a `MockClient` (see `core/llm_client.py`) — no API
key or network access required to verify the conversation loop,
history trimming, and command handling all work correctly.

## Definition of done for this version

Per the roadmap: MAX can hold a multi-turn text conversation and
respond in-character with no crashes across a 20-turn session. That
exact scenario is exercised by
`tests/test_max.py::test_twenty_turn_conversation_survives`.

## Project layout

```
aegis/
├── core/
│   ├── config.py       # settings from env / .env
│   ├── llm_client.py   # the ONLY file that talks to the Anthropic API
│   └── max.py          # personality, conversation state, command handling
├── apps/
│   └── cli.py           # text entrypoint
└── tests/
    └── test_max.py
```

Deliberately not building out the full repository structure from the
architecture doc (agents/, guardian/, memory/, devices/, etc.) yet —
those directories get created when the version that needs them
actually gets built, not before.

## What's next (v0.2)

**Tools + Guardian Lite.** A tool registry where every tool declares
a `risk_category`, and a lookup-table gate that auto-executes
LOW-risk actions and blocks-and-asks for MEDIUM/HIGH — built *before*
any action-capable agent, not after. See the revised roadmap for the
full reasoning.
