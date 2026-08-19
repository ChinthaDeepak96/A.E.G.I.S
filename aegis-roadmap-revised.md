# A.E.G.I.S. — Revised Development Roadmap

*(Replaces §34 Development Strategy, §35–40 Version Roadmap, and §64 Recommended Immediate Development Sequence. One canonical sequence — stop tracking Stage/Version/Phase as three separate lists.)*

## What changed and why

1. **Guardian is split into Guardian Lite and Guardian Full**, and Lite moves up to sit in front of the very first capability that can act on the world (Tools/Computer Agent), not four versions after it. The original sequence let the Computer Agent (v0.4) run keyboard/mouse/terminal/process control for three full version bumps before any risk classifier or confirmation gate existed — which directly contradicts the doc's own stated principle ("The Guardian should exist before serious autonomy").
2. **Voice moves from v0.2 to after the core reasoning loop is stable.** Voice is a transport layer — it doesn't unlock new reasoning or action capability, and debugging tool-selection through an STT/TTS round-trip is slower than debugging it through text logs. Text-first gets you to a working v1.0 loop faster; voice bolts on once that loop is boring and reliable.
3. **Every version now has a "Done when" line.** Previously only v1.0 (§60) had a concrete success definition; v0.1–v0.9 were capability lists with no clear finish condition.

---

## Canonical sequence

| # | Version | Builds | Done when |
|---|---------|--------|-----------|
| 1 | **v0.1 — Core Brain** | Text conversation, LLM integration, basic personality, command handling, base architecture skeleton | MAX can hold a multi-turn text conversation and respond in-character with no crashes across a 20-turn session |
| 2 | **v0.2 — Tools + Guardian Lite** | Tool registry (browser, files, terminal, system info) with a **risk_category** field per tool (LOW/MEDIUM/HIGH, hardcoded — no ML classifier yet); Guardian Lite = lookup table that auto-executes LOW, blocks-and-asks for MEDIUM/HIGH | MAX can list its available tools, pick the right one for a stated goal, and Guardian Lite correctly blocks a MEDIUM-risk tool call without needing a code change |
| 3 | **v0.3 — Computer Agent** | Open/close applications, read screen, keyboard/mouse, process management, file workflows — **all routed through Guardian Lite from day one** | MAX can open a named application and report success/failure, with every action appearing in a (simple, unstructured-log-is-fine) action trace |
| 4 | **v0.4 — Persistent Memory** | Working memory, long-term memory, semantic retrieval, personal preferences, project memory, basic memory management commands | MAX correctly recalls a fact stated 10+ turns earlier in a *previous* session, not just the current context window |
| 5 | **v0.5 — Voice** | Wake word, speech-to-text, text-to-speech, conversation loop (built on the now-stable text loop from v0.1–0.4) | A full v0.1–v0.4 interaction (ask → tool use → memory recall → response) works end-to-end by voice, not just text |
| 6 | **v0.6 — Agent System** | Computer Agent, Research Agent, Coding Agent, Communication Agent, Agent Orchestrator | Orchestrator correctly routes at least 3 distinct request types to the correct specialist agent without hardcoded if/else routing |
| 7 | **v0.7 — Guardian Full** | Upgrades Guardian Lite → real risk classifier, audit logs, permission management, secrets management, anomaly detection | Every MEDIUM/HIGH action produces a queryable audit record (timestamp, agent, action, risk, decision, outcome); secrets are never visible in a prompt or log |
| 8 | **v0.8 — Proactive Intelligence** | Event bus, schedules, triggers, notifications, background context, suggested actions | MAX initiates at least one useful unprompted notification (e.g. scheduled task reminder) during normal background operation, correctly gated by Guardian |
| 9 | **v0.9 — Vision** | Camera input, image understanding, OCR, object recognition, scene understanding | MAX can describe screen contents or a camera frame accurately enough to answer a follow-up question about it |
| 10 | **v1.0 — Personal A.E.G.I.S.** | Integration milestone: voice + memory + reasoning + tools + agents + autonomy-lite + Guardian + vision + context + computer control | *(unchanged from §60)* — MAX can hear, understand, remember, reason, select a tool, check permission, execute, observe, self-correct on simple failure, explain, and produce an audit record — in one pipeline, unattended |
| 11 | **v1.5 — World Model** | People, places, devices, projects, applications, objects, events, relationships, current state | World Model correctly answers a relationship query ("who is in the office right now") using only fused entity state, not a fresh LLM guess |
| 12 | **v2.0 — Sensor Fusion** | Vision + audio + GPS + device state + sensors → unified fused state | A fusion output correctly resolves at least one case where two individual sensors disagree (e.g. GPS says "home", camera says "office") |
| 13 | **v3.0 — Multi-Device** | PC, Mac, iPhone, Raspberry Pi, ESP32 as interchangeable "bodies" of one MAX identity | A task started on one device (e.g. PC) can be hardened/continued on a second device using shared memory/context, not a fresh session |
| 14 | **v4.0 — CARLA Autonomy** | Perception, sensor fusion, planning, Vehicle Agent, simulation | Vehicle Agent completes a defined CARLA scenario with Guardian correctly intervening at least once on a simulated risky action |
| 15 | **v5.0 — Physical A.E.G.I.S.** | Robotics, vehicles, embedded systems, wearables, real-world sensors, physical autonomy — treated as its own engineering domain with independent safety review | A physical action passes through Guardian **and** an independently-validated Safety Controller (per §47) before actuation, with a human override that does not depend on the same process |

---

## Guardian Lite vs. Guardian Full — what's actually different

| | Guardian Lite (v0.2+) | Guardian Full (v0.7+) |
|---|---|---|
| Risk source | Hardcoded per-tool `risk_category` field | Dynamic risk classifier (rules + model-assisted) |
| Decision | Lookup + confirm-or-deny | Lookup + confirm-or-deny + policy engine |
| Logging | None required | Structured, queryable audit trail |
| Secrets | N/A (no tools need them yet) | Dedicated secret storage, never in prompts |
| Anomaly detection | None | Present |
| Build cost | ~1 day | Ongoing subsystem |

The point of Lite isn't to be a lesser version of Full — it's to guarantee that **no version of A.E.G.I.S. ever exists where an action-capable agent runs with zero gate in front of it.** Full replaces Lite's lookup table with a real classifier later; it doesn't introduce gating for the first time.

---

## Open question worth deciding before v0.2

Should risk classification for anything above LOW be **rule-based/allowlisted only**, or can it ever be model-inferred? If Guardian's own judgment of "is this action risky" is itself an LLM call, then the thing enforcing your safety boundary can misjudge the same way the Planner can. Recommend: keep risk classification allowlist-driven through at least v1.0, and only consider model-assisted risk scoring once Guardian Full has an audit trail mature enough to catch it being wrong.
