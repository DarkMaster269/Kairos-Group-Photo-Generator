# AGENTS.md

Context file for AI coding agents (Gemini 3.5 Flash, Claude Sonnet 4.6, Opus 4.6) working on this project in Google Antigravity. Read this before starting any task. This file takes precedence over any single tool's own config file.

---

## Project Summary

**Kairos** — a hackathon submission for the "Perfect Group Photo Generator" track. Takes a burst of photos of the same group and composites the best-expression face per person into one seamless photo, using a dual AI-verification-gate architecture to catch and self-correct bad blends before showing the user.

Full specs live in:
- `PRD.md` — what we're building and why
- `TRD.md` — architecture, data contracts, pipeline behavior (binding — don't deviate without updating this file)
- `UX_DESIGN.md` — visual direction and design tokens
- `IMPLEMENTATION_PLAN.md` — the day-by-day build order
- `TESTING.md` — testing strategy and QA checklist

**Read the relevant doc before implementing that layer.** Don't infer pipeline behavior from first principles when `TRD.md` already specifies it.

## Tech Stack (do not substitute without discussion)

- Frontend: React + Vite + Tailwind
- Backend: Python + FastAPI
- Face detection: MediaPipe Face Mesh
- Face clustering/identity: `face_recognition` (dlib) or InsightFace
- Blending: OpenCV `seamlessClone`
- AI Gates: Claude Sonnet 4.6 or Gemini 3.5 Flash via API, multimodal, structured JSON output only
- Deployment: Vercel (frontend), Render/Fly.io (backend)

## Coding Standards

- Python: PEP 8, type hints on all function signatures, docstrings on public functions
- No magic numbers — expression-scoring thresholds, retry limits, etc. are named constants in a config module, not inline literals
- All API keys/secrets via environment variables — never hardcoded, never committed
- Every pipeline stage (detect → cluster → score → blend → gate) should be independently callable/testable, not one monolithic function — this project's timeline depends on being able to debug each stage in isolation
- Frontend: colors/type come from the CSS variables defined in `UX_DESIGN.md` / `design-mockup.html` — never hardcode hex values in components
- Commit messages: short, present tense, describe the "why" if not obvious from the diff

## Agent Task Allocation (see `IMPLEMENTATION_PLAN.md` for full detail)

- **Opus 4.6** — hardest reasoning tasks: face clustering logic, blend quality tuning, AI gate prompt design, subtle bug debugging. Use where correctness matters more than speed.
- **Claude Sonnet 4.6** — core pipeline and backend implementation, state machine logic, API wiring. Default choice for sustained "real" implementation work.
- **Gemini 3.5 Flash (High)** — well-specified, high-volume work: UI scaffolding from `UX_DESIGN.md`, boilerplate, test generation. Use when the spec is already unambiguous and speed matters more than deep judgment.

## Parallel Agent Rules (important — Antigravity has no merge conflict resolution)

When running multiple agents concurrently, **scope each to a non-overlapping file/directory set** and state that scope explicitly in the task prompt, e.g.:
- "Agent A: work only in `/backend/pipeline/`"
- "Agent B: work only in `/frontend/src/components/`"
- "Agent C: work only in `/backend/tests/`"

Never let two agents write to the same file in the same session — the last write silently wins and the other agent's work is lost. If a task genuinely needs to touch shared files (e.g., a shared types/schema file), do that step serially with one agent, then fan out.

## Non-Negotiables (from `PRD.md` — don't silently drop these under time pressure)

- The user must never be shown a blend that Gate 2 flagged as unnatural — always retry or fall back, never expose a broken composite.
- No face-swapping across different occasions/photos — only recombining frames from the same uploaded burst.
- Gate 2 (vision LLM) detects problems; it does not directly edit pixels. Don't build "repair" logic that assumes the LLM can regenerate image regions unless explicitly scoped as the Day 4 stretch goal in `TRD.md` §5.

## When Specs Conflict

If `PRD.md`/`TRD.md`/`UX_DESIGN.md` conflict with each other, or with a task prompt, flag it explicitly rather than guessing — these documents were written together but hackathon timelines mean they may drift. Prefer `TRD.md` for technical/architecture questions and `UX_DESIGN.md` for anything visual.

## Testing Expectations

See `TESTING.md` for full strategy. Minimum bar before marking any pipeline stage "done": run against at least one real (not synthetic) test burst and visually confirm correctness — this project's core risk is CV quality that looks fine in code review but wrong in an image.
