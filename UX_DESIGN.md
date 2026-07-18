# UX/UI Design — Kairos

A coded reference mockup lives at `design-mockup.html` — open it in a browser for the real thing. This doc explains the *why* behind it so agents building the real frontend stay consistent instead of drifting back to generic SaaS defaults.

---

## 1. Design Direction

**The subject is photography and darkroom editing — so that's where the visual language comes from, not generic "AI app" UI.**

Before automated tools existed, photo editors picked the best frame from a roll of film using a **contact sheet** — a grid of every exposure printed small, reviewed under a loupe, with the chosen frame circled in red grease pencil. That's a near-perfect metaphor for exactly what this tool automates. The entire interface is built around that idea instead of a generic upload-box-and-spinner SaaS flow.

**Explicitly avoided** (per current AI-generated-design defaults): warm cream + terracotta serif combo, near-black + single neon accent with no other point of view, and generic broadsheet/hairline-rule layouts used regardless of subject. If any screen starts looking like a template, that's the signal to revisit this doc, not ship it.

## 2. Design Tokens

**Color** — a darkroom palette: dim, warm-dark surfaces (like working under a safelight) with a genuine grease-pencil red as the signature accent.

| Token | Hex | Use |
|---|---|---|
| `--bg` | `#14100E` | Base background — warm near-black, not pure black |
| `--surface` | `#1F1A17` | Cards, panels, the "contact sheet" backing |
| `--paper` | `#F2ECE1` | Contact sheet strip, result frame mattes — literal photo paper |
| `--ink` | `#EFE7DA` | Primary text on dark surfaces |
| `--ink-dim` | `#9A8F82` | Secondary text, captions, EXIF-style data |
| `--grease-red` | `#C1272D` | Signature accent — the "circled pick" mark, primary CTA |
| `--negative-teal` | `#3FA7A0` | Secondary accent — AI verification / "system checking" state only |

**Typography** — one type family (IBM Plex), three roles, deliberately:

| Role | Face | Use |
|---|---|---|
| Display | IBM Plex Sans Condensed, Bold | Headlines — tight tracking, feels stamped/technical |
| Body | IBM Plex Sans, Regular/Medium | All reading text |
| Data | IBM Plex Mono | Camera-style readouts: confidence scores, "f/2.8 · ISO 400" flourishes, gate status |

Using one family across three roles is itself the choice — it reads as a considered system, not a random display+body pairing.

**Signature element:** the *contact sheet scoring animation* — uploaded photos lay out as a film-strip grid; as scoring runs, a hand-drawn-style red circle animates onto the winning face for each person, frame by frame, before the final composite "develops" in. This is the one moment worth spending all the visual boldness on — everything else stays quiet and disciplined around it.

## 3. Screen-by-Screen

### 3.1 Landing / Upload
Not a hero-stats-CTA template. Opens directly on a dark "lightbox" — a drop zone styled like a photo light table, with a mono-type readout ticking like a camera metering ("READY · AWAITING BURST"). This *is* the product, immediately.

### 3.2 Processing — The Contact Sheet
Uploaded photos arrange into a contact-sheet grid on paper-colored backing. Per-frame mono readout shows live scoring (`EYES 0.92 · SMILE 0.81`). As each person's winning frame is determined, a red grease-pencil circle draws itself around that face. This is the highest-value screen for the live demo — it visibly shows the "thinking," which is exactly what makes AI tools feel trustworthy rather than a black box.

### 3.3 AI Verification Readout
A distinct instrument-panel style strip (mono type, teal accent) showing Gate 1 and Gate 2 results as they resolve — "GATE 1 · INPUT ✓", "GATE 2 · OUTPUT — RETRYING (1/2)". This makes the self-correction loop visible instead of hidden, which is core to the pitch's technical story.

### 3.4 Reveal — Before / After
A single frame (the fallback/base photo) with a drag slider or crossfade to the composite. Paper matte border, like holding a physical print. Deliberately slow, satisfying reveal — this is the "wow" beat.

### 3.5 Result / Share
Final composite on paper matte, download action, and a small honest caption of what happened (e.g., "4 of 4 faces optimized" or, if fallback occurred, "showing original frame — composite didn't meet our quality bar," which is a *feature* to say out loud, not hide).

## 4. Copy Voice

- Plain, active, specific — "Everyone's eyes are open" not "Optimal frame selection achieved."
- Camera/darkroom vocabulary where it's genuinely clarifying (frame, exposure, develop) — never cute for its own sake.
- Failure states explain what happened in the system's own technical voice, not an apology: "Gate 2 flagged a lighting mismatch on frame 3 — retrying with an alternate frame," not "Oops, something went wrong!"

## 5. Real Design Inspiration (if you want to browse/build your own instead)

Curated galleries worth an hour of scrolling before you commit to a direction — none of these are templates to copy wholesale, they're calibration for what "doesn't look AI-generated" actually looks like in the wild:

- **mobbin.com** — real production app screens, filterable by flow (upload, onboarding, empty states) — best for interaction patterns
- **awwwards.com** — high-craft, often experimental web design — best for typography/motion inspiration
- **land-book.com** — curated landing pages, filterable by style/industry
- **lapa.ninja** — landing page gallery, good filtering by category
- **godly.website** — leans editorial/art-directed, good antidote to generic SaaS look
- **saaspo.com** — SaaS-specific but genuinely curated, not templated
- **onepagelove.com** — single-page product sites — closest format match to this project
- **screenlane.com** — mobile-first UI patterns if you build a companion mobile flow

## 6. Build Notes for Agents

- Colors/type are CSS variables (see `design-mockup.html`) — never hardcode hex values in components.
- Keyboard focus states, reduced-motion support, and mobile responsiveness are required, not optional — see `TESTING.md` for the checklist.
- Motion should be orchestrated (the contact-sheet reveal sequence) rather than scattered micro-animations everywhere — restraint is part of the direction.
