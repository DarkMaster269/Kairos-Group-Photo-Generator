# PRD — Kairos (working title)
### "Perfect Group Photo Generator" — Hackathon Track Submission

> **Kairos** (καιρός): the Greek word for the precise, opportune moment — as opposed to chronological time. Rename freely; used throughout these docs as a placeholder so the project isn't just "the photo app."

---

## 1. Problem Statement

Group photos almost never work on the first try. Someone blinks, someone's mid-word, someone's looking at their phone. Modern phones already shoot a rapid burst per shutter press — the "good" version of every face usually *exists* somewhere in that burst, but nobody has time to manually flip through 10 photos and Photoshop the best face for six different people into one image.

Kairos automates that manual "pick the best frame per person and composite" workflow that a professional photo editor would otherwise do by hand — and adds an AI layer that checks its own work before showing it to the user, rather than silently shipping a bad blend.

## 2. Target Users

- **Primary (hackathon):** Judges evaluating on Idea & Creativity, Scalability & Feasibility, Technical Complexity, Presentation, and Wow Factor (see Section 6).
- **Primary (product):** Anyone taking group photos on a phone — friend groups, family gatherings, event/conference photography, school photos.
- **Secondary:** Event photographers who currently do this manually in Lightroom/Photoshop and would pay for time saved.

## 3. Goals

**Hackathon goals**
- Ship a working, demoable end-to-end pipeline by July 18.
- Make the failure mode *graceful* — a live demo should never show a visibly broken face blend.
- Have a genuine "how did it do that" moment in the live demo.

**Product goals (the pitch)**
- Turn a burst of 5–15 photos of the same group into one composite where every person has their best expression.
- Do this without looking like a face-swap / deepfake — blends must be photorealistic and seamless.
- Self-verify output quality using AI before showing the result, and self-correct rather than exposing failures to the user.

## 4. Non-Goals (explicitly out of scope — per hackathon brief)

- Manual photo editing UI (crop/filter/retouch tools) — this is not a Lightroom clone.
- Naive "pick the single best whole photo from the burst" with no per-face analysis — the whole point is *combining* frames.
- Face-swapping between *different* photos/occasions, or any deepfake-adjacent identity-swap framing. This tool only recombines frames of the **same** burst, same moment, same people.
- Video support (burst photos only, not video frame extraction) — stretch goal at best, not core.
- User accounts / persistent galleries — nice-to-have stretch, not core for the demo.

## 5. User Stories

1. As a user, I upload a burst of photos taken seconds apart of the same group, and I get back one photo where everyone looks their best.
2. As a user, if my uploaded photos aren't actually a valid burst (different scenes, random unrelated images), I'm told clearly instead of getting a broken result.
3. As a user, if the automatic blend has visible artifacts, I never see that broken version — the system retries or falls back to a real, unedited frame instead.
4. As a judge, I can watch the whole pipeline happen live in under ~20 seconds and understand *why* each face was chosen.

## 6. Core Features (Functional Requirements)

| # | Feature | Priority |
|---|---|---|
| F1 | Multi-photo upload (burst of 5–15 images) | P0 |
| F2 | **AI Gate 1** — input validation (same group/scene, usable quality) | P0 |
| F3 | Face detection + landmark extraction per frame | P0 |
| F4 | Face matching/clustering — group detected faces into consistent per-person identities across frames | P0 |
| F5 | Expression scoring per face instance (eyes open, smiling, gaze-to-camera) | P0 |
| F6 | Best-frame selection per person | P0 |
| F7 | Seamless compositing/blending (Poisson blending) of selected faces onto a base frame | P0 |
| F8 | **AI Gate 2** — output validation (naturalness, artifact detection) | P0 |
| F9 | Retry loop — reattempt blend with adjusted parameters on Gate 2 failure | P0 |
| F10 | Graceful fallback — best single whole frame if retries exhausted | P0 |
| F11 | Before/after reveal UI | P0 |
| F12 | Per-face confidence/reasoning display ("why this frame was picked") | P1 |
| F13 | Downloadable/shareable result | P1 |
| F14 | Processing-time visualization (contact-sheet-style live scoring) | P1 — high demo value |
| F15 | History/gallery of past generations | P2 (stretch) |

## 7. Success Metrics

**Aligned to the stated judging criteria:**

| Criterion | Weight | How we address it |
|---|---|---|
| Idea & Creativity | 30% | Novel "AI-verifies-itself" double-gate architecture; contact-sheet framing |
| Scalability & Feasibility | 30% | Clear path to phone-OEM camera feature / event-photography SaaS; free-tier deployable today |
| Technical Complexity & Execution | 20% | Face clustering + expression scoring + Poisson blending + dual AI verification gates + retry state machine |
| Presentation | 15% | Live before/after reveal, visible self-correction moment |
| Wow Factor | 5% | Real-time face fix on judges' own photo, taken live |

**Product-level (post-hackathon framing):**
- Blend success rate without human-visible artifacts (target >90% on the retry+fallback system, not the raw first-attempt blend).
- End-to-end processing time under 15 seconds for a 10-photo burst.

## 8. Constraints

- **Timeline:** 5 days to submission (July 18).
- **Team:** solo/small team, building via Google Antigravity with Gemini 3.5 Flash, Claude Sonnet 4.6, and Opus 4.6 as coding agents.
- **Budget:** free-tier APIs and open-source CV libraries only (MediaPipe, OpenCV) — no paid compute for the hackathon build.
- **Demo environment:** must work reliably on unpredictable venue wifi/lighting — this drives the fallback-first design philosophy throughout.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Poisson blending looks artificial under bad lighting | AI Gate 2 catches it; fallback to unedited frame; budget Day 2 entirely for blend quality |
| Face clustering mismatches people across frames | Use embedding similarity threshold + manual override in demo dataset; test on real varied bursts early |
| AI Gate latency kills live demo | Cache/pre-warm a rehearsed demo run as backup; keep gate prompts short, use fastest available model for gates |
| Live demo photos are worse than test data | Rehearse Day 5 specifically with unpredictable, judge-taken photos |
