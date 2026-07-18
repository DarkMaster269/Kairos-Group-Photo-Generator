# TESTING.md — Testing Strategy

Testing philosophy for a 5-day hackathon build: fast feedback over exhaustive coverage, and **demo-safety** — the single most important test is "does this fail gracefully in front of judges" — matters more than unit test percentage coverage.

---

## 1. Test Data — Build This Early (Day 1)

Real photos matter more than synthetic ones for this project; face detection/scoring/blending quality is genuinely hard to judge from code alone.

- **Good bursts:** 4-5 real bursts (phone burst mode) of different real groups, varying lighting, 2-6 people each
- **Adversarial bursts (Day 3):** deliberately mismatched photos (different people, different scenes) to test Gate 1 rejection
- **Known-bad composites (Day 3):** save the visibly broken blends you'll inevitably produce while iterating on Day 2 — these become your Gate 2 validation set. If Gate 2 doesn't flag these, the gate isn't working, regardless of what the code review looks like.
- Store all test data in `/test-data/` with a short `README.md` noting what each set is for — don't let this get lost by Day 5.

## 2. Unit Tests

| Component | What to test |
|---|---|
| Expression scorer | Given known landmark coordinates, eyes-open/smile scores fall in expected ranges (use a few hand-labeled examples) |
| Face clustering | Given faces from multiple frames of a known 3-person burst, all instances of the same person land in the same cluster |
| Best-frame selection | Given a set of scored faces per person, the highest-scoring one is selected |
| Fallback selection | Given a set of whole-frame group scores, the correct highest-scoring single frame is returned |
| Retry state machine | Simulate Gate 2 fail → fail → fail transitions correctly reach FALLBACK state after max retries (per `TRD.md` §6) |

## 3. Integration Tests

- Full pipeline, good burst in → composite out, no exceptions, output image is valid
- Full pipeline, adversarial burst in → Gate 1 rejects with a specific, correct reason
- Full pipeline, forced Gate 2 failure (mock it) → confirm retry triggers, then confirm fallback triggers if retries exhausted
- API endpoint tests: upload → status polling → result retrieval, including error states

## 4. AI Gate Validation (specific to this project's core risk)

The gates are the differentiator — they need their own validation, separate from generic unit tests:

- Run Gate 1 against 5+ valid bursts (should pass) and 5+ invalid bursts (should fail with correct reason) — track precision/recall informally, don't ship a gate that rejects good input
- Run Gate 2 against your known-good composites (should pass) and known-bad composites from Day 2 (should fail, and ideally correctly identify *which* person/region is flagged)
- Watch gate latency specifically — if it's pushing the demo past ~15-20s total, that's a UX problem even if the gate is accurate (see `TRD.md` §7 latency budget)

## 5. Manual QA Checklist (run before Day 5 rehearsal, and again before submission)

- [ ] Upload works with 5, 10, and 15-photo bursts
- [ ] Upload gracefully rejects non-image files and mismatched bursts with a clear message
- [ ] A genuinely bad blend never reaches the user unflagged — force this case and confirm
- [ ] Fallback path produces a real, sensible photo (not blank/broken)
- [ ] Full flow works on a fresh browser session with no cache (simulates a judge's laptop)
- [ ] Full flow works on mobile viewport (per `UX_DESIGN.md` responsiveness requirement)
- [ ] Keyboard navigation and visible focus states work (accessibility floor per `UX_DESIGN.md`)
- [ ] Public deployed URL works, not just localhost
- [ ] Backend cold-start latency checked — pre-warm before the actual demo slot
- [ ] `demo-fallback` cached-response mode (per `TRD.md` §8) works if you flip it on, in case venue wifi fails

## 6. Demo-Day Rehearsal Plan (Day 5)

- Have 2-3 people who haven't seen the project take fresh burst photos of themselves live, on their own phones, in whatever lighting is available — this is the closest simulation of judge conditions
- Time the full pitch + live demo at least twice
- Prepare one pre-processed "known good" result as a backup slide/tab in case live capture goes wrong — never let the whole pitch depend on a single live network call
- Explicitly rehearse narrating the self-correction moment if it happens live ("see, Gate 2 just caught a lighting mismatch and it's retrying") — this can be a stronger demo moment than a clean first-try pass if you frame it confidently

## 7. What NOT to Spend Time On

Given the 5-day constraint, deliberately skip: full CI/CD pipelines, comprehensive cross-browser testing beyond Chrome/Safari basics, load/performance testing beyond single-demo-session scale, and automated E2E test frameworks (Playwright/Cypress) unless Day 4/5 has real slack — manual QA against the checklist above is sufficient for a hackathon submission and a better use of limited time.
