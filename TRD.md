# TRD — Kairos Technical Requirements Document

Companion to `PRD.md`. This is the source of truth for architecture, contracts, and pipeline behavior — AI coding agents in Antigravity should treat this as binding unless `AGENTS.md` says otherwise.

---

## 1. Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐     ┌──────────────┐
│   Frontend   │────▶│   Backend    │────▶│   CV Pipeline      │────▶│  AI Gates    │
│  (React/Vite)│◀────│  (FastAPI)   │◀────│  (Python worker)  │◀────│ (Vision LLM) │
└─────────────┘     └──────────────┘     └───────────────────┘     └──────────────┘
```

**Pipeline (per the agreed design):**

```
1. Upload burst (5–15 photos)
        │
        ▼
2. AI GATE 1 — Input Validation
   "Are these the same group / same scene / usable quality?"
        │  pass                              │ fail
        ▼                                     ▼
3. CV Core Pipeline                    Return error to user
   3a. Face detection (MediaPipe)      with specific reason
   3b. Face clustering (per-person
       identity across frames)
   3c. Expression scoring per face
   3d. Best-frame selection per person
   3e. Poisson blend compositing
        │
        ▼
4. AI GATE 2 — Output Validation
   "Does this look natural? Any artifacts?"
        │  pass                              │ fail
        ▼                                     ▼
5. Return composite to user          6. RETRY LOOP (max 2 attempts)
                                         - swap flagged person's face
                                           to their next-best-scored frame
                                         - adjust blend mask/feather radius
                                         - re-run Gate 2
                                              │  pass          │ still fail
                                              ▼                ▼
                                      Return composite   7. FALLBACK
                                                             Return best single
                                                             whole frame
                                                             (highest total
                                                             group score,
                                                             unedited)
```

**Critical design principle:** the user must never see a blend that Gate 2 has flagged. The system either fixes it, or hands back a real, unedited photo. This is the core reliability story for both the pitch and the live demo.

## 2. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Frontend | React + Vite + Tailwind | Fast iteration, matches Antigravity's default scaffolding |
| Backend | Python + FastAPI | Async-friendly, plays well with CV libs, easy to expose job status endpoints |
| Face detection/landmarks | MediaPipe Face Mesh (Tasks API) | Free, fast, well-documented, runs client-or-server side |
| Face clustering/identity | `face_recognition` (dlib ResNet embeddings) or InsightFace ArcFace | Needed to match the *same person* across different burst frames — MediaPipe alone gives per-frame landmarks, not cross-frame identity |
| Blending | OpenCV `seamlessClone` (Poisson/Mixed blending) | Built-in, well-tested, no training required |
| AI Gates | Claude Sonnet 4.6 (vision) via API, or Gemini 3.5 Flash for speed | Structured JSON output, multimodal, fast enough for live demo |
| Job/state | In-memory dict or Redis if time allows | 5-day timeline — start in-memory, upgrade only if needed |
| Deployment | Frontend: Vercel. Backend: Render/Fly.io | Free tier, fast deploys, public demo URL |

## 3. Data Models

```python
class BurstUpload:
    burst_id: str
    photos: list[PhotoFile]  # 5–15 images
    uploaded_at: datetime

class FaceInstance:
    face_id: str
    frame_index: int
    person_cluster_id: str       # which person this face belongs to (post-clustering)
    bbox: tuple[int, int, int, int]
    landmarks: list[tuple[float, float]]
    scores: ExpressionScore

class ExpressionScore:
    eyes_open: float       # 0-1
    smile: float           # 0-1
    gaze_forward: float    # 0-1
    composite_score: float # weighted combination

class PersonCluster:
    cluster_id: str
    face_instances: list[FaceInstance]
    best_face_id: str
    fallback_face_ids: list[str]  # ranked, for retry loop

class GateResult:
    passed: bool
    confidence: float
    issues: list[GateIssue]

class GateIssue:
    person_cluster_id: str | None
    issue_type: str   # "seam_artifact" | "warped_feature" | "lighting_mismatch" | "scene_mismatch" | etc.
    description: str

class PipelineResult:
    burst_id: str
    status: str  # "processing" | "complete" | "fallback" | "error"
    result_type: str  # "blended" | "fallback_single_frame"
    output_image_url: str
    retry_count: int
    gate_1_result: GateResult
    gate_2_result: GateResult
    per_person_reasoning: list[dict]  # for F12 "why this frame" UI
```

## 4. API Endpoints

```
POST   /api/burst              — upload burst photos, returns burst_id
GET    /api/burst/{id}/status  — poll processing status (or use WebSocket/SSE)
GET    /api/burst/{id}/result  — final PipelineResult once complete
POST   /api/burst/{id}/retry   — manual re-trigger (debug/demo safety net)
```

Use Server-Sent Events or polling (not raw WebSockets) for status updates — simpler to get right in 5 days, and sufficient since this is single-user-at-a-time per session, not the real-time multi-user case.

## 5. AI Gate Specifications

Both gates are multimodal LLM calls with **strict structured JSON output** — never free text, so the retry logic can parse it deterministically.

### Gate 1 — Input Validation

**Input:** all uploaded burst photos (thumbnails, downsized to keep tokens/latency low)

**Prompt intent:**
> "These images are claimed to be a burst — the same group of people, same scene, taken seconds apart. Verify: (1) same people appear across the images, (2) same background/location, (3) images are not blurry/corrupted/unusable, (4) at least 2 distinguishable faces are present. Return structured JSON only."

**Output contract:**
```json
{
  "valid": true,
  "person_count_estimate": 4,
  "issues": []
}
```
or
```json
{
  "valid": false,
  "person_count_estimate": 2,
  "issues": [{"type": "scene_mismatch", "description": "Images 3-4 appear to be a different location"}]
}
```

### Gate 2 — Output Validation

**Input:** the composite output image (+ optionally the source frame it was based on, for comparison)

**Prompt intent:**
> "This image is the output of an automated face-compositing tool that merges the best-expression face for each person from multiple photos into one image. Inspect closely for: visible seams around face boundaries, mismatched skin tone/lighting between a face and its surroundings, warped or duplicated features, unnatural edges. Rate confidence 0-1 that this looks like a single, naturally-taken photograph. Return structured JSON only."

**Output contract:**
```json
{
  "natural": true,
  "confidence": 0.94,
  "flagged_people": []
}
```
or
```json
{
  "natural": false,
  "confidence": 0.41,
  "flagged_people": [
    {"person_cluster_id": "p2", "issue_type": "seam_artifact", "description": "Visible edge along jawline"}
  ]
}
```

**Important constraint (from the earlier design discussion):** Gate 2 can *detect* problems but should **not** be relied on to *directly repair* pixels — vision LLMs don't do inline image editing. "Fixing" means the CV pipeline retries with a different face candidate or adjusted blend parameters, not the LLM regenerating pixels. AI-inpainting-based auto-repair is a Day 4+ stretch goal only, never the load-bearing path.

## 6. Retry / Fallback State Machine

```
State: BLENDING
  → Gate 2 pass  → COMPLETE (result_type = "blended")
  → Gate 2 fail  → RETRYING (attempt 1)

State: RETRYING (attempt N, max 2)
  action: for each flagged person, swap to their next-ranked face candidate
          OR widen/narrow the blend feather radius
  → Gate 2 pass       → COMPLETE (result_type = "blended")
  → Gate 2 fail, N<2  → RETRYING (attempt N+1)
  → Gate 2 fail, N=2  → FALLBACK

State: FALLBACK
  action: select the single whole original frame with the highest
          summed group expression score, return unedited
  → always → COMPLETE (result_type = "fallback_single_frame")
```

## 7. Non-Functional Requirements

- **Latency budget (target, for live demo confidence):**
  - Face detection + clustering + scoring: <3s for 10 photos
  - Blend compositing: <2s per attempt
  - Gate 1 call: <2s
  - Gate 2 call: <2s per attempt
  - Total worst case (2 retries): ~15s — communicate this via a processing UI (see `UX_DESIGN.md`), don't hide the wait, make it part of the show
- **Reliability:** pipeline must never crash on malformed input — always resolve to either a valid composite, a fallback frame, or a clear Gate 1 error message.
- **Privacy:** uploaded photos are processed in-memory/temp storage only; no persistence beyond the session unless F15 (gallery) is explicitly built.

## 8. Deployment Plan

- Frontend → Vercel (auto-deploy from `main`)
- Backend → Render or Fly.io free tier (watch cold-start latency — pre-warm before demo)
- Environment secrets (LLM API keys) via platform env vars, never committed — see `AGENTS.md` rules
- Keep a `demo-fallback` mode: a local `.env` flag that swaps live AI Gate calls for cached responses on a known-good rehearsed photo set, in case venue wifi fails
