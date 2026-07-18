# Kairos — Detailed Agent-Wise Implementation Plan

This document outlines the detailed build order for the Kairos pipeline, organized by day and task, with ready-to-use prompts for each agent:
- **Opus 4.6**: Computer Vision, math, and core processing logic.
- **Claude Sonnet 4.6**: State machine, backend routing, API integration, and system wiring.
- **Gemini 3.5 Flash (High)**: Scaffolding, testing, and UI integration.

Additionally, a custom prompt for **Lovable** is included at the end to generate the React frontend.

---

## Day 1 — Core Detection & Scoring Pipeline (no UI)

### Step 1.1: Project Setup and Configuration
- **Assignee:** Claude Sonnet 4.6
- **Prompt:**
```markdown
Set up the workspace structure and requirements for the Kairos backend.
1. Create a `backend/requirements.txt` containing:
   - fastapi>=0.110.0
   - uvicorn>=0.28.0
   - pydantic>=2.0.0
   - python-dotenv>=1.0.0
   - opencv-python-headless>=4.9.0
   - mediapipe>=0.10.11
   - deepface>=0.0.92
   - numpy>=1.26.0
   - pillow>=10.2.0
   - google-generativeai>=0.4.0
   - anthropic>=0.18.0
2. Create `backend/app/config.py` containing global configuration settings:
   - Scoring weights (EYE_WEIGHT = 0.4, SMILE_WEIGHT = 0.4, GAZE_WEIGHT = 0.2)
   - Default blending options (DEFAULT_FEATHER_RADIUS = 15, DEFAULT_MASK_EROSION = 5)
   - State machine parameters (MAX_RETRIES = 2)
   - Model options for DeepFace (defaulting to "ArcFace" or "Facenet512")
   - Face clustering confidence threshold (CLUSTER_THRESHOLD = 0.68)
   - Ensure all parameters are named constants with typing annotations and docstrings.
3. Set up a basic `backend/app/main.py` containing a FastAPI app with stub routers for:
   - POST /api/burst (returns a mock burst_id)
   - GET /api/burst/{id}/status (returns mock state "processing")
   - GET /api/burst/{id}/result (returns mock PipelineResult)
   - POST /api/burst/{id}/retry (returns mock status)
Make sure the folder structure complies with PEP 8.
```

### Step 1.2: MediaPipe Landmark Extraction (`detector.py`)
- **Assignee:** Opus 4.6
- **Prompt:**
```markdown
Write the module `backend/app/pipeline/detector.py` to extract facial landmarks using MediaPipe.
1. Create a class `FaceMeshDetector` that initializes the MediaPipe Face Mesh module.
2. Implement a method `detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]`:
   - It must take a numpy image array (BGR or RGB).
   - Run MediaPipe Face Mesh on the image.
   - For each detected face, extract:
     * A bounding box `bbox` as a tuple of `(x_min, y_min, width, height)` in absolute pixels.
     * All 468 (or 478 if iris is enabled) landmarks as a list of `(x, y, z)` absolute coordinates scaled to the image dimensions.
   - Return a list of face dictionaries, each containing:
     * `bbox`: Bounding box.
     * `landmarks`: The list of 3D landmark coordinates.
3. Include unit-testing code or a `__main__` block that loads a test image, runs the detector, and logs/visualizes the bounding boxes and landmarks. Ensure type hints and thorough docstrings are used.
```

### Step 1.3: DeepFace Face Clustering (`clustering.py`)
- **Assignee:** Opus 4.6
- **Prompt:**
```markdown
Write the module `backend/app/pipeline/clustering.py` to group faces of the same person across frames using DeepFace.
1. Create a class `FaceClusteringManager` that takes a distance threshold from `config.py` (default: 0.68).
2. Implement a method `cluster_faces(self, frames_faces: Dict[int, List[Dict[str, Any]]], frames_images: Dict[int, np.ndarray]) -> List[PersonCluster]`:
   - `frames_faces` maps a frame index to a list of detected faces (with bboxes and landmarks).
   - `frames_images` maps a frame index to the original numpy image array.
   - For each face instance in each frame:
     * Crop the face bounding box from the image.
     * Generate an embedding vector using DeepFace (e.g. `DeepFace.represent(img_path, model_name="ArcFace", enforce_detection=False)`).
   - Cluster these embeddings across all frames to group the face instances into consistent person identities. Since we are in a burst of 5-15 photos, the number of clusters should ideally match the number of people in the photo.
   - You can use DBSCAN (from `sklearn.cluster` if installed, or implement a simple clustering logic using cosine/Euclidean distance matrix comparisons).
   - Ensure the clustering is robust to expression/lighting changes.
3. Return a list of `PersonCluster` objects (or matching dict structures) containing:
   - `cluster_id` (e.g. "person_0", "person_1")
   - `face_instances`: A list of references to the face dictionaries, each updated with the `person_cluster_id`.
4. Include clean error handling if no faces are detected or if clustering fails. Add type hints and docstrings.
```

### Step 1.4: Expression Scoring Module (`scoring.py`)
- **Assignee:** Opus 4.6
- **Prompt:**
```markdown
Write the module `backend/app/pipeline/scoring.py` to evaluate facial expressions using MediaPipe landmarks.
1. Implement helper functions for three metrics:
   - **Eyes Open (EAR)**: Compute the Eye Aspect Ratio for left and right eyes using the MediaPipe landmarks:
     * `EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)`
     * Normalize the output between 0.0 (closed) and 1.0 (fully open).
   - **Smile Curve**: Compute the curvature and width-height ratio of the mouth landmarks.
     * Use the corners of the mouth and the top/bottom lips.
     * Score should scale from 0.0 (neutral/frown) to 1.0 (broad smile).
   - **Gaze Forward**: Determine how centered the pupil/iris is relative to the eye corners.
     * Calculate the ratio of distance between the pupil center (iris landmarks 468-477 if available, or estimated center) and the inner/outer eye corners.
     * Return 1.0 if looking straight at the camera, scaling down as the gaze drifts.
2. Implement `score_face(landmarks: List[Tuple[float, float, float]]) -> ExpressionScore`:
   - Compute individual `eyes_open`, `smile`, and `gaze_forward` scores.
   - Calculate the `composite_score` as a weighted combination of these scores using parameters from `config.py`.
3. Ensure the module handles head-tilt or roll by calculating relative coordinate distances rather than absolute vertical/horizontal axes.
```

### Step 1.5: Day 1 CLI Pipeline Verification Script
- **Assignee:** Claude Sonnet 4.6
- **Prompt:**
```markdown
Create a CLI validation script `backend/test_pipeline_day1.py` that ties the detector, clustering, and scoring modules together.
1. The script should:
   - Accept a path to a directory containing a photo burst (e.g., JPEG/PNG files).
   - Load each image.
   - Run detection (`detector.py`), embedding generation + clustering (`clustering.py`), and scoring (`scoring.py`) for each frame.
   - Print a clean summary to the console:
     * Detected clusters (people) and the number of face instances in each.
     * For each cluster, list the face index, frame index, and composite expression score.
     * Clearly print the recommended "best" frame for each person.
   - Save face crops grouped by cluster to a directory `/test-data/output_clusters/` for visual review.
2. The script must be executable from the terminal. Write robust file I/O handling and log all actions using the `logging` module.
```

---

## Day 2 — Blending (the hardest, highest-value day)

### Step 2.1: Face Alignment and Warping (`aligner.py`)
- **Assignee:** Opus 4.6
- **Prompt:**
```markdown
Write the module `backend/app/pipeline/aligner.py` to align and warp a source face to fit a target face.
1. Implement a function `align_face(src_image: np.ndarray, src_landmarks: List[Tuple[float, float, float]], dst_landmarks: List[Tuple[float, float, float]]) -> np.ndarray`:
   - It must compute an affine or similarity transformation matrix that maps the source face landmarks to the destination/target face landmarks.
   - Use stable landmark points that do not move significantly with expressions (e.g., outer corners of eyes, nose bridge, ears).
   - Apply `cv2.warpAffine` or `cv2.warpPerspective` to warp the entire `src_image` so the face matches the position, rotation, and size of the destination face in the base frame.
   - Return the warped source image.
2. Ensure the code handles cases where faces are tilted, slightly turned, or at different distances from the camera.
```

### Step 2.2: Blending and Mask Generation (`blender.py`)
- **Assignee:** Opus 4.6
- **Prompt:**
```markdown
Write the module `backend/app/pipeline/blender.py` to blend an aligned source face onto a base image.
1. Implement a method `create_face_mask(base_image: np.ndarray, landmarks: List[Tuple[float, float, float]], erosion_pixels: int, feather_radius: int) -> np.ndarray`:
   - Use the outer boundary face landmarks (jawline, eyebrows) to construct a convex hull.
   - Create a binary mask of the convex hull.
   - Erode the mask slightly to pull the boundary away from hair or ears.
   - Apply Gaussian blur to the mask borders to create a soft feathering transition.
2. Implement `blend_face(base_image: np.ndarray, warped_src_image: np.ndarray, mask: np.ndarray) -> np.ndarray`:
   - Find the bounding box of the non-zero region in the mask.
   - Find the center of the bounding box.
   - Use OpenCV's Poisson blending: `cv2.seamlessClone(warped_src_image, base_image, mask, center, cv2.NORMAL_CLONE)` (or `cv2.MIXED_CLONE` depending on lighting).
   - Return the blended composite image.
3. Expose hooks in the arguments to adjust parameters: mask erosion size, blend mode, and feather/blur radius, which Day 3's retry loop will use to fix bad blends.
```

### Step 2.3: Day 2 Integration Script
- **Assignee:** Claude Sonnet 4.6
- **Prompt:**
```markdown
Create a script `backend/test_blender_day2.py` to execute the full face replacement pipeline.
1. The script should:
   - Accept a path to a burst directory.
   - Perform detection, clustering, and scoring.
   - Select a base image (the frame with the highest overall group score).
   - Identify the best face instance for each of the other clusters.
   - Align the best face from its source frame to the base frame (`aligner.py`).
   - Blend the face onto the base image (`blender.py`).
   - Save the composite result to `/test-data/composite_result.jpg`.
2. Save side-by-side comparison images of the original base face and the blended face to `/test-data/blends/` for manual inspection.
```

---

## Day 3 — AI Gates, Retry Loop, Robustness

### Step 3.1: AI Gates Integration (`gates.py`)
- **Assignee:** Claude Sonnet 4.6
- **Prompt:**
```markdown
Write the module `backend/app/pipeline/gates.py` to implement the AI verification gates using Gemini and Anthropic APIs.
1. Create a class `AIGateways` that reads API keys from environment variables (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`).
2. Implement `check_gate1_inputs(self, images: List[np.ndarray]) -> GateResult`:
   - Verify: (1) Same group/scene, (2) Usable quality, (3) At least 2 faces present.
   - Call the chosen LLM (Gemini 3.5 Flash or Claude 3.5 Sonnet) with a downsized grid of all burst photos.
   - Request structured JSON output matches this schema:
     `{"valid": bool, "person_count_estimate": int, "issues": [{"type": str, "description": str}]}`
3. Implement `check_gate2_output(self, composite: np.ndarray, base_image: np.ndarray) -> GateResult`:
   - Inspect the composite for: seams, lighting mismatches, warped facial features, duplicate structures.
   - Send the composite (and base frame for comparison) to the vision LLM.
   - Request structured JSON output matching:
     `{"natural": bool, "confidence": float, "flagged_people": [{"person_cluster_id": str, "issue_type": str, "description": str}]}`
4. Implement a `demo-fallback` configuration toggle that reads cached JSON responses from a local folder `/backend/app/cache/` instead of hitting the live APIs, ensuring the demo works even if there is no internet connection.
```

### Step 3.2: Retry/Fallback State Machine and Routes
- **Assignee:** Claude Sonnet 4.6
- **Prompt:**
```markdown
Integrate the pipeline and endpoints in the FastAPI app (`backend/app/main.py` and `backend/app/pipeline/state_machine.py`).
1. Create a `PipelineCoordinator` that implements the retry/fallback loop:
   - Validate input burst (Gate 1).
   - Run CV pipeline (detect, cluster, score).
   - Composite best faces onto the base frame.
   - Validate composite (Gate 2).
   - If Gate 2 fails, trigger a retry (up to 2 times):
     * Swap the flagged face with the next-highest scored face candidate for that cluster.
     * AND/OR adjust parameters (e.g. increase mask erosion or modify feathering).
     * Re-run Gate 2 check.
   - If retries fail, execute fallback: select the single original frame with the highest summed group score and mark the result type as `fallback_single_frame`.
2. Implement backend endpoints:
   - `POST /api/burst`: Accepts multiple uploaded image files, initiates an async background task to run the pipeline, and returns a `burst_id`.
   - `GET /api/burst/{id}/status`: Stream progress updates using Server-Sent Events (SSE) or simple polling. Return details of the current state: `Gate 1 validation`, `scoring`, `blending`, `Gate 2 checking`, `retrying`, or `done`.
   - `GET /api/burst/{id}/result`: Returns the final `PipelineResult` (containing output image URL/base64, retry counts, gate details, and face statistics).
   - `POST /api/burst/{id}/retry`: Manually forces a retry cycle for live demo safety.
3. Manage job status using an in-memory dictionary.
```

---

## Day 4 — Frontend Polish, Full Pipeline Wiring, Deployment

### Step 4.1: Frontend Setup & Lovable Integration
- **Assignee:** Gemini 3.5 Flash (High)
- **Prompt:**
```markdown
Initialize the React frontend project under `/frontend/` using Vite and Tailwind CSS.
1. Run initialization script in `/frontend/` to create a standard Vite+React template.
2. Install dependencies: `lucide-react`, `tailwindcss`, `autoprefixer`, `postcss`.
3. Set up the Tailwind configuration.
4. Integrate the React code generated by Lovable (from the Lovable Prompt) into `src/` (components, landing page, results view, hooks).
5. Ensure page structure uses proper semantic elements, has mobile responsiveness, keyboard accessibility, and outline indicators for elements in focus.
```

### Step 4.2: Frontend-Backend API Wiring
- **Assignee:** Claude Sonnet 4.6
- **Prompt:**
```markdown
Wire the React frontend components to the FastAPI backend API endpoints.
1. Write an API service layer in `frontend/src/services/api.ts` (or similar file) to talk to:
   - `POST /api/burst` (handles multi-file upload)
   - `GET /api/burst/{id}/status` (polls status or connects to SSE endpoint to track processing states)
   - `GET /api/burst/{id}/result` (fetches final composite, scores, and statistics)
   - `POST /api/burst/{id}/retry` (handles manual retry trigger)
2. Connect state variables in the components so the progress indicator updates in real-time as the backend moves through:
   - Input Gate Checking
   - Processing Contact Sheet (displaying thumbnails and face nodes)
   - Output Gate Checking
   - Retrying (if Gate 2 fails)
   - Showing Final Result (rendering Before/After comparison slider)
3. Handle error states cleanly, showing user-friendly warning cards if Gate 1 fails or if the backend endpoint is unreachable.
```

---

## Day 5 — Rehearsal, Stress Test, Submission

### Step 5.1: Test Suite Scaffolding
- **Assignee:** Gemini 3.5 Flash (High)
- **Prompt:**
```markdown
Create unit and integration tests under a `/backend/tests/` directory.
1. Create `test_scoring.py` containing tests for landmarks math (EAR, smile curvature ratios) using mocked nose, eye, and lip coordinates.
2. Create `test_state_machine.py` to mock Gate 2 API responses (fail, fail, fail) and verify the coordinator accurately transitions through `RETRYING (1/2)` -> `RETRYING (2/2)` -> `FALLBACK` and selects the correct highest-scoring base frame.
3. Create `test_api.py` using `fastapi.testclient.TestClient` to verify upload, polling, and results response schemas.
4. Ensure tests can be executed via `pytest`.
```

### Step 5.2: Deploys and Demo-Day Preparations
- **Assignee:** Claude Sonnet 4.6
- **Prompt:**
```markdown
Perform deployment and configuration verification steps.
1. Build frontend code with `npm run build` and resolve any bundling/TypeScript errors.
2. Set up Vercel project configuration (`vercel.json`) for the frontend.
3. Configure Backend setup for deployment on Render/Fly.io (write a basic `Dockerfile` or start script).
4. Verify the `demo-fallback` environment variable. Ensure that when set to `TRUE`, the API router loads predefined image assets and cached gate responses from `backend/app/cache/` instead of querying external endpoints.
```

---

## Lovable Frontend Generation Prompt
*Copy-paste the prompt below into Lovable to generate the entire React interface.*

```text
Create a premium single-page web application for "Kairos", a photo-composite tool that takes a burst of group photos, finds the best expression for each person, blends them into one photo, and runs double validation gates. 

Focus entirely on layout, interactive state machines, grid arrangements, progress monitoring, and accessibility. Do not prescribe colors or font families—rely on system variables or classes so these can be styled externally.

Generate the following screens and interaction flows:

1. Viewfinder Lightbox (Upload View)
- A large, centered interactive drop zone modeled after a camera viewfinder. 
- Displays corner overlay brackets (viewfinder reticles) and EXIF metadata markers (e.g. "READY", "AWAITING BURST", "f/2.8 · ISO 400 · 35mm").
- Supports multi-file selection (target: 5-15 files).
- When files are dropped or selected, show an upload progress indicator and then transition to the processing view.

2. Interactive Contact Sheet (Processing View)
- An exposure grid resembling a darkroom contact sheet film strip.
- Arranges uploaded thumbnails. Each thumbnail frame displays:
  - An index tag (e.g., "01", "02").
  - Small vector/SVG face markers mapped to detected face positions.
  - Hovering a face reveals a small popover showing expression details (e.g., "Eyes open: 95%", "Smile: 90%").
- Highlight selected "winning" faces for each person. Draw a circled selection ring around the winning face in each frame.
- Display a detailed console log or telemetry line showing which faces are currently winning (e.g. "Person A: Frame 3, Person B: Frame 5").

3. AI Gate Verification Panel (Status Readout)
- An instrument panel or terminal console displaying status updates.
- Display three main checkpoints:
  - "Gate 1: Input Check" (Validates if photos belong to the same scene, group, and are clear).
  - "Gate 2: Output Check" (Validates the composite for seamless blending, lighting, and anomalies).
  - "Pipeline Actions/Retries" (Displays retry counts, swapped frame details, or fallback warnings).
- Checkpoints should have state indicators (e.g., "Checking", "Passed", "Flagged", or "Failed") with secondary detail messages (e.g., "Passed: 4 people detected", "Flagged: Seam artifact on Person B - Swapping face to Frame 2").
- Support animating these status lines sequentially to simulate the processing pipeline.

4. Before / After Slide Viewer (Reveal View)
- A visual split comparison window.
- Displays the original "base" photo on one side, and the final blended composite on the other.
- Provides a horizontal range slider handle that users can drag back and forth to adjust the split line (using clip-path).
- Shows labels: "BEFORE - ORIGINAL" and "AFTER - COMPOSITE".

5. Results Sheet (Final View)
- Renders the finalized composite photo.
- Includes a descriptive badge detailing the optimization (e.g., "3 of 3 faces optimized", "Gate 2 confidence: 96%").
- Provides a "Download Image" action button.
- If the pipeline resulted in a fallback (failed to find a secure blend), show a clear notice badge indicating "Showing original base frame—composite did not meet quality thresholds."

Interactive Simulation Layer:
- Include a simulation engine so the UI works immediately with mock data. 
- Triggering the upload should run through a 5-second simulated sequence: 
  1. Uploading -> 2. Gate 1 passes -> 3. Face mesh extraction -> 4. Scoring & Clustering (animating the circle marks onto winning frames) -> 5. Blending -> 6. Gate 2 flags a seam -> 7. Retry triggers (updating status logs and showing face swap) -> 8. Gate 2 passes on Retry -> 9. Reveal slider becomes interactive.
- Structure all backend API requests into a clean, detached services file (`api.ts`) using fetch hooks so it can be easily wired to endpoints like `POST /api/burst` and SSE/polling status updates.
- Ensure the layout is fully responsive, keyboard navigable (with visible focus indicators), and respects reduced-motion preferences.
```
