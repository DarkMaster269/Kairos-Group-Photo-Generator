[Uploading README.md…]()
# Kairos — Perfect Group Photo Generator

**Kairos** is an intelligent group photo optimization tool. It captures the opportune moment (καιρός) from a burst of group photographs and merges the best expressions of each person into a single, seamless, high-quality composite photo.

What makes Kairos unique is its **dual AI-verification-gate architecture** and **retry-based state machine**. The tool self-checks the input burst and its own blended output using vision LLMs, adjusting parameters or swapping candidates to correct anomalies before the final image is ever presented to the user.

---

## 📸 The Problem & The Solution

**The Problem:** Group photos are notoriously difficult. Someone blinks, someone looks away, or someone is mid-smile. While burst shots capture the perfect expression for each individual across different frames, manually slicing and blending them in Photoshop is time-consuming and requires professional editing skills.

**The Solution:** Kairos automates this entire process:
1. **Scans** a burst of 5–15 photos.
2. **Detects** and **clusters** faces of the same individuals across frames.
3. **Scores** facial expressions based on eye-openness, smiles, and camera gaze.
4. **Poisson-blends** the highest-scoring face for each person onto a single base frame.
5. **AI self-verifies** quality, retrying with adapted parameters if seams or lighting anomalies are found, or falling back gracefully to the best unedited single frame.

---

## ⚙️ Architecture & Pipeline Flow

```
                  ┌───────────────┐
                  │ Upload Burst  │ (5–15 photos of the same group/scene)
                  └───────┬───────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │ GATE 1 (AI Verification) │ ────[ FAIL ]──▶ [ Return Error to User ]
            └─────────────┬────────────┘
                          │
                       [ PASS ]
                          │
                          ▼
            ┌──────────────────────────┐
            │    CV Core Pipeline      │
            │  - Face Detection        │ (MTCNN + MediaPipe FaceMesh)
            │  - Identity Clustering   │ (Spatial Tracking & Linkage)
            │  - Expression Scoring    │ (Openness, Smiles, Gaze)
            │  - Best Face Selection   │
            │  - Poisson Blending      │ (OpenCV seamlessClone)
            └─────────────┬────────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │ GATE 2 (AI Verification) │ ◀───┐
            └─────────────┬────────────┘     │
                          │                  │ [ FAIL, Retry < 2 ]
                       [ PASS ]              │
                          ├──────────────────┘
                          │
                       [ FAIL, Retry >= 2 ]
                          │
                          ├─────────────────────────────────┐
                          │                                 │
                          ▼                                 ▼
              ┌───────────────────────┐         ┌───────────────────────┐
              │   Return Composite    │         │   Fallback Single     │
              │   Blended Image       │         │   Best-Scored Frame   │
              └───────────────────────┘         └───────────────────────┘
```

### 🧠 The Dual AI Gates

1. **Gate 1 (Input Validation):** Downsizes and constructs a grid of the uploaded burst. A vision LLM inspects it to verify that all photos represent the same group and scene in usable quality, and that at least 2 faces are visible.
2. **Gate 2 (Output Verification):** Inspects the final composite against the original base frame. It detects visible seams, double-edges, warped features, or lighting/skin tone mismatches. If issues are found, it flags specific individuals (`person_0`, `person_1`) and types of artifacts to direct the retry loop.

---

## 🛠️ Tech Stack

### Backend (Python + FastAPI)
* **Web Framework:** FastAPI for fast, asynchronous API endpoints.
* **Face Detection & Landmarks:** MediaPipe Face Mesh (Tasks API) for high-fidelity 478-point landmark mapping.
* **Global Scene Detection:** DeepFace (MTCNN detector backend) with rotation augmentation (0°, -20°, 20°) to locate tilted/leaning faces.
* **Identity Clustering:** Spatial Euclidean distance-based linkage tracking with disjointness constraints (no same-frame merges).
* **Face Blending:** OpenCV `seamlessClone` (Poisson clone & Mixed clone) with dynamic edge feathering and mask erosion.
* **AI Gateways:** Google Gemini API (`gemini-1.5-flash`) & Anthropic Claude API (`claude-3-5-sonnet-20240620`).

### Frontend (React + Vite + Vanilla CSS)
* **Framework:** React + Vite for quick compilation and clean modular hooks.
* **Styling:** Premium Vanilla CSS variables matching dark mode, film-black, and warm paper/ink tones.
* **Interactive Loupe:** Magnifying comparator that allows sliding to reveal original base frames underneath the composite image.
* **Pipeline Log/Receipt:** Live receipt printout displaying progress updates, person counts, gate confidence scores, and retry summaries.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.11+
* Node.js 18+
* API Keys for Gemini (`GEMINI_API_KEY`) and/or Anthropic (`ANTHROPIC_API_KEY`).

---

### 📥 Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install python-multipart
   ```

4. Create a `.env` file in the `backend` directory based on the configuration fields:
   ```env
   # API Configuration keys
   GEMINI_API_KEY=your_gemini_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here

   # Configuration flags
   DEMO_FALLBACK=False       # Set to True to bypass live LLM API calls with mock caches
   GATE_PROVIDER=gemini      # 'gemini' or 'anthropic'
   GEMINI_MODEL=gemini-1.5-flash
   ```

5. Run the FastAPI development server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   The backend API will be available at `http://localhost:8000`. API documentation is accessible at `http://localhost:8000/docs`.

---

### 💻 Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install the node modules:
   ```bash
   npm install
   ```

3. Run the Vite development server:
   ```bash
   npm run dev
   ```
   Open `http://localhost:5173` in your browser.

---

## 🐳 Deployment (Docker)

To run the application inside Docker, run the container from the repository root context to ensure python packages and model assets are bundled correctly.

1. Build the Docker image:
   ```bash
   docker build -t kairos-backend -f backend/Dockerfile .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8000:8000 -e GEMINI_API_KEY=your_key_here -e DEMO_FALLBACK=False kairos-backend
   ```

---

## 🧪 Testing

The repository contains unit and integration tests verifying all critical pipeline components.

To run the test suite:
```bash
cd backend
pytest
```

To run a manual end-to-end verification CLI script on a burst photo folder:
```bash
# Offline demo-fallback mode:
python backend/test_pipeline_day3.py test_photos

# Live API mode:
python backend/test_pipeline_day3.py test_photos --live
```

---

## 🛡️ Demo-Safety / Offline Fallback Mode
For presentations or unpredictable venue WiFi conditions, you can toggle **Offline Fallback Mode** by setting `DEMO_FALLBACK=True` in your environment variables. 
This forces the backend gates to skip live network requests and load structured cache files from the `backend/app/cache` directory.
