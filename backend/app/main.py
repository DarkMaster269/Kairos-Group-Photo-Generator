import os
import shutil
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Lazy import of pipeline modules to prevent boot startup delay
# Lazy import of pipeline modules to prevent boot startup delay
from app.pipeline.state_machine import PipelineCoordinator, jobs_db
from app.schemas import (
    ExpressionScore,
    FaceInstance,
    GateIssue,
    GateResult,
    PipelineResult,
    StatusResponse,
    BurstUploadResponse,
)

app = FastAPI(
    title="Kairos API",
    description="Backend API for the Perfect Group Photo Generator, including AI gates and seamless blending.",
    version="0.1.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

coordinator = PipelineCoordinator()

# Helper path for temporary uploads
TEMP_BURSTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "temp_bursts"
)


# ==============================================================================
# Endpoint Implementations
# ==============================================================================

@app.post("/api/burst", response_model=BurstUploadResponse, status_code=201)
async def upload_burst(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Burst of 5 to 15 group photographs")
):
    """
    Upload a burst of photos (5 to 15 images) taken seconds apart of the same group.
    Initiates the background processing pipeline and returns a burst_id immediately.
    """
    if len(files) < 5 or len(files) > 15:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid photo count. Uploaded {len(files)} photos. Expected between 5 and 15 photos."
        )

    # Generate a unique burst identifier
    burst_id = str(uuid.uuid4())
    
    # Check that files are images and save them to a temp folder
    temp_dir = os.path.join(TEMP_BURSTS_DIR, burst_id)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        for idx, file in enumerate(files):
            if not file.content_type.startswith("image/"):
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not a valid image format."
                )
            
            # Secure file names (e.g. frame_00.jpg, frame_01.jpg etc.)
            ext = os.path.splitext(file.filename)[1]
            if not ext:
                ext = ".jpg"
            file_path = os.path.join(temp_dir, f"frame_{idx:02d}{ext}")
            
            with open(file_path, "wb") as f_out:
                shutil.copyfileobj(file.file, f_out)
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded files: {e}"
        )

    # Initialise job entry in jobs_db
    jobs_db[burst_id] = {
        "status": "pending",
        "progress_percentage": 0,
        "message": "Burst photos received. Queueing processing...",
        "result": None,
        "photo_dir": temp_dir  # Save path for manual retry triggers
    }

    # Dispatch to background executor
    background_tasks.add_task(coordinator.run_pipeline, burst_id, temp_dir)

    return BurstUploadResponse(
        burst_id=burst_id,
        uploaded_at=datetime.utcnow(),
        photo_count=len(files)
    )


@app.get("/api/burst/{burst_id}/status", response_model=StatusResponse)
async def get_burst_status(burst_id: str):
    """
    Retrieve the status or streaming progress of a specific burst job.
    """
    if burst_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Burst job not found.")
        
    job = jobs_db[burst_id]
    return StatusResponse(
        burst_id=burst_id,
        status=job["status"],
        progress_percentage=job["progress_percentage"],
        message=job["message"]
    )


@app.get("/api/burst/{burst_id}/result", response_model=PipelineResult)
async def get_burst_result(burst_id: str):
    """
    Retrieve the final PipelineResult for a completed burst job.
    """
    if burst_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Burst job not found.")
        
    job = jobs_db[burst_id]
    if job["status"] not in ("complete", "error", "fallback"):
        raise HTTPException(
            status_code=400,
            detail=f"Burst job is not complete. Current status: {job['status']}"
        )
        
    if job["result"] is None:
        raise HTTPException(
            status_code=500,
            detail="Job completed but no pipeline result was generated."
        )

    return job["result"]


@app.post("/api/burst/{burst_id}/retry", response_model=StatusResponse)
async def retry_burst_blend(burst_id: str, background_tasks: BackgroundTasks):
    """
    Manually force/re-trigger a pipeline retry with alternative parameters.
    """
    if burst_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Burst job not found.")
        
    job = jobs_db[burst_id]
    photo_dir = job.get("photo_dir")
    
    if not photo_dir or not os.path.exists(photo_dir):
        raise HTTPException(
            status_code=400,
            detail="Uploaded photos are no longer available for this job."
        )

    # Re-queue the background task
    job["status"] = "pending"
    job["progress_percentage"] = 0
    job["message"] = "Manually triggered retry. Queueing processing..."
    job["result"] = None

    background_tasks.add_task(coordinator.run_pipeline, burst_id, photo_dir)

    return StatusResponse(
        burst_id=burst_id,
        status="pending",
        progress_percentage=0,
        message="Manual retry scheduled in background."
    )
