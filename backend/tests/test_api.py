import io
from datetime import datetime
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.pipeline.state_machine import jobs_db
from app.schemas import PipelineResult, GateResult

client = TestClient(app)

def test_upload_burst_invalid_file_count():
    """Verify that uploading less than 5 files returns HTTP 400."""
    files = [
        ("files", ("image1.jpg", io.BytesIO(b"dummy_data"), "image/jpeg")),
        ("files", ("image2.jpg", io.BytesIO(b"dummy_data"), "image/jpeg")),
    ]
    response = client.post("/api/burst", files=files)
    assert response.status_code == 400
    assert "Invalid photo count" in response.json()["detail"]

def test_upload_burst_invalid_file_type():
    """Verify that uploading non-image files returns HTTP 400."""
    files = [
        ("files", (f"file_{i}.txt", io.BytesIO(b"dummy_data"), "text/plain"))
        for i in range(5)
    ]
    response = client.post("/api/burst", files=files)
    assert response.status_code == 400
    assert "is not a valid image format" in response.json()["detail"]

@patch("app.main.coordinator.run_pipeline")
def test_upload_burst_success(mock_run_pipeline):
    """Verify that uploading 5 valid images returns HTTP 201 and creates a background task."""
    files = [
        ("files", (f"image_{i}.jpg", io.BytesIO(b"dummy_data"), "image/jpeg"))
        for i in range(5)
    ]
    
    response = client.post("/api/burst", files=files)
    assert response.status_code == 201
    
    data = response.json()
    assert "burst_id" in data
    assert "uploaded_at" in data
    assert data["photo_count"] == 5
    
    burst_id = data["burst_id"]
    # Check that job is recorded in jobs_db
    assert burst_id in jobs_db
    assert jobs_db[burst_id]["status"] == "pending"

def test_get_status_not_found():
    """Verify that status polling for a non-existent burst returns HTTP 404."""
    response = client.get("/api/burst/non-existent-id/status")
    assert response.status_code == 404
    assert "job not found" in response.json()["detail"].lower()

def test_get_status_success():
    """Verify that status polling returns correct job details."""
    burst_id = "test-status-poll-id"
    jobs_db[burst_id] = {
        "status": "processing",
        "progress_percentage": 45,
        "message": "Detecting faces...",
        "result": None
    }
    
    response = client.get(f"/api/burst/{burst_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["burst_id"] == burst_id
    assert data["status"] == "processing"
    assert data["progress_percentage"] == 45
    assert data["message"] == "Detecting faces..."

def test_get_result_not_ready():
    """Verify that fetching results for an incomplete job returns HTTP 400."""
    burst_id = "test-result-incomplete-id"
    jobs_db[burst_id] = {
        "status": "processing",
        "progress_percentage": 50,
        "message": "Processing...",
        "result": None
    }
    
    response = client.get(f"/api/burst/{burst_id}/result")
    assert response.status_code == 400
    assert "job is not complete" in response.json()["detail"].lower()

def test_get_result_success():
    """Verify that fetching results for a complete job returns the correct schemas."""
    burst_id = "test-result-complete-id"
    mock_pipeline_res = PipelineResult(
        burst_id=burst_id,
        status="complete",
        result_type="blended",
        output_image_url="data:image/jpeg;base64,mock...",
        retry_count=1,
        gate_1_result=GateResult(passed=True, confidence=1.0, issues=[]),
        gate_2_result=GateResult(passed=True, confidence=0.9, issues=[]),
        per_person_reasoning=[]
    )
    jobs_db[burst_id] = {
        "status": "complete",
        "progress_percentage": 100,
        "message": "Success",
        "result": mock_pipeline_res
    }
    
    response = client.get(f"/api/burst/{burst_id}/result")
    assert response.status_code == 200
    data = response.json()
    assert data["burst_id"] == burst_id
    assert data["status"] == "complete"
    assert data["result_type"] == "blended"
    assert data["output_image_url"] == "data:image/jpeg;base64,mock..."
    assert data["retry_count"] == 1
    assert data["gate_1_result"]["passed"] is True
    assert data["gate_2_result"]["passed"] is True

@patch("app.main.coordinator.run_pipeline")
def test_manual_retry_trigger(mock_run_pipeline):
    """Verify that manual retry re-enqueues the background job and returns pending status."""
    burst_id = "test-retry-id"
    jobs_db[burst_id] = {
        "status": "complete",
        "progress_percentage": 100,
        "message": "Previous Run Complete",
        "result": None,
        "photo_dir": "mock_photo_dir"
    }
    
    # Mock exists check
    with patch("app.main.os.path.exists", return_value=True):
        response = client.post(f"/api/burst/{burst_id}/retry")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert data["progress_percentage"] == 0
        
        # Verify job state resets
        assert jobs_db[burst_id]["status"] == "pending"
        assert jobs_db[burst_id]["result"] is None
        assert mock_run_pipeline.called
