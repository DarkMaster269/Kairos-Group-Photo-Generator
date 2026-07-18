import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from app import config
from app.pipeline.state_machine import PipelineCoordinator, jobs_db
from app.schemas import GateResult, GateIssue, PipelineResult
from app.pipeline.clustering import PersonCluster

@patch("app.pipeline.state_machine.os.listdir")
@patch("app.pipeline.state_machine.cv2.imread")
@patch("app.pipeline.state_machine.cv2.imencode")
@patch("app.pipeline.state_machine.FaceMeshDetector")
@patch("app.pipeline.state_machine.FaceClusteringManager")
@patch("app.pipeline.state_machine.align_face")
@patch("app.pipeline.state_machine.create_face_mask")
@patch("app.pipeline.state_machine.blend_face")
@patch("app.pipeline.state_machine.AIGateways")
def test_pipeline_coordinator_retry_loop_fallback(
    mock_ai_gateways,
    mock_blend_face,
    mock_create_face_mask,
    mock_align_face,
    mock_clustering_mgr,
    mock_detector,
    mock_imencode,
    mock_imread,
    mock_listdir
):
    """Verify that the PipelineCoordinator retry loop works and correctly falls back after max retries."""
    burst_id = "test-mock-burst-retry-loop"
    photo_dir = "mock_dir"
    
    # 1. Mock file system files
    mock_listdir.return_value = ["frame_00.jpg", "frame_01.jpg", "frame_02.jpg"]
    mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Mock imencode to return success
    mock_imencode.return_value = (True, b"dummy_jpeg_bytes")
    
    # 2. Mock AI Gateways class instances and responses
    gateways_inst = MagicMock()
    mock_ai_gateways.return_value = gateways_inst
    
    # Gate 1 always passes
    gateways_inst.check_gate1_inputs.return_value = GateResult(passed=True, confidence=1.0, issues=[])
    
    # Gate 2 fails three times in a row (Attempt 0, Attempt 1, Attempt 2)
    gate2_fail_res = GateResult(
        passed=False,
        confidence=0.4,
        issues=[GateIssue(person_cluster_id="person_0", issue_type="seam_artifact", description="Visible seam")]
    )
    gateways_inst.check_gate2_output.return_value = gate2_fail_res
    
    # 3. Mock face mesh detector
    detector_inst = MagicMock()
    mock_detector.return_value = detector_inst
    detector_inst.detect_faces.return_value = [
        {"bbox": (10, 10, 20, 20), "landmarks": [(0.0, 0.0, 0.0)] * 478}
    ]
    
    # 4. Mock face clustering manager
    clustering_inst = MagicMock()
    mock_clustering_mgr.return_value = clustering_inst
    
    # Set up clusters (two people)
    c0 = PersonCluster(
        cluster_id="person_0",
        face_instances=[
            # Person 0 has multiple candidates (best from frame 0, next from frame 1, etc.)
            {"bbox": (10, 10, 20, 20), "landmarks": [(0.0, 0.0, 0.0)] * 478, "frame_index": 0},
            {"bbox": (10, 10, 20, 20), "landmarks": [(0.0, 0.0, 0.0)] * 478, "frame_index": 1},
            {"bbox": (10, 10, 20, 20), "landmarks": [(0.0, 0.0, 0.0)] * 478, "frame_index": 2},
        ]
    )
    c1 = PersonCluster(
        cluster_id="person_1",
        face_instances=[
            {"bbox": (40, 40, 20, 20), "landmarks": [(0.0, 0.0, 0.0)] * 478, "frame_index": 0}
        ]
    )
    clustering_inst.cluster_faces.return_value = [c0, c1]
    
    # 5. Instantiate and run coordinator
    coordinator = PipelineCoordinator()
    coordinator.run_pipeline(burst_id, photo_dir)
    
    # Verify the results in jobs_db
    assert burst_id in jobs_db
    job = jobs_db[burst_id]
    
    # Overall status should be complete (completed the run, even if fallback was chosen)
    assert job["status"] == "complete"
    assert job["progress_percentage"] == 100
    
    result: PipelineResult = job["result"]
    assert result is not None
    
    # Since Gate 2 failed 3 times, result_type must be "fallback_single_frame"
    assert result.result_type == "fallback_single_frame"
    assert result.status == "fallback"
    
    # It must have performed exactly 2 retries (3 total attempts)
    assert result.retry_count == 2
    
    # Verify Gate 2 failed in the final report
    assert result.gate_2_result.passed is False
    assert len(result.gate_2_result.issues) == 1
    assert result.gate_2_result.issues[0].person_cluster_id == "person_0"
    
    # Ensure check_gate2_output was called 3 times (attempt 0, 1, 2)
    assert gateways_inst.check_gate2_output.call_count == 3
