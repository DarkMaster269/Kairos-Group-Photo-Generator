from datetime import datetime
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

class ExpressionScore(BaseModel):
    eyes_open: float = Field(..., ge=0.0, le=1.0, description="Eyes openness metric, 0 to 1")
    smile: float = Field(..., ge=0.0, le=1.0, description="Smile expression metric, 0 to 1")
    gaze_forward: float = Field(..., ge=0.0, le=1.0, description="Gaze alignment metric, 0 to 1")
    composite_score: float = Field(..., ge=0.0, le=1.0, description="Weighted composite score")


class FaceInstance(BaseModel):
    face_id: str
    frame_index: int
    person_cluster_id: str
    bbox: Tuple[int, int, int, int] = Field(..., description="(x_min, y_min, width, height) in pixels")
    landmarks: List[Tuple[float, float, float]] = Field(..., description="MediaPipe Face Mesh landmark coordinates")
    scores: ExpressionScore


class GateIssue(BaseModel):
    person_cluster_id: Optional[str] = None
    issue_type: str = Field(..., description="E.g., seam_artifact, warped_feature, lighting_mismatch, scene_mismatch")
    description: str


class GateResult(BaseModel):
    passed: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    issues: List[GateIssue] = []
    person_count_estimate: Optional[int] = Field(None, description="Gate 1 only: estimated number of people in burst")


class PipelineResult(BaseModel):
    burst_id: str
    status: str = Field(..., description="processing | complete | fallback | error")
    result_type: Optional[str] = Field(None, description="blended | fallback_single_frame")
    output_image_url: Optional[str] = None
    retry_count: int = 0
    gate_1_result: Optional[GateResult] = None
    gate_2_result: Optional[GateResult] = None
    per_person_reasoning: List[dict] = Field([], description="Explanations for why each face was picked")


class StatusResponse(BaseModel):
    burst_id: str
    status: str = Field(..., description="pending | processing | complete | error | fallback")
    progress_percentage: int = Field(0, ge=0, le=100)
    message: str = Field("", description="Human readable stage status description")


class BurstUploadResponse(BaseModel):
    burst_id: str
    uploaded_at: datetime
    photo_count: int
