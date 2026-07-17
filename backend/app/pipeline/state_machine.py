"""
state_machine.py — Core pipeline runner and retry state machine.

Coordinates the end-to-end Kairos CV pipeline:
  1. Input Validation (Gate 1).
  2. CV Core Pipeline (detect, cluster, score).
  3. Image Compositing (align and Poisson blend).
  4. Blending Quality Check (Gate 2).
  5. Retry Loop (swapping faces, adjusting parameters).
  6. Fallback (returning best single unedited frame).

Maintains job statuses in an in-memory database dictionary (jobs_db)
to allow status polling.
"""

import base64
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app import config
from app.schemas import GateResult, GateIssue, PipelineResult
from app.pipeline.detector import FaceMeshDetector
from app.pipeline.clustering import FaceClusteringManager
from app.pipeline.scoring import score_face
from app.pipeline.aligner import align_face, warp_landmarks, compute_similarity_matrix
from app.pipeline.blender import (
    create_face_mask,
    blend_face,
    BLEND_NORMAL,
    BLEND_MIXED,
)
from app.pipeline.gates import AIGateways

logger = logging.getLogger(__name__)

# ── In-Memory Job Database ───────────────────────────────────────────────────
# Format: { burst_id: { "status": str, "progress_percentage": int, "message": str, "result": PipelineResult } }
jobs_db: Dict[str, Dict[str, Any]] = {}


class PipelineCoordinator:
    """Manages the lifecycle of a burst group photo optimization job."""

    def __init__(self) -> None:
        self.gateways = AIGateways()

    def run_pipeline(self, burst_id: str, photo_dir: str) -> None:
        """Run the optimization pipeline in a background thread.

        Updates the status in `jobs_db` as progress is made.

        Args:
            burst_id:   Unique identifier for this burst run.
            photo_dir:  Directory containing the saved uploaded images.
        """
        logger.info("[%s] Beginning burst pipeline execution.", burst_id)
        self._update_status(burst_id, "processing", 10, "Validating burst photos (Gate 1)...")

        # Resolve image file paths
        valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_paths = sorted([
            os.path.join(photo_dir, f)
            for f in os.listdir(photo_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        ])

        if not image_paths:
            self._fail_job(burst_id, "No valid image files found in the uploaded burst.")
            return

        # ── Step 1: Gate 1 Validation ─────────────────────────────────────────
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)

        gate1_res = self.gateways.check_gate1_inputs(images)
        if not gate1_res.passed:
            logger.warning("[%s] Gate 1 input validation failed.", burst_id)
            # Create a failed result
            result = PipelineResult(
                burst_id=burst_id,
                status="error",
                result_type=None,
                output_image_url=None,
                retry_count=0,
                gate_1_result=gate1_res,
                gate_2_result=None,
                per_person_reasoning=[]
            )
            jobs_db[burst_id] = {
                "status": "error",
                "progress_percentage": 100,
                "message": f"Validation failed: {', '.join([i.description for i in gate1_res.issues])}",
                "result": result
            }
            return

        # ── Step 2: Face Detection, Clustering & Scoring ─────────────────────
        self._update_status(burst_id, "processing", 30, "Detecting faces across frames...")
        
        frames_faces: Dict[int, List[Dict[str, Any]]] = {}
        frames_images: Dict[int, np.ndarray] = {}
        detector = FaceMeshDetector(max_num_faces=15, refine_landmarks=True)

        try:
            for idx, img in enumerate(images):
                faces = detector.detect_faces(img)
                frames_faces[idx] = faces
                frames_images[idx] = img
        except Exception as e:
            logger.error("[%s] Face detection failed: %s", burst_id, e)
            self._fail_job(burst_id, f"Face detection failed: {e}")
            return
        finally:
            detector.close()

        if not any(frames_faces.values()):
            self._fail_job(burst_id, "No faces detected in any frame.")
            return

        self._update_status(burst_id, "processing", 50, "Grouping identities and scoring expressions...")
        clustering_mgr = FaceClusteringManager()
        clusters = clustering_mgr.cluster_faces(frames_faces, frames_images)

        # Score all faces
        for cluster in clusters:
            for face in cluster.face_instances:
                face["scores"] = score_face(face["landmarks"])
            # Sort instances: best first
            cluster.face_instances.sort(
                key=lambda f: f["scores"].composite_score, reverse=True
            )

        # Base frame selection (highest group score sum)
        frame_scores = {}
        for idx in frames_images:
            score_sum = sum(
                face["scores"].composite_score
                for c in clusters
                for face in c.face_instances
                if face["frame_index"] == idx
            )
            frame_scores[idx] = score_sum

        base_frame_idx = max(frame_scores, key=frame_scores.get)
        base_image = frames_images[base_frame_idx]
        base_h, base_w = base_image.shape[:2]

        logger.info("[%s] Base frame selected: Frame %02d (Score sum: %.3f)",
                    burst_id, base_frame_idx, frame_scores[base_frame_idx])

        # ── Step 3: State Machine Blending and Gate 2 Retry Loop ──────────────
        self._update_status(burst_id, "processing", 70, "Creating composite photo blend...")

        # Maintain index pointer of which face to use for each cluster (0 = best, 1 = next-best, etc.)
        cluster_candidate_ptrs: Dict[str, int] = {c.cluster_id: 0 for c in clusters}
        
        # Blending parameters (overridden dynamically during retries)
        erosion_pixels = config.DEFAULT_MASK_EROSION
        feather_radius = config.DEFAULT_FEATHER_RADIUS
        blend_mode = BLEND_NORMAL

        attempt = 0
        gate2_res = None
        composite = None

        while attempt <= config.MAX_RETRIES:
            logger.info("[%s] Blending attempt %d/%d", burst_id, attempt, config.MAX_RETRIES)
            if attempt > 0:
                self._update_status(burst_id, "processing", 80, f"Retrying blend (attempt {attempt})...")

            try:
                composite = base_image.copy()
                
                # Perform blends for clusters where the selected candidate is NOT in the base frame
                for cluster in clusters:
                    cluster_id = cluster.cluster_id
                    ptr = cluster_candidate_ptrs[cluster_id]
                    
                    # Safety check if candidates run out
                    if ptr >= len(cluster.face_instances):
                        # Fallback to base frame face (i.e. do not warp or blend)
                        logger.info("[%s] Cluster %s ran out of candidates. Using base frame face.", burst_id, cluster_id)
                        continue

                    selected_face = cluster.face_instances[ptr]
                    best_frame_idx = selected_face["frame_index"]

                    # Skip blend if the chosen candidate is already in the base frame
                    if best_frame_idx == base_frame_idx:
                        continue

                    # Retrieve destination alignment landmarks
                    base_face = next((f for f in cluster.face_instances if f["frame_index"] == base_frame_idx), None)
                    src_image = frames_images[best_frame_idx]
                    src_landmarks = selected_face["landmarks"]

                    if base_face is not None:
                        dst_landmarks = base_face["landmarks"]
                    else:
                        # Fallback scene warp bridge
                        bridge = next(
                            (c for c in clusters
                             if any(f["frame_index"] == best_frame_idx for f in c.face_instances)
                             and any(f["frame_index"] == base_frame_idx for f in c.face_instances)
                             and c.cluster_id != cluster_id),
                            None
                        )
                        if bridge:
                            b_src = next(f for f in bridge.face_instances if f["frame_index"] == best_frame_idx)
                            b_dst = next(f for f in bridge.face_instances if f["frame_index"] == base_frame_idx)
                            M = compute_similarity_matrix(b_src["landmarks"], b_dst["landmarks"])
                            dst_landmarks = warp_landmarks(src_landmarks, M)
                        else:
                            continue

                    # Warp and blend
                    warped = align_face(src_image, src_landmarks, dst_landmarks, (base_w, base_h))
                    mask = create_face_mask(composite, dst_landmarks, erosion_pixels, feather_radius)
                    
                    if np.count_nonzero(mask) > 100:
                        composite = blend_face(composite, warped, mask, blend_mode)

                # Call Gate 2
                self._update_status(burst_id, "processing", 85, "Checking blend quality (Gate 2)...")
                gate2_res = self.gateways.check_gate2_output(composite, base_image)

                if gate2_res.passed:
                    logger.info("[%s] Gate 2 verification passed at attempt %d.", burst_id, attempt)
                    break
                
                # If Gate 2 failed, prepare retry parameters
                attempt += 1
                if attempt <= config.MAX_RETRIES:
                    logger.warning("[%s] Gate 2 rejected composite. Adjusting parameters for retry.", burst_id)
                    
                    # Adapt face selections and blend parameters based on the flagged issues
                    for issue in gate2_res.issues:
                        flagged_id = issue.person_cluster_id
                        
                        # 1. Swap face candidate if cluster is explicitly flagged
                        if flagged_id and flagged_id in cluster_candidate_ptrs:
                            cluster_candidate_ptrs[flagged_id] += 1
                            logger.info("[%s] Swapping %s to next-best face (candidate index %d)",
                                        burst_id, flagged_id, cluster_candidate_ptrs[flagged_id])

                        # 2. Tune blending constants based on issue type
                        if issue.issue_type == "seam_artifact":
                            erosion_pixels += 3
                            feather_radius += 5
                            logger.info("[%s] Flagged seam_artifact: increasing erosion to %dpx, feather to %dpx",
                                        burst_id, erosion_pixels, feather_radius)
                        elif issue.issue_type == "lighting_mismatch":
                            blend_mode = BLEND_MIXED
                            logger.info("[%s] Flagged lighting_mismatch: switching blend mode to MIXED_CLONE",
                                        burst_id, blend_mode)

            except Exception as blend_err:
                logger.error("[%s] Blend attempt %d failed: %s", burst_id, attempt, blend_err)
                attempt += 1

        # ── Step 4: Finalize or Fallback ──────────────────────────────────────
        self._update_status(burst_id, "processing", 95, "Finalizing output image...")
        
        result_type = "blended"
        final_image = composite

        # Execute fallback if all retry attempts failed Gate 2 validation
        if gate2_res is None or not gate2_res.passed:
            logger.warning("[%s] Retries exhausted without Gate 2 pass. Executing fallback to best single frame.", burst_id)
            result_type = "fallback_single_frame"
            final_image = frames_images[base_frame_idx]

        # Base64 encode the final image to send back in response
        _, buffer = cv2.imencode(".jpg", final_image)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_base64}"

        # Build reasoning info
        reasoning = []
        for cluster in clusters:
            best_face = cluster.face_instances[0]
            reasoning.append({
                "cluster_id": cluster.cluster_id,
                "selected_frame": best_face["frame_index"],
                "reason": f"Best facial metrics (composite score {best_face['scores'].composite_score:.2f})."
            })

        # Save success result
        pipeline_result = PipelineResult(
            burst_id=burst_id,
            status="complete" if result_type == "blended" else "fallback",
            result_type=result_type,
            output_image_url=data_url,
            retry_count=min(attempt, config.MAX_RETRIES),
            gate_1_result=gate1_res,
            gate_2_result=gate2_res,
            per_person_reasoning=reasoning
        )

        jobs_db[burst_id] = {
            "status": "complete",
            "progress_percentage": 100,
            "message": "Optimization complete!" if result_type == "blended" else "Returned best unedited single frame.",
            "result": pipeline_result
        }

        # Retain temporary photo upload directory so that the manual /retry endpoint remains functional
        jobs_db[burst_id]["photo_dir"] = photo_dir

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _update_status(self, burst_id: str, status: str, progress: int, message: str) -> None:
        """Update jobs_db status tracking information."""
        jobs_db[burst_id] = {
            "status": status,
            "progress_percentage": progress,
            "message": message,
            "result": None
        }

    def _fail_job(self, burst_id: str, error_message: str) -> None:
        """Helper to mark job as failed."""
        logger.error("[%s] Job failed: %s", burst_id, error_message)
        result = PipelineResult(
            burst_id=burst_id,
            status="error",
            result_type=None,
            output_image_url=None,
            retry_count=0,
            gate_1_result=None,
            gate_2_result=None,
            per_person_reasoning=[]
        )
        jobs_db[burst_id] = {
            "status": "error",
            "progress_percentage": 100,
            "message": error_message,
            "result": result
        }
