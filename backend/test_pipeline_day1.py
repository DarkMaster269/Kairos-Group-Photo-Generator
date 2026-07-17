#!/usr/bin/env python3
"""
test_pipeline_day1.py — CLI validation script for the Day 1 Core Pipeline.

This script runs face detection, cross-frame face clustering, and expression
scoring on a folder of burst photos. It outputs a summary of findings to the
console and saves cropped faces grouped by identity into `/test-data/output_clusters/`.

Usage:
    python test_pipeline_day1.py <path_to_burst_directory>
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Any

import cv2
import numpy as np

# Adjust python path to allow importing app package
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.pipeline.detector import FaceMeshDetector
from app.pipeline.clustering import FaceClusteringManager, _crop_face
from app.pipeline.scoring import score_face, ExpressionScore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("test_pipeline_day1")


def get_image_files(directory: str) -> List[str]:
    """Find and return all image files in the directory sorted alphabetically."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if not os.path.isdir(directory):
        logger.error("Provided path is not a directory: %s", directory)
        return []
    
    files = []
    for file in os.listdir(directory):
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_extensions:
            files.append(os.path.join(directory, file))
            
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Verify Face Detection, Clustering, and Scoring on a photo burst."
    )
    parser.add_argument(
        "burst_dir",
        type=str,
        help="Path to the directory containing the burst photos (JPG/PNG)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output directory for face crops. Defaults to repo-root/test-data/output_clusters/"
    )
    args = parser.parse_args()

    # 1. Resolve and validate directory paths
    burst_dir = os.path.abspath(args.burst_dir)
    logger.info("Starting Day 1 Pipeline Verification for burst: %s", burst_dir)
    
    image_paths = get_image_files(burst_dir)
    if not image_paths:
        logger.error("No valid image files found in %s. Exiting.", burst_dir)
        sys.exit(1)
        
    logger.info("Found %d images to process.", len(image_paths))

    # Resolve output directory
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        # Default to repo_root/test-data/output_clusters/
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(repo_root, "test-data", "output_clusters")
    
    logger.info("Crops will be saved to: %s", output_dir)

    # 2. Run Face Detection per frame
    frames_faces: Dict[int, List[Dict[str, Any]]] = {}
    frames_images: Dict[int, np.ndarray] = {}
    
    detector = FaceMeshDetector(max_num_faces=15, refine_landmarks=True)
    
    try:
        for idx, path in enumerate(image_paths):
            logger.info("Processing frame %d/%d: %s", idx + 1, len(image_paths), os.path.basename(path))
            img = cv2.imread(path)
            if img is None:
                logger.warning("Could not read image %s. Skipping.", path)
                continue
                
            faces = detector.detect_faces(img)
            frames_images[idx] = img
            frames_faces[idx] = faces
            logger.info("  ↳ Detected %d face(s) in frame %d.", len(faces), idx)
    finally:
        detector.close()

    if not any(frames_faces.values()):
        logger.error("No faces detected in any frame. Cannot cluster. Exiting.")
        sys.exit(1)

    # 3. Run Face Clustering
    logger.info("Clustering faces across all frames...")
    clustering_mgr = FaceClusteringManager()
    clusters = clustering_mgr.cluster_faces(frames_faces, frames_images)
    
    if not clusters:
        logger.error("Face clustering failed or returned no clusters. Exiting.")
        sys.exit(1)

    logger.info("Clustering completed. Found %d distinct identities.", len(clusters))

    # 4. Perform Face Scoring
    logger.info("Scoring face expressions...")
    for cluster in clusters:
        for face in cluster.face_instances:
            landmarks = face["landmarks"]
            # Compute EAR, smile, gaze scores
            score: ExpressionScore = score_face(landmarks)
            # Store score metrics in the face dictionary
            face["scores"] = score

    # 5. Save Face Crops and Print Summary
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print(" KAIROS PIPELINE DAY 1 SUMMARY".center(80))
    print("=" * 80)
    print(f"Burst Directory: {burst_dir}")
    print(f"Total Frames Processed: {len(image_paths)}")
    print(f"Identified Unique Clusters (People): {len(clusters)}")
    print("-" * 80)

    for cluster in clusters:
        cluster_id = cluster.cluster_id
        instances = cluster.face_instances
        
        # Sort face instances by composite score descending
        # Ensure instances have score key
        instances.sort(key=lambda x: x["scores"].composite_score, reverse=True)
        
        best_instance = instances[0]
        best_frame = best_instance["frame_index"]
        best_score = best_instance["scores"].composite_score
        
        print(f"\nCluster ID: {cluster_id} ({len(instances)} instances)")
        print(f"  RECOMMENDED PICK: Frame {best_frame:02d} (Score: {best_score:.4f})")
        print("  All Instances:")
        
        # Create output directory for this cluster
        cluster_output_dir = os.path.join(output_dir, cluster_id)
        os.makedirs(cluster_output_dir, exist_ok=True)
        
        for f_idx, face in enumerate(instances):
            frame_idx = face["frame_index"]
            scores = face["scores"]
            face_id = face["face_id"]
            
            is_pick = "★ WINNER ★" if frame_idx == best_frame else ""
            print(f"    - Frame {frame_idx:02d} | Face ID: {face_id} | "
                  f"Composite: {scores.composite_score:.4f} "
                  f"(Eyes: {scores.eyes_open:.2f}, Smile: {scores.smile:.2f}, Gaze: {scores.gaze_forward:.2f}) {is_pick}")
            
            # Crop and save crop image for manual review
            original_image = frames_images[frame_idx]
            crop = _crop_face(original_image, face["bbox"])
            if crop is not None:
                # Save crop
                crop_filename = f"frame_{frame_idx:02d}_score_{scores.composite_score:.2f}.jpg"
                crop_path = os.path.join(cluster_output_dir, crop_filename)
                
                # Draw small overlay score on crop for visual review
                annotated_crop = crop.copy()
                label = f"Frame {frame_idx} | {scores.composite_score:.2f}"
                cv2.putText(
                    annotated_crop, label, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA
                )
                cv2.imwrite(crop_path, annotated_crop)

    print("=" * 80)
    logger.info("Day 1 verification complete. Visual face crops saved to %s", output_dir)


if __name__ == "__main__":
    main()
