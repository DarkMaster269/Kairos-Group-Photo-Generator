#!/usr/bin/env python3
"""
test_blender_day2.py — Full face-replacement pipeline CLI verification.

Runs the complete Kairos Day 2 pipeline end-to-end:
  detect → cluster → score → select base frame → align → blend → save

For each person cluster the script:
  1.  Finds the face instance with the highest composite expression score.
  2.  Compares it with that person's face in the chosen base frame.
  3.  If the best face is NOT already in the base frame, aligns the source
      image and Poisson-blends the face onto the evolving composite.

Outputs:
  • /test-data/composite_result.jpg  — final composite group photo.
  • /test-data/blends/<person_N>.jpg — side-by-side crop comparison per person.

Usage:
    py -X utf8 test_blender_day2.py <path_to_burst_directory>
    py -X utf8 test_blender_day2.py <path_to_burst_directory> --output_dir /path/to/out
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.pipeline.detector import FaceMeshDetector
from app.pipeline.clustering import FaceClusteringManager, _crop_face
from app.pipeline.scoring import score_face, ExpressionScore
from app.pipeline.aligner import align_face, warp_landmarks, compute_similarity_matrix
from app.pipeline.blender import (
    create_face_mask,
    blend_face,
    BLEND_NORMAL,
    BLEND_MIXED,
)
from app import config

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_blender_day2")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_image_files(directory: str) -> List[str]:
    """Return sorted list of image paths in *directory*."""
    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if not os.path.isdir(directory):
        logger.error("Not a directory: %s", directory)
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in valid_ext
    )


def _frame_group_score(frame_idx: int, clusters: List[Any]) -> float:
    """Sum of composite scores for all faces detected in *frame_idx*."""
    total = 0.0
    for cluster in clusters:
        for face in cluster.face_instances:
            if face["frame_index"] == frame_idx:
                total += face["scores"].composite_score
    return total


def _find_face_in_frame(cluster: Any, frame_idx: int) -> Optional[Dict[str, Any]]:
    """Return the face instance for *cluster* in *frame_idx*, or None."""
    for face in cluster.face_instances:
        if face["frame_index"] == frame_idx:
            return face
    return None


def _make_side_by_side(
    img_a: np.ndarray,
    img_b: np.ndarray,
    label_a: str,
    label_b: str,
    height: int = 200,
) -> np.ndarray:
    """Return a side-by-side comparison image with labels."""
    def _resize(img: np.ndarray, h: int) -> np.ndarray:
        ratio = h / max(img.shape[0], 1)
        w = max(1, int(img.shape[1] * ratio))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    a_r = _resize(img_a, height)
    b_r = _resize(img_b, height)

    divider = np.full((height, 4, 3), 80, dtype=np.uint8)
    side_by_side = np.concatenate([a_r, divider, b_r], axis=1)

    # Labels
    def _label(canvas: np.ndarray, text: str, x_offset: int) -> None:
        cv2.putText(
            canvas, text, (x_offset + 4, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
        )

    _label(side_by_side, label_a, 0)
    _label(side_by_side, label_b, a_r.shape[1] + 4)
    return side_by_side


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kairos Day 2 — Full detect → cluster → score → align → blend pipeline."
    )
    parser.add_argument("burst_dir", help="Path to burst photo directory.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Root output directory. Defaults to repo-root/test-data/.",
    )
    parser.add_argument(
        "--blend_mode",
        choices=["normal", "mixed"],
        default="normal",
        help="Poisson blend mode: 'normal' (default) or 'mixed'.",
    )
    parser.add_argument(
        "--erosion",
        type=int,
        default=config.DEFAULT_MASK_EROSION,
        help=f"Mask erosion in pixels (default: {config.DEFAULT_MASK_EROSION}).",
    )
    parser.add_argument(
        "--feather",
        type=int,
        default=config.DEFAULT_FEATHER_RADIUS,
        help=f"Mask feather/blur radius (default: {config.DEFAULT_FEATHER_RADIUS}).",
    )
    args = parser.parse_args()

    burst_dir = os.path.abspath(args.burst_dir)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_root = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(repo_root, "test-data")
    composite_path = os.path.join(output_root, "composite_result.jpg")
    blends_dir = os.path.join(output_root, "blends")
    blend_mode_flag = BLEND_NORMAL if args.blend_mode == "normal" else BLEND_MIXED

    os.makedirs(blends_dir, exist_ok=True)

    logger.info("=== Kairos Day 2 Pipeline ===")
    logger.info("Burst directory : %s", burst_dir)
    logger.info("Composite output: %s", composite_path)
    logger.info("Blend crops dir : %s", blends_dir)
    logger.info("Blend mode      : %s | erosion=%dpx | feather=%dpx",
                args.blend_mode, args.erosion, args.feather)

    # ── 1. Load images ────────────────────────────────────────────────────────
    image_paths = get_image_files(burst_dir)
    if not image_paths:
        logger.error("No images found in %s. Exiting.", burst_dir)
        sys.exit(1)
    logger.info("Found %d images to process.", len(image_paths))

    frames_images: Dict[int, np.ndarray] = {}
    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            logger.warning("Could not read %s — skipping.", path)
            continue
        frames_images[idx] = img

    # ── 2. Detect ─────────────────────────────────────────────────────────────
    frames_faces: Dict[int, List[Dict[str, Any]]] = {}
    detector = FaceMeshDetector(max_num_faces=15, refine_landmarks=True)

    try:
        for idx, img in frames_images.items():
            logger.info("Detecting faces in frame %d/%d …", idx + 1, len(frames_images))
            faces = detector.detect_faces(img)
            frames_faces[idx] = faces
            logger.info("  ↳ %d face(s) detected.", len(faces))
    finally:
        detector.close()

    if not any(frames_faces.values()):
        logger.error("No faces detected in any frame. Exiting.")
        sys.exit(1)

    # ── 3. Cluster ────────────────────────────────────────────────────────────
    logger.info("Clustering faces …")
    clustering_mgr = FaceClusteringManager()
    clusters = clustering_mgr.cluster_faces(frames_faces, frames_images)
    logger.info("Found %d distinct person cluster(s).", len(clusters))

    # ── 4. Score ──────────────────────────────────────────────────────────────
    logger.info("Scoring expressions …")
    for cluster in clusters:
        for face in cluster.face_instances:
            face["scores"] = score_face(face["landmarks"])

    # Sort each cluster's instances by composite score descending
    for cluster in clusters:
        cluster.face_instances.sort(
            key=lambda f: f["scores"].composite_score, reverse=True
        )

    # ── 5. Select base frame ──────────────────────────────────────────────────
    frame_scores = {
        idx: _frame_group_score(idx, clusters)
        for idx in frames_images
    }
    base_frame_idx = max(frame_scores, key=frame_scores.get)
    base_image = frames_images[base_frame_idx].copy()
    base_h, base_w = base_image.shape[:2]

    logger.info(
        "Base frame selected: Frame %02d (group score sum = %.4f).",
        base_frame_idx, frame_scores[base_frame_idx],
    )
    for idx, score in sorted(frame_scores.items()):
        marker = " ← BASE" if idx == base_frame_idx else ""
        logger.info("  Frame %02d group score: %.4f%s", idx, score, marker)

    # ── 6. Align + Blend each cluster onto the base frame ─────────────────────
    composite = base_image.copy()
    blend_count = 0

    print("\n" + "=" * 80)
    print(" KAIROS DAY 2 BLEND PLAN".center(80))
    print("=" * 80)

    for cluster in clusters:
        cluster_id = cluster.cluster_id
        best_face = cluster.face_instances[0]  # highest composite score
        best_frame_idx = best_face["frame_index"]
        best_score = best_face["scores"].composite_score

        # Does this person's best face already live in the base frame?
        if best_frame_idx == base_frame_idx:
            print(f"\n[{cluster_id}] Best face is already in the base frame "
                  f"(Frame {best_frame_idx:02d}, score={best_score:.4f}). Skipping blend.")
            continue

        print(f"\n[{cluster_id}] Best face: Frame {best_frame_idx:02d} "
              f"(score={best_score:.4f}) → blending onto base Frame {base_frame_idx:02d}.")

        # Locate where this person sits in the BASE frame (for mask landmarks).
        base_face = _find_face_in_frame(cluster, base_frame_idx)

        # Source: person's best-expression image + landmarks
        src_image = frames_images[best_frame_idx]
        src_landmarks = best_face["landmarks"]

        try:
            # ── Alignment ────────────────────────────────────────────────────
            if base_face is not None:
                # Happy path: we know exactly where this person sits in the
                # base frame — use those landmarks for both the transform
                # target and the mask.
                dst_landmarks = base_face["landmarks"]
                print(f"  Person detected in base frame — using base-frame landmarks for mask.")
            else:
                # Person was not detected in the base frame (e.g. they were
                # partially occluded in that shot). Estimate destination
                # landmarks by projecting the source landmarks through the
                # similarity transform derived from the overall scene.
                # We use the centroid cluster from ANY frame that overlaps with
                # the base frame to compute a representative scene transform.
                # Fall back to the cluster's second-best frame if one exists.
                fallback_face = next(
                    (f for f in cluster.face_instances[1:] if f["frame_index"] in frames_images),
                    None,
                )
                if fallback_face is None:
                    logger.warning(
                        "[%s] Only one face instance found and it is not in the base frame. "
                        "Cannot align — skipping.", cluster_id
                    )
                    print(f"  WARNING: Person not in base frame and no fallback available — skip.")
                    continue

                print(f"  Person NOT in base frame — projecting landmarks via similarity transform.")
                M = compute_similarity_matrix(src_landmarks, src_landmarks)  # identity starter
                # Compute actual transform using best-available frame pair
                try:
                    ref_frame_idx = fallback_face["frame_index"]
                    ref_image = frames_images[ref_frame_idx]
                    # Use the global scene alignment: map any source frame
                    # to the base-frame coordinate system by finding a
                    # person cluster that IS visible in both frames.
                    bridge_cluster = next(
                        (c for c in clusters
                         if _find_face_in_frame(c, best_frame_idx) is not None
                         and _find_face_in_frame(c, base_frame_idx) is not None
                         and c.cluster_id != cluster_id),
                        None,
                    )
                    if bridge_cluster:
                        bridge_src = _find_face_in_frame(bridge_cluster, best_frame_idx)
                        bridge_dst = _find_face_in_frame(bridge_cluster, base_frame_idx)
                        M = compute_similarity_matrix(
                            bridge_src["landmarks"], bridge_dst["landmarks"]
                        )
                        dst_landmarks = warp_landmarks(src_landmarks, M)
                        print(f"  Bridge cluster '{bridge_cluster.cluster_id}' used for scene transform.")
                    else:
                        logger.warning(
                            "[%s] No bridge cluster found for scene transform. "
                            "Skipping.", cluster_id
                        )
                        print(f"  WARNING: No bridge cluster available — skip.")
                        continue
                except Exception as exc:
                    logger.error("[%s] Scene transform failed: %s — skipping.", cluster_id, exc)
                    continue

            # ── Warp source image to base frame dimensions ────────────────────
            warped_src = align_face(
                src_image=src_image,
                src_landmarks=src_landmarks,
                dst_landmarks=dst_landmarks,
                output_size=(base_w, base_h),
            )

            # ── Build face mask from DESTINATION landmarks ────────────────────
            dst_lms_for_mask = base_face["landmarks"] if base_face else dst_landmarks
            mask = create_face_mask(
                base_image=composite,
                landmarks=dst_lms_for_mask,
                erosion_pixels=args.erosion,
                feather_radius=args.feather,
            )

            # Sanity check: if mask is nearly empty, skip (face too small)
            if np.count_nonzero(mask) < 100:
                logger.warning("[%s] Mask has <100 non-zero pixels — face too small. Skipping.", cluster_id)
                print(f"  WARNING: Mask too small — skip.")
                continue

            # ── Poisson blend ─────────────────────────────────────────────────
            prev_composite = composite.copy()
            composite = blend_face(
                base_image=composite,
                warped_src_image=warped_src,
                mask=mask,
                blend_mode=blend_mode_flag,
            )
            blend_count += 1
            print(f"  ✓ Blended successfully ({args.blend_mode} mode).")

            # ── Side-by-side comparison ───────────────────────────────────────
            _save_comparison(
                cluster_id=cluster_id,
                base_face=base_face,
                best_face=best_face,
                base_frame_idx=base_frame_idx,
                best_frame_idx=best_frame_idx,
                composite_before=prev_composite,
                composite_after=composite,
                blends_dir=blends_dir,
            )

        except Exception as exc:
            logger.error("[%s] Blend failed with error: %s — skipping.", cluster_id, exc, exc_info=True)
            print(f"  ERROR: {exc} — reverting to previous composite.")
            # composite is already unchanged (we copied it to prev_composite before modifying)

    print("\n" + "=" * 80)

    # ── 7. Save final composite ───────────────────────────────────────────────
    cv2.imwrite(composite_path, composite)
    logger.info("Composite result saved to %s", composite_path)
    logger.info("Total faces blended: %d / %d clusters.", blend_count, len(clusters))

    # Annotate the composite with cluster IDs and scores for review
    _save_annotated_composite(composite, clusters, base_frame_idx, composite_path)

    print(f"\nDone! {blend_count} face(s) blended.")
    print(f"  → Composite : {composite_path}")
    print(f"  → Side-by-side crops : {blends_dir}/")


# ── Output helpers ────────────────────────────────────────────────────────────

def _save_comparison(
    cluster_id: str,
    base_face: Optional[Dict[str, Any]],
    best_face: Dict[str, Any],
    base_frame_idx: int,
    best_frame_idx: int,
    composite_before: np.ndarray,
    composite_after: np.ndarray,
    blends_dir: str,
) -> None:
    """Save a side-by-side crop of the base face vs the blended face."""
    bbox = best_face["bbox"]
    base_crop = _crop_face(composite_before, bbox, padding_pct=0.6)
    blended_crop = _crop_face(composite_after, bbox, padding_pct=0.6)

    if base_crop is None or blended_crop is None:
        logger.debug("[%s] Crop region too small for comparison image.", cluster_id)
        return

    label_a = f"Base Fr{base_frame_idx:02d} ({cluster_id})"
    label_b = f"Blended Fr{best_frame_idx:02d} score={best_face['scores'].composite_score:.2f}"
    comparison = _make_side_by_side(base_crop, blended_crop, label_a, label_b, height=250)

    out_path = os.path.join(blends_dir, f"{cluster_id}_comparison.jpg")
    cv2.imwrite(out_path, comparison)
    logger.info("Saved comparison: %s", out_path)


def _save_annotated_composite(
    composite: np.ndarray,
    clusters: List[Any],
    base_frame_idx: int,
    composite_path: str,
) -> None:
    """Overlay cluster ID + score labels on the composite and save."""
    annotated = composite.copy()
    for cluster in clusters:
        best_face = cluster.face_instances[0]
        bx, by, bw, bh = best_face["bbox"]
        score = best_face["scores"].composite_score
        frame_idx = best_face["frame_index"]
        is_base = frame_idx == base_frame_idx

        colour = (0, 255, 0) if is_base else (0, 200, 255)
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), colour, 1)
        label = f"{cluster.cluster_id} | {score:.2f}"
        if not is_base:
            label += f" (fr{frame_idx:02d})"
        cv2.putText(
            annotated, label, (bx, max(by - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA,
        )

    annotated_path = composite_path.replace(".jpg", "_annotated.jpg")
    cv2.imwrite(annotated_path, annotated)
    logger.info("Annotated composite saved to %s", annotated_path)


if __name__ == "__main__":
    main()
