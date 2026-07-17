"""
clustering.py — Spatial tracking-based face clustering across burst frames.

Groups detected face instances from multiple frames into consistent per-person
identities. Because subject spatial position remains highly stable throughout
a short photo burst sequence, face instances are clustered based on the 2D
Euclidean distance between their bounding box center coordinates (x_center, y_center).

Applies a Frame Disjointness Constraint: two clusters can never be merged
if they both contain face instances from the same frame index (since a single
physical person can only appear once per frame).

Part of the Kairos CV pipeline: detect → **cluster** → score → blend → gate.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class PersonCluster:
    """A group of face instances identified as the same person across frames.

    Attributes:
        cluster_id:     Stable identifier, e.g. ``"person_0"``.
        face_instances: Face dicts from the detector, each augmented with
                        ``face_id``, ``person_cluster_id``, and ``frame_index`` keys.
    """

    cluster_id: str
    face_instances: List[Dict[str, Any]] = field(default_factory=list)


# ── Spatial Clustering Manager ───────────────────────────────────────────────

class FaceClusteringManager:
    """Groups detected faces across burst frames into per-person clusters.

    Uses bounding box center coordinate Euclidean distances for clustering,
    augmented with a Frame Disjointness Constraint to prevent adjacent people
    from being merged.

    Args:
        threshold:  Maximum Euclidean distance in pixels between cluster centers
                    to allow merging. Defaults to 45.0.
    """

    def __init__(
        self,
        threshold: float = 45.0,
    ) -> None:
        self._threshold = threshold
        logger.debug("FaceClusteringManager spatial init (threshold=%.1f px)", threshold)

    def cluster_faces(
        self,
        frames_faces: Dict[int, List[Dict[str, Any]]],
        frames_images: Dict[int, np.ndarray],
    ) -> List[PersonCluster]:
        """Cluster detected face instances across frames based on spatial coordinates.

        Args:
            frames_faces:  Mapping of ``frame_index → [face_dict, ...]``.
            frames_images: Mapping of ``frame_index → BGR image ndarray`` (retained
                           for API compatibility).

        Returns:
            A list of :class:`PersonCluster` objects.
        """
        # 1. Flatten all face instances with their metadata and centers
        flat_instances: List[Dict[str, Any]] = []
        for frame_index, face_dicts in frames_faces.items():
            for face_idx, face_dict in enumerate(face_dicts):
                x, y, w, h = face_dict["bbox"]
                center_x = x + w / 2.0
                center_y = y + h / 2.0
                
                flat_instances.append({
                    "frame_index": frame_index,
                    "face_index": face_idx,
                    "center": (center_x, center_y),
                    "face_dict": face_dict
                })

        n = len(flat_instances)
        if n == 0:
            logger.warning("No faces provided for clustering — returning empty list.")
            return []

        # 2. Build pairwise Euclidean distance matrix
        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                pt1 = flat_instances[i]["center"]
                pt2 = flat_instances[j]["center"]
                d = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5
                dist[i][j] = d
                dist[j][i] = d

        # 3. Agglomerative clustering with Frame Disjointness Constraint
        # labels[i] holds the current cluster ID for instance i
        labels = list(range(n))

        while True:
            unique_ids = sorted(set(labels))
            if len(unique_ids) <= 1:
                break

            best_dist = float("inf")
            merge_a = merge_b = -1

            for ci_idx, ci in enumerate(unique_ids):
                members_i = [k for k in range(n) if labels[k] == ci]
                frames_i = {flat_instances[mi]["frame_index"] for mi in members_i}
                
                for cj in unique_ids[ci_idx + 1:]:
                    members_j = [k for k in range(n) if labels[k] == cj]
                    frames_j = {flat_instances[mj]["frame_index"] for mj in members_j}

                    # Frame Disjointness Constraint:
                    # Do not merge if the two clusters share any frame index
                    if not frames_i.isdisjoint(frames_j):
                        continue

                    # Average linkage distance
                    total = sum(dist[mi][mj] for mi in members_i for mj in members_j)
                    avg_dist = total / (len(members_i) * len(members_j))

                    if avg_dist < best_dist:
                        best_dist = avg_dist
                        merge_a, merge_b = ci, cj

            # Break if no more merges are possible or the best distance exceeds the threshold
            if best_dist >= self._threshold or merge_a == -1:
                break

            # Perform merge: absorb merge_b into merge_a
            for k in range(n):
                if labels[k] == merge_b:
                    labels[k] = merge_a

        # 4. Group instances into PersonCluster structures
        groups: Dict[int, List[Dict[str, Any]]] = {}
        for idx, lbl in enumerate(labels):
            groups.setdefault(lbl, []).append(flat_instances[idx])

        clusters: List[PersonCluster] = []
        for cluster_num, members in enumerate(groups.values()):
            cluster_id = f"person_{cluster_num}"
            pc = PersonCluster(cluster_id=cluster_id)
            
            for m in members:
                face_dict = m["face_dict"]
                frame_idx = m["frame_index"]
                face_idx = m["face_index"]
                
                # Augment the face dict in-place
                face_dict["face_id"] = f"face_{frame_idx}_{face_idx}"
                face_dict["person_cluster_id"] = cluster_id
                face_dict["frame_index"] = frame_idx
                
                pc.face_instances.append(face_dict)
                
            clusters.append(pc)

        logger.info(
            "Clustered %d face instances into %d person(s) using spatial coordinate tracking.",
            n, len(clusters)
        )
        return clusters


# ── Module Level Utility ─────────────────────────────────────────────────────

def _crop_face(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding_pct: float = 0.40,
) -> Any:
    """Crop a face from image with extra padding, clamped to image bounds.

    Used by CLI scripts to export face crops for visual validation.
    """
    x, y, w, h = bbox
    pad_w = int(w * padding_pct)
    pad_h = int(h * padding_pct)
    img_h, img_w = image.shape[:2]

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w < 20 or crop_h < 20:
        return None

    return image[y1:y2, x1:x2]
