#!/usr/bin/env python3
"""
Standalone YOLO inference script for microPAD quad detection using pose keypoints.

Called by MATLAB cut_micropads.m to perform AI-based polygon detection.
Accepts image path and outputs detected quad coordinates to stdout.

This script uses YOLOv11-pose to detect quadrilateral concentration zones on microPAD
images by predicting 4 corner keypoints directly. No polygon simplification is needed.

Usage:
    python detect_quads.py <image_path> <model_path> [--conf THRESHOLD] [--imgsz SIZE]

Output Format (stdout):
    numDetections
    x1 y1 x2 y2 x3 y3 x4 y4 confidence
    x1 y1 x2 y2 x3 y3 x4 y4 confidence
    ...

Note: Coordinates are 0-based (Python/OpenCV convention).
      MATLAB code must add 1 for 1-based indexing.
      Keypoints are ordered clockwise from top-left: TL, TR, BR, BL.
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, List
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLO pose-based quad detection for microPAD analysis"
    )
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("model_path", type=str, help="Path to YOLO pose model (.pt)")
    parser.add_argument(
        "--conf", type=float, default=0.6, help="Confidence threshold (default: 0.6)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size (default: 640)"
    )
    return parser.parse_args()


def order_corners_clockwise(quad: np.ndarray) -> np.ndarray:
    """Order vertices clockwise starting from top-left.

    Top-left is defined as the corner with minimum (x + y).
    Clockwise order from that corner ensures consistent vertex ordering.

    Args:
        quad: 4x2 numpy array of corner coordinates

    Returns:
        4x2 numpy array ordered clockwise from top-left (TL, TR, BR, BL)
    """
    # Find top-left corner (minimum x + y)
    sum_coords = quad[:, 0] + quad[:, 1]  # x + y for each corner
    top_left_idx = np.argmin(sum_coords)

    # Find centroid
    centroid = quad.mean(axis=0)

    # Calculate angles from centroid (for clockwise ordering)
    angles = np.arctan2(quad[:, 1] - centroid[1], quad[:, 0] - centroid[0])

    # Sort by angle (counter-clockwise from right horizontal)
    order = np.argsort(angles)
    quad_sorted = quad[order]

    # Rotate array to start from top-left
    # Find where top-left ended up in sorted array
    tl_new_pos = np.where(order == top_left_idx)[0][0]
    quad_ordered = np.roll(quad_sorted, -tl_new_pos, axis=0)

    return quad_ordered


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: Array [x_min, y_min, x_max, y_max]
        box2: Array [x_min, y_min, x_max, y_max]

    Returns:
        IoU score (0.0 to 1.0)
    """
    # Compute intersection area
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    if x_max_inter < x_min_inter or y_max_inter < y_min_inter:
        return 0.0

    intersection = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)

    # Compute union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def apply_nms(quads: List[np.ndarray], confidences: List[float], iou_threshold: float = 0.20) -> Tuple[List[np.ndarray], List[float]]:
    """Apply Non-Maximum Suppression to remove overlapping detections.

    Args:
        quads: List of 4x2 keypoint arrays
        confidences: List of confidence scores
        iou_threshold: IoU threshold for suppression (default: 0.20)

    Returns:
        Tuple of (filtered_quads, filtered_confidences)
    """
    if len(quads) == 0:
        return [], []

    # Convert quads to bounding boxes [x_min, y_min, x_max, y_max]
    boxes = []
    for quad in quads:
        x_coords = quad[:, 0]
        y_coords = quad[:, 1]
        boxes.append([x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()])
    boxes = np.array(boxes)

    # Sort by confidence (descending)
    indices = np.argsort(confidences)[::-1]

    keep = []
    while len(indices) > 0:
        # Keep highest confidence detection
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # Compute IoU with remaining detections
        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]

        ious = np.array([compute_iou(current_box, box) for box in remaining_boxes])

        # Remove detections with IoU > threshold
        indices = indices[1:][ious <= iou_threshold]

    # Return filtered results
    filtered_quads = [quads[i] for i in keep]
    filtered_confidences = [confidences[i] for i in keep]

    return filtered_quads, filtered_confidences


def detect_quads(image_path: str, model_path: str, conf_threshold: float = 0.6, imgsz: int = 640) -> Tuple[List[np.ndarray], List[float]]:
    """Run YOLO pose inference and extract keypoint coordinates.

    Args:
        image_path: Path to input image
        model_path: Path to YOLO pose model (.pt file)
        conf_threshold: Confidence threshold for detections
        imgsz: Inference image size

    Returns:
        Tuple of (quads, confidences) where quads is list of 4x2 keypoint arrays
    """
    from ultralytics import YOLO

    # Load pose model
    model = YOLO(model_path)

    # Run inference
    results = model.predict(
        image_path,
        imgsz=imgsz,
        conf=conf_threshold,
        verbose=False
    )

    result = results[0]

    # Check if any detections exist
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return [], []

    quads = []
    confidences = []

    # Extract keypoints from pose model
    # result.keypoints.xy shape: [N, 4, 2] for N detections, 4 keypoints, (x, y) coords
    keypoints_xy = result.keypoints.xy.cpu().numpy()
    boxes_conf = result.boxes.conf.cpu().numpy()

    # Process each detection
    for kpts, conf in zip(keypoints_xy, boxes_conf):
        # kpts shape: [4, 2] for 4 corners
        # Keypoints are ordered: TL, TR, BR, BL (clockwise from top-left)

        # Validate keypoint ordering (ensure clockwise from top-left)
        ordered_kpts = order_corners_clockwise(kpts)

        quads.append(ordered_kpts.astype(np.float64))
        confidences.append(float(conf))

    # Apply Non-Maximum Suppression to remove overlapping detections
    quads, confidences = apply_nms(quads, confidences, iou_threshold=0.20)

    return quads, confidences


def main():
    """Main entry point.

    IMPORTANT: Outputs 0-based pixel coordinates (Python/OpenCV convention).
    MATLAB callers must add 1 to convert to 1-based indexing.
    """
    args = parse_args()

    # Validate inputs
    if not Path(args.image_path).exists():
        print(f"ERROR: Image not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.model_path).exists():
        print(f"ERROR: Model not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Run detection
        quads, confidences = detect_quads(
            args.image_path,
            args.model_path,
            args.conf,
            args.imgsz
        )

        # Output format: number of detections followed by quad data
        print(len(quads))

        for quad, conf in zip(quads, confidences):
            # Flatten quad to single line: x1 y1 x2 y2 x3 y3 x4 y4 confidence
            coords = ' '.join([f"{x:.6f} {y:.6f}" for x, y in quad])
            print(f"{coords} {conf:.6f}")

    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
