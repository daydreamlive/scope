"""Pose visualization utilities for skeleton drawing."""

import cv2
import numpy as np

# COCO skeleton edge definitions
# Each pair defines a connection between two keypoints
EDGE_LINKS = [
    [0, 17],  # nose to mid-shoulder
    [13, 15],  # left knee to left ankle
    [14, 16],  # right knee to right ankle
    [12, 14],  # right hip to right knee
    [12, 17],  # right hip to mid-shoulder
    [5, 6],  # left shoulder to right shoulder
    [11, 13],  # left hip to left knee
    [7, 9],  # left elbow to left wrist
    [5, 7],  # left shoulder to left elbow
    [17, 11],  # mid-shoulder to left hip
    [6, 8],  # right shoulder to right elbow
    [8, 10],  # right elbow to right wrist
    [1, 3],  # left eye to left ear
    [0, 1],  # nose to left eye
    [0, 2],  # nose to right eye
    [2, 4],  # right eye to right ear
]

# Colors for each edge (BGR format for OpenCV)
EDGE_COLORS = [
    [255, 0, 0],  # blue
    [255, 85, 0],  # blue-cyan
    [170, 255, 0],  # cyan-green
    [85, 255, 0],  # green
    [85, 255, 0],  # green
    [85, 0, 255],  # purple
    [255, 170, 0],  # cyan
    [0, 177, 58],  # dark green
    [0, 179, 119],  # teal
    [179, 179, 0],  # cyan
    [0, 119, 179],  # orange
    [0, 179, 179],  # yellow
    [119, 0, 179],  # magenta
    [179, 0, 179],  # pink
    [178, 0, 118],  # red-pink
    [178, 0, 118],  # red-pink
]


class PoseVisualization:
    """Pose drawing utilities for skeleton visualization."""

    @staticmethod
    def draw_skeleton(
        image: np.ndarray,
        keypoints: np.ndarray,
        edge_links: list[list[int]] = EDGE_LINKS,
        edge_colors: list[list[int]] = EDGE_COLORS,
        joint_thickness: int = 10,
        keypoint_radius: int = 10,
        confidence_threshold: float = 0.5,
    ) -> np.ndarray:
        """Draw pose skeleton on image.

        Args:
            image: Input image (HWC format, BGR)
            keypoints: Array of keypoints [N, 3] where each row is (x, y, confidence)
            edge_links: List of keypoint index pairs defining skeleton edges
            edge_colors: List of BGR colors for each edge
            joint_thickness: Line thickness for skeleton edges
            keypoint_radius: Radius for keypoint circles
            confidence_threshold: Minimum confidence to draw keypoint/edge

        Returns:
            Image with skeleton overlay
        """
        overlay = image.copy()

        # Draw edges/links between keypoints
        for (kp1, kp2), color in zip(edge_links, edge_colors):
            if kp1 < len(keypoints) and kp2 < len(keypoints):
                # Check if both keypoints are valid (confidence > threshold)
                if len(keypoints[kp1]) >= 3 and len(keypoints[kp2]) >= 3:
                    conf1, conf2 = keypoints[kp1][2], keypoints[kp2][2]
                    if conf1 > confidence_threshold and conf2 > confidence_threshold:
                        p1 = (int(keypoints[kp1][0]), int(keypoints[kp1][1]))
                        p2 = (int(keypoints[kp2][0]), int(keypoints[kp2][1]))
                        cv2.line(
                            overlay,
                            p1,
                            p2,
                            color=color,
                            thickness=joint_thickness,
                            lineType=cv2.LINE_AA,
                        )

        # Draw keypoints
        for keypoint in keypoints:
            if len(keypoint) >= 3 and keypoint[2] > confidence_threshold:
                x, y = int(keypoint[0]), int(keypoint[1])
                cv2.circle(
                    overlay, (x, y), keypoint_radius, (0, 255, 0), -1, cv2.LINE_AA
                )

        return cv2.addWeighted(overlay, 0.75, image, 0.25, 0)

    @staticmethod
    def draw_poses(
        image: np.ndarray,
        poses: np.ndarray,
        edge_links: list[list[int]] = EDGE_LINKS,
        edge_colors: list[list[int]] = EDGE_COLORS,
        joint_thickness: int = 10,
        keypoint_radius: int = 10,
        confidence_threshold: float = 0.5,
    ) -> np.ndarray:
        """Draw multiple poses on image.

        Args:
            image: Input image (HWC format, BGR)
            poses: Array of poses [num_poses, num_keypoints, 3]
            edge_links: List of keypoint index pairs defining skeleton edges
            edge_colors: List of BGR colors for each edge
            joint_thickness: Line thickness for skeleton edges
            keypoint_radius: Radius for keypoint circles
            confidence_threshold: Minimum confidence to draw keypoint/edge

        Returns:
            Image with all skeletons overlaid
        """
        result = image.copy()

        for pose in poses:
            result = PoseVisualization.draw_skeleton(
                result,
                pose,
                edge_links,
                edge_colors,
                joint_thickness,
                keypoint_radius,
                confidence_threshold,
            )

        return result


def iterate_over_batch_predictions(predictions: list[np.ndarray], batch_size: int):
    """Process batch predictions from TensorRT output.

    Args:
        predictions: List of prediction arrays [num_detections, boxes, scores, joints]
        batch_size: Number of images in batch

    Yields:
        Tuple of (image_index, pred_boxes, pred_scores, pred_joints)
    """
    num_detections, batch_boxes, batch_scores, batch_joints = predictions

    for image_index in range(batch_size):
        num_detection_in_image = int(num_detections[image_index, 0])

        # Handle case where no detections are found
        if num_detection_in_image == 0:
            pred_scores = np.array([])
            pred_boxes = np.array([]).reshape(0, 4)
            pred_joints = np.array([]).reshape(0, 17, 3)
        else:
            pred_scores = batch_scores[image_index, :num_detection_in_image]
            pred_boxes = batch_boxes[image_index, :num_detection_in_image]
            pred_joints = batch_joints[image_index, :num_detection_in_image].reshape(
                (num_detection_in_image, -1, 3)
            )

        yield image_index, pred_boxes, pred_scores, pred_joints


def render_pose_image(
    predictions: list[np.ndarray],
    detect_resolution: int = 640,
    joint_thickness: int = 10,
    keypoint_radius: int = 10,
) -> np.ndarray:
    """Convert TensorRT predictions to pose visualization image.

    Args:
        predictions: List of prediction arrays from TensorRT
        detect_resolution: Resolution of detection (for output image size)
        joint_thickness: Line thickness for skeleton edges
        keypoint_radius: Radius for keypoint circles

    Returns:
        Pose visualization image (HWC, BGR, uint8)
    """
    try:
        image_index, pred_boxes, pred_scores, pred_joints = next(
            iter(iterate_over_batch_predictions(predictions, 1))
        )
    except Exception:
        # Return black image on error
        return np.zeros((detect_resolution, detect_resolution, 3), dtype=np.uint8)

    # Handle case where no poses are detected
    if pred_joints.shape[0] == 0:
        return np.zeros((detect_resolution, detect_resolution, 3), dtype=np.uint8)

    # Add middle joint between shoulders (keypoints 5 and 6)
    # This creates keypoint 17 for the mid-shoulder connection
    try:
        middle_joints = (pred_joints[:, 5] + pred_joints[:, 6]) / 2
        new_pred_joints = np.concatenate(
            [pred_joints, middle_joints[:, np.newaxis]], axis=1
        )
    except Exception:
        new_pred_joints = pred_joints

    # Create black background for pose visualization
    black_image = np.zeros((detect_resolution, detect_resolution, 3), dtype=np.uint8)

    try:
        image = PoseVisualization.draw_poses(
            image=black_image,
            poses=new_pred_joints,
            edge_links=EDGE_LINKS,
            edge_colors=EDGE_COLORS,
            joint_thickness=joint_thickness,
            keypoint_radius=keypoint_radius,
        )
    except Exception:
        return black_image

    return image
