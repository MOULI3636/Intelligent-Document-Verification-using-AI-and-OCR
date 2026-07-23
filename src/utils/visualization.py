"""
Visualization Utilities for Document Analysis and Research Benchmarking.

Renders bounding box annotations, confidence heatmaps, side-by-side engine outputs,
and Matplotlib/Seaborn metric distributions for research evaluation.
"""

from typing import Dict, List, Optional, Tuple, Any
import cv2
import matplotlib.pyplot as plt
import numpy as np


def render_ocr_overlay(
    image: np.ndarray,
    bboxes: List[List[Tuple[int, int]]],
    texts: List[str],
    confidences: Optional[List[float]] = None,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Renders bounding boxes and recognized text labels directly onto a document image.

    Args:
        image (np.ndarray): Input BGR or Grayscale image array.
        bboxes (List[List[Tuple[int, int]]]): Bounding polygon coordinates [(x1,y1), (x2,y2), ...].
        texts (List[str]): Corresponding text string predictions.
        confidences (Optional[List[float]]): Model confidence scores [0.0 - 1.0].
        box_color (Tuple[int, int, int]): BGR color tuple for box outlines.
        thickness (int): Line thickness in pixels.

    Returns:
        np.ndarray: Annotated RGB image copy.
    """
    if len(image.shape) == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        canvas = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for idx, polygon in enumerate(bboxes):
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=box_color, thickness=thickness)

        text = texts[idx] if idx < len(texts) else ""
        if confidences and idx < len(confidences):
            label = f"{text} ({confidences[idx]:.2f})"
        else:
            label = text

        if len(polygon) > 0:
            x_min = int(min(p[0] for p in polygon))
            y_min = int(min(p[1] for p in polygon))
            
            # Draw label background box
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                canvas,
                (x_min, max(0, y_min - text_h - baseline)),
                (x_min + text_w, max(0, y_min)),
                box_color,
                cv2.FILLED
            )
            cv2.putText(
                canvas,
                label,
                (x_min, max(baseline, y_min - baseline)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

    return canvas


def plot_side_by_side(
    original_img: np.ndarray,
    processed_img: np.ndarray,
    title_left: str = "Original Input",
    title_right: str = "Processed Output"
) -> plt.Figure:
    """
    Plots two images side-by-side for visual document pre-processing evaluation.

    Args:
        original_img (np.ndarray): Left image array.
        processed_img (np.ndarray): Right image array.
        title_left (str): Subplot title for left image.
        title_right (str): Subplot title for right image.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    cmap_left = "gray" if len(original_img.shape) == 2 else None
    cmap_right = "gray" if len(processed_img.shape) == 2 else None

    axes[0].imshow(original_img, cmap=cmap_left)
    axes[0].set_title(title_left, fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(processed_img, cmap=cmap_right)
    axes[1].set_title(title_right, fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    return fig


def plot_metric_distribution(
    metrics_data: Dict[str, List[float]],
    title: str = "OCR Metric Error Distributions (CER / WER)"
) -> plt.Figure:
    """
    Plots boxplots and kernel density estimates for evaluation error distributions.

    Args:
        metrics_data (Dict[str, List[float]]): Dictionary mapping metric name to list of values.
        title (str): Main plot title.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(metrics_data.keys())
    values = [metrics_data[k] for k in names]

    bp = ax.boxplot(values, patch_artist=True, labels=names)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B0"]

    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Error Rate Score", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig
