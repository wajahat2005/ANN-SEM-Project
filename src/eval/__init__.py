"""
Evaluation metrics, clinical threshold calibration, and error analysis.
"""

from src.eval.metrics import compute_classification_metrics, plot_confusion_matrix
from src.eval.calibration import (
    calibrate_threshold_for_sensitivity,
    compute_ece,
    plot_reliability_diagram,
    TemperatureScaler,
)
from src.eval.error_analysis import (
    analyze_misclassifications,
    plot_misclassified_grid,
    compute_per_class_accuracy,
)

__all__ = [
    "compute_classification_metrics",
    "plot_confusion_matrix",
    "calibrate_threshold_for_sensitivity",
    "compute_ece",
    "plot_reliability_diagram",
    "TemperatureScaler",
    "analyze_misclassifications",
    "plot_misclassified_grid",
    "compute_per_class_accuracy",
]
