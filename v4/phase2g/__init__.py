"""Phase 2G: strict no-refit CIFAR-100 architecture transfer."""
PROTOCOL_ID = "V4-P2G-CIFAR100-NOREFIT-ARCH-001"
METRICS = (
    "margin",
    "feature_drift",
    "ece",
    "balanced_accuracy",
    "accuracy",
    "mean_confidence",
)
