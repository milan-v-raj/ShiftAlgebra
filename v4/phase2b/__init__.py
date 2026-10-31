"""Phase 2B CPU-only adaptive composition hazard mixture."""

PROTOCOL_ID = "V4-P2B-ACHM-CPU-001"
SOURCE_DEVELOPMENT_PROTOCOL = "V4-P2-CIFAR-BANK-PILOT-001"
PRIMITIVES = (
    "Contrast",
    "GaussianBlur",
    "ImpulseNoise",
    "Invert",
    "Rotate90",
    "Swirl",
)
CHANCE = 0.1
Q = 0.6
