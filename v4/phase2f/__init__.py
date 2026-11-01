"""Phase 2F prospective full-vector CIFAR-100 benchmark."""
PROTOCOL_ID='V4-P2F-CIFAR100-FULLVECTOR-CPU-001'
PRIMITIVES=('Contrast','GaussianBlur','ImpulseNoise','Invert','Rotate90','Swirl')
METRICS=('margin','feature_drift','ece','balanced_accuracy','accuracy','mean_confidence')
BOUNDED_METRICS=('margin','ece','balanced_accuracy','accuracy','mean_confidence')
PERFORMANCE_METRICS=('margin','balanced_accuracy','accuracy','mean_confidence')
