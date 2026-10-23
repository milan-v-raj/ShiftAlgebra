#!/usr/bin/env python3
"""One-shot confirmation pipeline for preregistered Phase 1B."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

from v4.benchmark.schema import BenchmarkManifest
from v4.phase1.behavior import evaluate_protocol
from v4.phase1.classifiers import ClassifierConfig, train_classifier
from v4.phase1.corruptions import PRIMITIVES
from v4.phase1.data import balanced_subset
from v4.phase1.forecast import run_forecast_evaluation
from v4.phase1.pipeline import (
    CHECKPOINTS,
    DEFAULT_CIFAR,
    DEFAULT_EMNIST,
    DEFAULT_MANIFEST,
    load_dataset,
)
from v4.phase1b import PROTOCOL_ID
from v4.phase1b.calibration import fixed_configuration, sha256
from v4.phase1b.corruptions import apply_calibrated_word, validate_strengths

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "v4/phase1b/results"
DEFAULT_STRENGTHS = ROOT / "v4/phase1b/FROZEN_STRENGTHS.json"


def confirmation_partition(dataset: str) -> tuple[int, int]:
    if dataset == "cifar10":
        return 900, 100
    if dataset == "emnist":
        return 360, 40
    raise ValueError(dataset)


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    frozen = json.loads(args.strengths.read_text())
    if frozen["protocol"] != PROTOCOL_ID \
            or frozen["status"] != "STRENGTHS_FROZEN_TARGETS_UNOPENED":
        raise ValueError("Phase-1B strengths are not in the required frozen state")
    frozen_dataset = frozen["datasets"][args.dataset]
    strengths = validate_strengths(frozen_dataset["strengths"])

    width, epochs, train_batch, eval_batch, _ = fixed_configuration(args.dataset)
    bundle = load_dataset(args.dataset, args.cifar_root, args.emnist_root)
    config = ClassifierConfig(
        dataset=bundle.name,
        input_channels=bundle.train_images.shape[1],
        class_count=bundle.class_count,
        width=width,
        seed=1301,
        epochs=epochs,
        batch_size=train_batch,
        learning_rate=3e-3,
    )
    model, mean, std, training = train_classifier(bundle, config, args.checkpoint_dir)
    checkpoint = args.checkpoint_dir / (
        f"{bundle.name.lower().replace('-', '')}-{config.model_id}.pt"
    )
    observed_hash = sha256(checkpoint)
    if observed_hash != frozen_dataset["checkpoint_sha256"]:
        raise ValueError(
            f"checkpoint hash changed: {observed_hash} != "
            f"{frozen_dataset['checkpoint_sha256']}"
        )
    if config.model_id != frozen_dataset["model_id"]:
        raise ValueError("classifier model ID changed after severity calibration")

    manifest = BenchmarkManifest.read(args.manifest)
    conditions = [condition for condition in manifest.conditions if condition.dataset == bundle.name]
    if len(conditions) != 167:
        raise ValueError(f"manifest has {len(conditions)} {bundle.name} conditions")
    per_class, offset = confirmation_partition(args.dataset)
    subset = balanced_subset(bundle.test_labels, per_class, offset=offset)

    prefix = f"{args.dataset}-{config.model_id}-p1b-confirm"
    behavior_csv = args.results_dir / f"{prefix}-behavior.csv"
    fingerprint_json = args.results_dir / f"{prefix}-fingerprints.json"
    forecast_json = args.results_dir / f"{prefix}-forecast.json"
    transform = partial(apply_calibrated_word, strengths=strengths)
    evaluation = None
    if not args.resume_from_behavior or not behavior_csv.is_file():
        evaluation = evaluate_protocol(
            model,
            config.model_id,
            bundle.name,
            bundle.test_images[subset],
            bundle.test_labels[subset],
            subset,
            [condition.word for condition in conditions],
            [condition.condition_id for condition in conditions],
            mean,
            std,
            behavior_csv,
            fingerprint_json,
            batch_size=eval_batch,
            word_transform=transform,
        )

    forecast = run_forecast_evaluation(
        behavior_csv,
        PRIMITIVES,
        forecast_json,
        lp_steps=1500,
        potential_steps=2500,
        split_name="confirm",
        protocol_id=PROTOCOL_ID,
        include_target_rows=True,
    )
    forecast.update(
        {
            "dataset": bundle.name,
            "model_id": config.model_id,
            "severity_operator": "H[T,alpha](x)=(1-alpha)x+alpha*T(x)",
            "frozen_strengths": strengths,
            "frozen_strengths_file": str(args.strengths),
        }
    )
    forecast_json.write_text(json.dumps(forecast, indent=2, sort_keys=True) + "\n")

    summary = {
        "status": "PHASE1B_CONFIRMATORY_ONE_SHOT",
        "protocol": PROTOCOL_ID,
        "dataset": bundle.name,
        "class_count": bundle.class_count,
        "model_id": config.model_id,
        "checkpoint_sha256": observed_hash,
        "classifier_training": training,
        "evaluation": evaluation,
        "forecast": forecast,
        "manifest_sha256": manifest.sha256(),
        "frozen_strengths": strengths,
        "confirmation_partition": {
            "per_class": per_class,
            "offset": offset,
            "samples": int(subset.numel()),
            "reused_base_images_from_phase1": True,
            "phase1b_target_conditions_previously_opened": False,
        },
        "target_access_during_fit": False,
        "elapsed_seconds": time.time() - started,
    }
    run_json = args.results_dir / f"{prefix}-run.json"
    run_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Phase-1B confirmation result: {run_json}")
    print(json.dumps(forecast["methods"], indent=2, sort_keys=True))
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=("cifar10", "emnist"), required=True)
    result.add_argument("--cifar-root", type=Path, default=DEFAULT_CIFAR)
    result.add_argument("--emnist-root", type=Path, default=DEFAULT_EMNIST)
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINTS)
    result.add_argument("--results-dir", type=Path, default=RESULTS)
    result.add_argument("--strengths", type=Path, default=DEFAULT_STRENGTHS)
    result.add_argument(
        "--resume-from-behavior",
        action="store_true",
        help="reuse an already completed behavior CSV after a forecast-only runtime failure",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
