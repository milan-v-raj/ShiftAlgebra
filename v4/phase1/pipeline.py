#!/usr/bin/env python3
"""Phase-1 development pipeline for the independent Mason protocol bridge."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from v4.benchmark.schema import BenchmarkManifest

from .behavior import evaluate_protocol
from .classifiers import ClassifierConfig, train_classifier
from .corruptions import PRIMITIVES
from .data import balanced_subset, load_cifar10, load_emnist_balanced
from .forecast import run_forecast_evaluation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CIFAR = ROOT / "data/external/cifar-mirror/data/CIFAR-10/cifar-10-batches-py"
DEFAULT_EMNIST = ROOT / "data/external/emnist-packed/emnist-balanced"
DEFAULT_MANIFEST = ROOT / "v4/benchmark/manifests/mason_exact.json"
RESULTS = ROOT / "v4/phase1/results"
CHECKPOINTS = ROOT / "checkpoints_v4/phase1"


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_dataset(name: str, cifar_root: Path, emnist_root: Path):
    if name == "cifar10":
        return load_cifar10(cifar_root)
    if name == "emnist":
        return load_emnist_balanced(emnist_root)
    raise ValueError(name)


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    bundle = load_dataset(args.dataset, args.cifar_root, args.emnist_root)
    config = ClassifierConfig(
        dataset=bundle.name,
        input_channels=bundle.train_images.shape[1],
        class_count=bundle.class_count,
        width=args.width,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
    )
    model, mean, std, training = train_classifier(
        bundle, config, args.checkpoint_dir, force=args.force_train
    )

    dataset_key = bundle.name.lower().replace("-", "")
    prefix = f"{dataset_key}-{config.model_id}-{args.split_name}"
    behavior_csv = args.results_dir / f"{prefix}-behavior.csv"
    fingerprint_json = args.results_dir / f"{prefix}-fingerprints.json"
    forecast_json = args.results_dir / f"{prefix}-forecast.json"

    manifest = BenchmarkManifest.read(args.manifest)
    conditions = [condition for condition in manifest.conditions if condition.dataset == bundle.name]
    if len(conditions) != 167:
        raise ValueError(f"manifest has {len(conditions)} {bundle.name} conditions")
    subset = balanced_subset(bundle.test_labels, args.dev_per_class, offset=args.dev_offset)
    evaluation = None
    if not args.skip_evaluation or not behavior_csv.is_file():
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
            batch_size=args.eval_batch_size,
        )
    forecast = run_forecast_evaluation(
        behavior_csv,
        PRIMITIVES,
        forecast_json,
        lp_steps=args.lp_steps,
        potential_steps=args.potential_steps,
        split_name=args.split_name,
    )

    summary = {
        "status": (
            "CONFIRMATORY_ONE_SHOT" if args.split_name == "confirm"
            else "DEVELOPMENT_ONLY_NOT_CONFIRMATORY"
        ),
        "protocol": "V4-P1-MASON-001",
        "dataset": bundle.name,
        "model_id": config.model_id,
        "classifier_training": training,
        "evaluation": evaluation,
        "forecast": forecast,
        "manifest_sha256": manifest.sha256(),
        "dataset_provenance": bundle.provenance,
        "transport_commits": {
            "cifar_mirror": _git_commit(args.cifar_root.parents[2]) if args.dataset == "cifar10" else None,
            "emnist_mirror": _git_commit(args.emnist_root.parent) if args.dataset == "emnist" else None,
        },
        "development_subset": {
            "per_class": args.dev_per_class,
            "offset": args.dev_offset,
            "samples": int(subset.numel()),
        },
        "elapsed_seconds": time.time() - started,
    }
    summary_path = args.results_dir / f"{prefix}-run.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Phase-1 development result: {summary_path}")
    print(json.dumps(forecast["methods"], indent=2, sort_keys=True))
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=("cifar10", "emnist"), required=True)
    result.add_argument("--cifar-root", type=Path, default=DEFAULT_CIFAR)
    result.add_argument("--emnist-root", type=Path, default=DEFAULT_EMNIST)
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument("--results-dir", type=Path, default=RESULTS)
    result.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINTS)
    result.add_argument("--width", type=int, default=24)
    result.add_argument("--seed", type=int, default=1301)
    result.add_argument("--epochs", type=int, default=8)
    result.add_argument("--train-batch-size", type=int, default=256)
    result.add_argument("--eval-batch-size", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=3e-3)
    result.add_argument("--dev-per-class", type=int, default=100)
    result.add_argument("--dev-offset", type=int, default=0)
    result.add_argument("--split-name", choices=("dev", "confirm"), default="dev")
    result.add_argument("--lp-steps", type=int, default=1500)
    result.add_argument("--potential-steps", type=int, default=2500)
    result.add_argument("--force-train", action="store_true")
    result.add_argument("--skip-evaluation", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
