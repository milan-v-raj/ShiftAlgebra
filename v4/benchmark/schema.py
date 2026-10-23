"""Versioned schema for CompShift-PP manifests.

The schema records information access explicitly so an R2 method cannot be
silently compared with an R3 method that sees the composed target images.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "0.1.0"


class InformationRegime(str, Enum):
    R0_SOURCE_ONLY = "R0"
    R1_PRIMITIVE_ONLY = "R1"
    R2_PAIR_CALIBRATED = "R2"
    R3_UNLABELED_TARGET = "R3"
    R4_LABELED_TARGET = "R4"


class ConditionRole(str, Enum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    FINAL = "final"
    UNASSIGNED = "unassigned"


@dataclass(frozen=True)
class InformationAccess:
    regime: InformationRegime
    source_labels: bool
    primitive_images: bool
    primitive_behavior_labels: bool
    pair_behavior_labels: bool
    composed_target_images: bool
    composed_target_labels: bool

    def validate(self) -> None:
        r = self.regime
        if r in (InformationRegime.R0_SOURCE_ONLY, InformationRegime.R1_PRIMITIVE_ONLY,
                 InformationRegime.R2_PAIR_CALIBRATED) and self.composed_target_images:
            raise ValueError(f"{r.value} cannot access composed target images")
        if r != InformationRegime.R4_LABELED_TARGET and self.composed_target_labels:
            raise ValueError("only R4 can access composed target labels at prediction time")
        if r == InformationRegime.R0_SOURCE_ONLY and (
            self.primitive_images or self.primitive_behavior_labels or self.pair_behavior_labels
        ):
            raise ValueError("R0 cannot access primitive or pair information")
        if r == InformationRegime.R1_PRIMITIVE_ONLY:
            if self.pair_behavior_labels:
                raise ValueError("R1 cannot access pair behavior labels")
            if not self.primitive_images or not self.primitive_behavior_labels:
                raise ValueError("R1 must declare primitive image and behavior access")
        if r == InformationRegime.R2_PAIR_CALIBRATED:
            if not self.pair_behavior_labels:
                raise ValueError("R2 must declare pair calibration access")
            if not self.primitive_images or not self.primitive_behavior_labels:
                raise ValueError("R2 must declare primitive image and behavior access")
        if r in (InformationRegime.R3_UNLABELED_TARGET, InformationRegime.R4_LABELED_TARGET) \
                and not self.composed_target_images:
            raise ValueError(f"{r.value} requires composed target images")
        if r == InformationRegime.R4_LABELED_TARGET and not self.composed_target_labels:
            raise ValueError("R4 requires composed target labels")


@dataclass(frozen=True)
class Condition:
    condition_id: str
    dataset: str
    word: tuple[str, ...]
    severities: tuple[float | str, ...]
    role: ConditionRole = ConditionRole.UNASSIGNED
    word_family: str = ""
    random_field_id: str = "deterministic"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.condition_id or not self.dataset or not self.word:
            raise ValueError("condition_id, dataset, and word are required")
        if len(self.word) != len(self.severities):
            raise ValueError("word and severity lengths must match")
        if any(not primitive for primitive in self.word):
            raise ValueError("primitive names must be non-empty")
        for severity in self.severities:
            if isinstance(severity, (int, float)) and not float("-inf") < float(severity) < float("inf"):
                raise ValueError("numeric severities must be finite")
            if isinstance(severity, str) and not severity:
                raise ValueError("string severities must be non-empty")


@dataclass
class BenchmarkManifest:
    benchmark_name: str
    conditions: list[Condition]
    source: Mapping[str, Any]
    schema_version: str = SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version {self.schema_version}")
        if not self.benchmark_name or not self.conditions:
            raise ValueError("benchmark name and at least one condition are required")
        seen: set[str] = set()
        for condition in self.conditions:
            condition.validate()
            if condition.condition_id in seen:
                raise ValueError(f"duplicate condition_id {condition.condition_id}")
            seen.add(condition.condition_id)

    def to_dict(self) -> dict:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "benchmark_name": self.benchmark_name,
            "source": dict(self.source),
            "metadata": dict(self.metadata),
            "conditions": [
                {
                    **asdict(condition),
                    "word": list(condition.word),
                    "severities": list(condition.severities),
                    "role": condition.role.value,
                    "metadata": dict(condition.metadata),
                }
                for condition in self.conditions
            ],
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: str | Path) -> "BenchmarkManifest":
        raw = json.loads(Path(path).read_text())
        conditions = [
            Condition(
                condition_id=item["condition_id"],
                dataset=item["dataset"],
                word=tuple(item["word"]),
                severities=tuple(item["severities"]),
                role=ConditionRole(item.get("role", "unassigned")),
                word_family=item.get("word_family", ""),
                random_field_id=item.get("random_field_id", "deterministic"),
                metadata=item.get("metadata", {}),
            )
            for item in raw["conditions"]
        ]
        manifest = cls(
            benchmark_name=raw["benchmark_name"],
            conditions=conditions,
            source=raw.get("source", {}),
            schema_version=raw["schema_version"],
            metadata=raw.get("metadata", {}),
        )
        manifest.validate()
        return manifest
