from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import log
from typing import Iterable


class PromotionStatus(str, Enum):
    APPROVED = "approved"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ReleaseMetrics:
    version: str
    accuracy: float
    p95_latency_ms: float
    error_rate: float
    drift_psi: float


@dataclass(frozen=True)
class PromotionPolicy:
    min_accuracy: float = 0.90
    max_p95_latency_ms: float = 250.0
    max_error_rate: float = 0.02
    max_drift_psi: float = 0.20


@dataclass(frozen=True)
class PromotionResult:
    status: PromotionStatus
    version: str
    violations: tuple[str, ...]


def evaluate_release(
    metrics: ReleaseMetrics, policy: PromotionPolicy | None = None
) -> PromotionResult:
    policy = policy or PromotionPolicy()
    violations: list[str] = []

    if metrics.accuracy < policy.min_accuracy:
        violations.append("accuracy below threshold")
    if metrics.p95_latency_ms > policy.max_p95_latency_ms:
        violations.append("p95 latency above threshold")
    if metrics.error_rate > policy.max_error_rate:
        violations.append("error rate above threshold")
    if metrics.drift_psi > policy.max_drift_psi:
        violations.append("feature drift above threshold")

    status = PromotionStatus.BLOCKED if violations else PromotionStatus.APPROVED
    return PromotionResult(status, metrics.version, tuple(violations))


def population_stability_index(
    expected: Iterable[float], actual: Iterable[float], epsilon: float = 1e-6
) -> float:
    """Calculate PSI from two equal-length probability distributions."""
    expected_values = list(expected)
    actual_values = list(actual)
    if len(expected_values) != len(actual_values) or not expected_values:
        raise ValueError("distributions must be non-empty and equal length")
    if any(value < 0 for value in expected_values + actual_values):
        raise ValueError("distribution values cannot be negative")

    expected_total = sum(expected_values)
    actual_total = sum(actual_values)
    if expected_total <= 0 or actual_total <= 0:
        raise ValueError("distribution totals must be positive")

    psi = 0.0
    for expected_value, actual_value in zip(expected_values, actual_values):
        expected_ratio = max(expected_value / expected_total, epsilon)
        actual_ratio = max(actual_value / actual_total, epsilon)
        psi += (actual_ratio - expected_ratio) * log(actual_ratio / expected_ratio)
    return psi


def deterministic_score(features: list[float]) -> float:
    """Small deterministic scorer used in place of a proprietary model artifact."""
    if not features:
        raise ValueError("at least one feature is required")
    clipped_average = sum(max(-10.0, min(10.0, value)) for value in features) / len(features)
    return round(1 / (1 + 2.718281828 ** (-clipped_average)), 6)
