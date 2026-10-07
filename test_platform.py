import unittest

from platform_core import (
    PromotionStatus,
    ReleaseMetrics,
    deterministic_score,
    evaluate_release,
    population_stability_index,
)


class MLOpsPlatformTests(unittest.TestCase):
    def test_healthy_release_is_approved(self) -> None:
        metrics = ReleaseMetrics("2.3.0", 0.94, 165, 0.006, 0.08)
        self.assertEqual(evaluate_release(metrics).status, PromotionStatus.APPROVED)

    def test_drift_blocks_release(self) -> None:
        metrics = ReleaseMetrics("2.3.1", 0.95, 150, 0.005, 0.31)
        result = evaluate_release(metrics)
        self.assertEqual(result.status, PromotionStatus.BLOCKED)
        self.assertIn("feature drift above threshold", result.violations)

    def test_identical_distributions_have_zero_psi(self) -> None:
        psi = population_stability_index([10, 20, 30], [10, 20, 30])
        self.assertAlmostEqual(psi, 0.0)

    def test_distribution_shift_has_positive_psi(self) -> None:
        psi = population_stability_index([70, 20, 10], [20, 30, 50])
        self.assertGreater(psi, 0.2)

    def test_score_is_bounded_and_deterministic(self) -> None:
        first = deterministic_score([0.1, 0.5, 1.0])
        second = deterministic_score([0.1, 0.5, 1.0])
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLessEqual(first, 1)


if __name__ == "__main__":
    unittest.main()
