"""
Feedback Simulator for the Feedback Learning Benchmark.

Simulates agent feedback on retrieval results under three modes:
- oracle: perfect feedback (upper bound on improvement)
- realistic: 80% correct with 20% noise (expected real-world behavior)
- noisy: 50/50 random (should show minimal improvement, acts as lower bound)
"""

import random
from typing import Dict, List, Set

from alma.types import FeedbackSignal


class FeedbackSimulator:
    """Simulates agent feedback on retrieval results.

    Given a set of retrieved memory IDs and the ground-truth correct IDs,
    generates synthetic feedback signals that mimic how an agent would
    mark memories as USED or IGNORED.

    Args:
        mode: Simulation mode. One of:
            - ``'oracle'``: All correct IDs marked USED, all others IGNORED.
              Represents the theoretical upper bound on feedback quality.
            - ``'realistic'``: 80% of correct IDs marked USED (20% missed),
              plus 10% of incorrect IDs randomly marked USED (false positives).
              Models typical agent behavior with occasional mistakes.
            - ``'noisy'``: Each memory is randomly marked USED or IGNORED
              with 50/50 probability regardless of correctness.
              Should show minimal or no improvement, validating that
              the feedback mechanism is not just noise-driven.
        seed: Optional random seed for reproducibility.

    Example:
        >>> sim = FeedbackSimulator(mode="realistic", seed=42)
        >>> feedback = sim.generate_feedback(
        ...     retrieved_ids=["m1", "m2", "m3"],
        ...     correct_ids={"m1"},
        ... )
        >>> feedback["m1"]
        <FeedbackSignal.USED: 'used'>
    """

    VALID_MODES = ("oracle", "realistic", "noisy")

    def __init__(self, mode: str = "realistic", seed: int | None = None) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of {self.VALID_MODES}"
            )
        self.mode = mode
        self._rng = random.Random(seed)

    def generate_feedback(
        self,
        retrieved_ids: List[str],
        correct_ids: Set[str],
    ) -> Dict[str, FeedbackSignal]:
        """Generate simulated feedback signals for retrieved memories.

        Args:
            retrieved_ids: Ordered list of memory IDs returned by retrieval.
            correct_ids: Set of memory IDs that are genuinely relevant
                to the query (ground truth).

        Returns:
            Dict mapping each retrieved memory ID to a FeedbackSignal
            (USED or IGNORED) based on the simulation mode.
        """
        if self.mode == "oracle":
            return self._oracle_feedback(retrieved_ids, correct_ids)
        elif self.mode == "realistic":
            return self._realistic_feedback(retrieved_ids, correct_ids)
        else:
            return self._noisy_feedback(retrieved_ids)

    def _oracle_feedback(
        self,
        retrieved_ids: List[str],
        correct_ids: Set[str],
    ) -> Dict[str, FeedbackSignal]:
        """Perfect feedback: correct = USED, all others = IGNORED."""
        return {
            mid: FeedbackSignal.USED if mid in correct_ids else FeedbackSignal.IGNORED
            for mid in retrieved_ids
        }

    def _realistic_feedback(
        self,
        retrieved_ids: List[str],
        correct_ids: Set[str],
    ) -> Dict[str, FeedbackSignal]:
        """Realistic feedback: 80% correct USED, 10% false positives.

        - 80% of correct IDs are marked USED (20% missed as false negatives)
        - 10% of incorrect IDs are randomly marked USED (false positives)
        - All others are marked IGNORED
        """
        feedback: Dict[str, FeedbackSignal] = {}
        for mid in retrieved_ids:
            if mid in correct_ids:
                # 80% chance of correctly marking as USED
                if self._rng.random() < 0.8:
                    feedback[mid] = FeedbackSignal.USED
                else:
                    feedback[mid] = FeedbackSignal.IGNORED
            else:
                # 10% chance of false positive (incorrectly marking as USED)
                if self._rng.random() < 0.1:
                    feedback[mid] = FeedbackSignal.USED
                else:
                    feedback[mid] = FeedbackSignal.IGNORED
        return feedback

    def _noisy_feedback(
        self,
        retrieved_ids: List[str],
    ) -> Dict[str, FeedbackSignal]:
        """Noisy feedback: 50/50 random regardless of correctness."""
        return {
            mid: (
                FeedbackSignal.USED
                if self._rng.random() < 0.5
                else FeedbackSignal.IGNORED
            )
            for mid in retrieved_ids
        }
