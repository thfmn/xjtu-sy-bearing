"""Tests for compute_twostage_rul — ONSET-19 acceptance criteria."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.rul_labels import compute_twostage_rul


class TestTwostageRulFlatBeforeOnsetDecayAfter:
    """ONSET-19 Acceptance: Two-stage RUL shows flat line (max_rul) before onset, then decay after."""

    def test_pre_onset_is_constant_max_rul(self):
        """All samples before onset_idx should be exactly max_rul."""
        rul = compute_twostage_rul(num_files=200, onset_idx=100, max_rul=125)
        np.testing.assert_array_equal(rul[:100], 125.0)

    def test_post_onset_decays_monotonically(self):
        """Samples from onset_idx onward should strictly decrease to 0."""
        rul = compute_twostage_rul(num_files=200, onset_idx=100, max_rul=125)
        post_onset = rul[100:]
        # Monotonically non-increasing
        assert np.all(np.diff(post_onset) <= 0), "Post-onset RUL must be non-increasing"

    def test_post_onset_ends_at_zero(self):
        """Last sample (failure) must have RUL = 0."""
        rul = compute_twostage_rul(num_files=200, onset_idx=100, max_rul=125)
        assert rul[-1] == 0.0

    def test_flat_then_decay_shape(self):
        """Verify overall shape: flat plateau then linear decay."""
        rul = compute_twostage_rul(num_files=200, onset_idx=100, max_rul=125)

        # Pre-onset: all same value (flat line)
        pre = rul[:100]
        assert np.std(pre) == 0.0, "Pre-onset should be perfectly flat"

        # Post-onset: not flat (decaying)
        post = rul[100:]
        assert np.std(post) > 0.0, "Post-onset should not be flat"

    def test_transition_at_onset_idx(self):
        """Just before onset is max_rul; at onset is <= max_rul (decay starts)."""
        rul = compute_twostage_rul(num_files=200, onset_idx=100, max_rul=125)
        assert rul[99] == 125.0, "Sample just before onset should be max_rul"
        # At onset: min(125, 200-100-1) = min(125, 99) = 99
        assert rul[100] == 99.0, "At onset should be min(max_rul, files_remaining-1)"

    def test_post_onset_capped_at_max_rul(self):
        """When post-onset count > max_rul+1, RUL at onset is capped at max_rul."""
        rul = compute_twostage_rul(num_files=300, onset_idx=50, max_rul=125)
        # Post-onset has 250 samples, so uncapped would be 249 at onset
        # Should be capped at 125
        assert rul[50] == 125.0, "Post-onset first RUL should be capped at max_rul"
        assert rul[-1] == 0.0

    def test_short_post_onset_uncapped(self):
        """When post-onset count < max_rul+1, RUL at onset is uncapped."""
        rul = compute_twostage_rul(num_files=200, onset_idx=150, max_rul=125)
        # Post-onset has 50 samples, so RUL at onset = 49 (< 125, uncapped)
        assert rul[150] == 49.0

    def test_all_15_bearings_shape(self):
        """Verify two-stage RUL shape for all 15 XJTU-SY bearings using real onset indices."""
        # Real bearing file counts and onset indices from configs/onset_labels.yaml
        bearings = {
            "Bearing1_1": (123, 69),
            "Bearing1_2": (161, 43),
            "Bearing1_3": (158, 59),
            "Bearing1_4": (122, 79),
            "Bearing1_5": (52, 27),
            "Bearing2_1": (491, 452),
            "Bearing2_2": (161, 47),
            "Bearing2_3": (533, 122),
            "Bearing2_4": (42, 9),
            "Bearing2_5": (339, 121),
            "Bearing3_1": (2538, 748),
            "Bearing3_2": (2496, 169),
            "Bearing3_3": (371, 339),
            "Bearing3_4": (1515, 1417),
            "Bearing3_5": (114, 29),
        }

        for bearing_id, (num_files, onset_idx) in bearings.items():
            rul = compute_twostage_rul(num_files, onset_idx, max_rul=125)

            # Correct length
            assert len(rul) == num_files, f"{bearing_id}: wrong length"

            # Pre-onset flat at max_rul
            if onset_idx > 0:
                np.testing.assert_array_equal(
                    rul[:onset_idx],
                    125.0,
                    err_msg=f"{bearing_id}: pre-onset not flat at 125",
                )

            # Post-onset ends at 0
            assert rul[-1] == 0.0, f"{bearing_id}: last sample not 0"

            # Post-onset monotonically decreasing
            post = rul[onset_idx:]
            assert np.all(np.diff(post) <= 0), f"{bearing_id}: post-onset not monotonic"
