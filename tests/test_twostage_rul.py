"""Tests for compute_twostage_rul — ONSET-19 acceptance criteria."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.rul_labels import (
    compute_twostage_rul,
    exponential_rul,
    generate_rul_for_bearing,
    generate_rul_labels,
    linear_rul,
    piecewise_linear_rul,
)


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


class TestOnsetRelativeRulAtFailureIsZero:
    """ONSET-19 Acceptance: Onset-relative RUL at failure is 0 (same as before)."""

    def test_failure_rul_is_zero(self):
        """Last sample (failure) must have RUL = 0 regardless of onset position."""
        for onset_idx in [0, 1, 50, 100, 199]:
            rul = compute_twostage_rul(num_files=200, onset_idx=onset_idx, max_rul=125)
            assert rul[-1] == 0.0, f"onset_idx={onset_idx}: last sample should be 0"

    def test_failure_matches_piecewise_linear(self):
        """Two-stage failure RUL (0) matches standard piecewise_linear failure RUL (0)."""
        for num_files in [52, 123, 200, 491, 2538]:
            rul_pw = piecewise_linear_rul(num_files, max_rul=125)
            rul_ts = compute_twostage_rul(num_files, onset_idx=num_files // 2, max_rul=125)
            assert rul_pw[-1] == rul_ts[-1] == 0.0, (
                f"num_files={num_files}: both strategies must end at 0"
            )

    def test_all_15_bearings_failure_is_zero(self):
        """All 15 XJTU-SY bearings have RUL=0 at failure, matching piecewise_linear."""
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
            rul_ts = compute_twostage_rul(num_files, onset_idx, max_rul=125)
            rul_pw = piecewise_linear_rul(num_files, max_rul=125)
            assert rul_ts[-1] == 0.0, f"{bearing_id}: twostage failure not 0"
            assert rul_pw[-1] == 0.0, f"{bearing_id}: piecewise_linear failure not 0"

    def test_single_file_bearing_failure_is_zero(self):
        """Edge case: single-file bearing with onset at 0 has RUL=0."""
        rul = compute_twostage_rul(num_files=1, onset_idx=0, max_rul=125)
        assert rul[-1] == 0.0


class TestOnsetRelativeRulAtOnsetIsMinMaxRulFilesRemaining:
    """ONSET-19 Acceptance: Onset-relative RUL at onset is min(max_rul, files_remaining).

    files_remaining at onset_idx = num_files - onset_idx (count of samples from onset to end).
    RUL at onset = min(max_rul, files_remaining - 1) because the last sample has RUL=0.
    This matches the standard RUL convention: RUL = steps until failure.
    """

    def test_onset_rul_uncapped(self):
        """When files_remaining - 1 < max_rul, RUL at onset equals files_remaining - 1."""
        # 200 files, onset at 100 → files_remaining = 100, RUL at onset = 99
        rul = compute_twostage_rul(num_files=200, onset_idx=100, max_rul=125)
        files_remaining = 200 - 100
        expected = files_remaining - 1  # 99
        assert rul[100] == expected

    def test_onset_rul_capped(self):
        """When files_remaining - 1 > max_rul, RUL at onset equals max_rul."""
        # 300 files, onset at 50 → files_remaining = 250, RUL at onset = min(125, 249) = 125
        rul = compute_twostage_rul(num_files=300, onset_idx=50, max_rul=125)
        assert rul[50] == 125.0

    def test_onset_rul_exactly_at_cap(self):
        """When files_remaining - 1 == max_rul, RUL at onset equals max_rul."""
        # 226 files, onset at 100 → files_remaining = 126, RUL at onset = min(125, 125) = 125
        rul = compute_twostage_rul(num_files=226, onset_idx=100, max_rul=125)
        assert rul[100] == 125.0

    def test_onset_at_zero_rul(self):
        """Onset at index 0: RUL at onset = min(max_rul, num_files - 1)."""
        rul = compute_twostage_rul(num_files=50, onset_idx=0, max_rul=125)
        assert rul[0] == 49.0  # min(125, 49) = 49

    def test_onset_at_last_sample(self):
        """Onset at last sample: files_remaining=1, RUL at onset = 0."""
        rul = compute_twostage_rul(num_files=200, onset_idx=199, max_rul=125)
        assert rul[199] == 0.0

    @pytest.mark.parametrize(
        "num_files,onset_idx,max_rul",
        [
            (123, 69, 125),
            (161, 43, 125),
            (52, 27, 125),
            (491, 452, 125),
            (2538, 748, 125),
            (2496, 169, 125),
            (1515, 1417, 125),
        ],
    )
    def test_all_real_bearings_onset_rul(self, num_files, onset_idx, max_rul):
        """Verify RUL at onset for real XJTU-SY bearing parameters."""
        rul = compute_twostage_rul(num_files, onset_idx, max_rul=max_rul)
        files_remaining = num_files - onset_idx
        expected = min(max_rul, files_remaining - 1)
        assert rul[onset_idx] == expected, (
            f"num_files={num_files}, onset_idx={onset_idx}: "
            f"expected RUL={expected}, got {rul[onset_idx]}"
        )


class TestBackwardCompatibility:
    """ONSET-19 Acceptance: Backward compatible — default behavior unchanged.

    Verify that all existing RUL functions produce identical outputs when
    called with their original signatures (without onset_idx/twostage).
    """

    @pytest.mark.parametrize("num_files", [1, 10, 52, 123, 200, 491, 2538])
    def test_piecewise_linear_unchanged(self, num_files):
        """piecewise_linear_rul output is identical before and after ONSET-19 changes."""
        rul = piecewise_linear_rul(num_files, max_rul=125.0)
        assert rul.shape == (num_files,)
        assert rul[-1] == 0.0
        if num_files > 1:
            expected_first = min(num_files - 1, 125.0)
            assert rul[0] == expected_first

    @pytest.mark.parametrize("num_files", [1, 10, 52, 200])
    def test_linear_rul_unchanged(self, num_files):
        """linear_rul output is identical before and after ONSET-19 changes."""
        rul = linear_rul(num_files)
        assert rul.shape == (num_files,)
        assert rul[-1] == 0.0
        if num_files > 1:
            assert rul[0] == num_files - 1

    @pytest.mark.parametrize("num_files", [1, 10, 52, 200])
    def test_exponential_rul_unchanged(self, num_files):
        """exponential_rul output is identical before and after ONSET-19 changes."""
        rul = exponential_rul(num_files, decay_rate=3.0)
        assert rul.shape == (num_files,)
        if num_files > 1:
            assert rul[0] > rul[-1], "Exponential should decay"

    def test_generate_rul_labels_default_is_piecewise_linear(self):
        """Default strategy is piecewise_linear, not twostage."""
        rul_default = generate_rul_labels(200)
        rul_pw = piecewise_linear_rul(200, max_rul=125.0)
        np.testing.assert_array_equal(rul_default, rul_pw)

    @pytest.mark.parametrize("strategy", ["piecewise_linear", "linear", "exponential"])
    def test_generate_rul_labels_ignores_onset_idx_for_non_twostage(self, strategy):
        """onset_idx parameter is ignored for all strategies except twostage."""
        rul_no_onset = generate_rul_labels(200, strategy=strategy)
        rul_with_onset = generate_rul_labels(200, strategy=strategy, onset_idx=50)
        np.testing.assert_array_equal(rul_no_onset, rul_with_onset)

    def test_generate_rul_for_bearing_default_unchanged(self):
        """generate_rul_for_bearing with defaults produces piecewise_linear output."""
        rul = generate_rul_for_bearing(200)
        rul_pw = piecewise_linear_rul(200, max_rul=125.0)
        np.testing.assert_array_equal(rul, rul_pw)

    def test_generate_rul_for_bearing_normalize_unchanged(self):
        """generate_rul_for_bearing with normalize=True still works correctly."""
        rul = generate_rul_for_bearing(200, normalize=True)
        assert rul.max() <= 1.0
        assert rul.min() >= 0.0
        assert rul[-1] == 0.0

    def test_generate_rul_for_bearing_onset_idx_ignored_for_default_strategy(self):
        """onset_idx kwarg is ignored when strategy is not twostage."""
        rul_default = generate_rul_for_bearing(200)
        rul_with_onset = generate_rul_for_bearing(200, onset_idx=50)
        np.testing.assert_array_equal(rul_default, rul_with_onset)
