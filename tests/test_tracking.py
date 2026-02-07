#  Copyright (C) 2026 by Tobias Hoffmann
#  thoffmann-ml@proton.me
#  https://github.com/thfmn/xjtu-sy-bearing
#
#  This work is licensed under the MIT License. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#  and to permit persons to whom the Software is furnished to do so, subject to the condition that the above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  For more information, visit: https://opensource.org/licenses/MIT
#
#  Author:    Tobias Hoffmann
#  Email:     thoffmann-ml@proton.me
#  License:   MIT
#  Date:      2025-2026
#  Package:   xjtu-sy-bearing onset and RUL prediction ML Pipeline

"""Tests for ExperimentTracker context manager behavior."""

import tempfile

import mlflow
import pytest

from src.utils.tracking import ExperimentTracker


@pytest.fixture()
def tracker(tmp_path):
    """Create an MLflow-backed tracker with a temporary tracking directory."""
    tracking_uri = str(tmp_path / "mlruns")
    t = ExperimentTracker(
        backend="mlflow",
        experiment_name="test_experiment",
        tracking_uri=tracking_uri,
    )
    return t


class TestStartRunContextManager:
    """Tests for ExperimentTracker.start_run() context manager."""

    def test_end_run_called_on_normal_exit(self, tracker):
        """Context manager calls end_run with FINISHED on normal exit."""
        with tracker.start_run("test_run") as run:
            run.log_metrics({"loss": 0.5})

        # After exiting the context manager, no active run should remain
        active = mlflow.active_run()
        assert active is None

    def test_end_run_called_on_exception(self, tracker):
        """Context manager calls end_run with FAILED on exception."""
        with pytest.raises(ValueError, match="deliberate"):
            with tracker.start_run("failing_run") as run:
                run.log_metrics({"loss": 1.0})
                raise ValueError("deliberate error")

        # Run should still be ended (no leaked active run)
        active = mlflow.active_run()
        assert active is None

    def test_run_status_finished_on_success(self, tracker):
        """Run status is FINISHED after successful context manager exit."""
        with tracker.start_run("success_run") as run:
            run.log_metrics({"accuracy": 0.95})

        # Get the most recent run and check its status
        runs = tracker.list_runs()
        assert len(runs) >= 1
        latest = runs[0]
        assert latest.status == "FINISHED"

    def test_run_status_failed_on_exception(self, tracker):
        """Run status is FAILED after exception in context manager."""
        with pytest.raises(RuntimeError):
            with tracker.start_run("error_run") as run:
                run.log_metrics({"loss": 999.0})
                raise RuntimeError("training crashed")

        runs = tracker.list_runs()
        assert len(runs) >= 1
        latest = runs[0]
        assert latest.status == "FAILED"

    def test_no_explicit_end_run_needed(self, tracker):
        """Verify that no explicit tracker.end_run() is needed after with block."""
        # This test ensures the bug fix: previously, users had to call
        # tracker.end_run() explicitly after the with block because the
        # finally clause had `pass` instead of calling end_run.
        with tracker.start_run("auto_close_run") as run:
            run.log_params({"lr": 0.001})

        # Starting a second run should work without any manual end_run() call
        with tracker.start_run("second_run") as run:
            run.log_params({"lr": 0.01})

        active = mlflow.active_run()
        assert active is None
