"""Границы учебного эксперимента, бюджет и достоверность офлайн-повтора."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from harness import budget, coursework
from harness.builder_mcp import Bridge
from harness.run import configuration, launch


class BudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        (self.run / "control").mkdir()
        coursework.write(self.run / "manifest.json", {"limits": {"max_seconds": 900, "max_builds": 15}})

    def test_all_build_calls_are_limited_but_finish_remains_possible(self) -> None:
        with patch("harness.budget.time.monotonic", return_value=100):
            budget.begin(self.run)
            for _ in range(15):
                budget.reserve_build(self.run)
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                budget.reserve_build(self.run)
            self.assertIsNotNone(budget.check(self.run))

    def test_deadline_blocks_both_build_and_finish(self) -> None:
        with patch("harness.budget.time.monotonic", return_value=100):
            budget.begin(self.run)
        with patch("harness.budget.time.monotonic", return_value=1000):
            with self.assertRaises(TimeoutError):
                budget.check(self.run)
            with self.assertRaises(TimeoutError):
                budget.reserve_build(self.run)

    def test_legacy_run_has_no_new_deadline(self) -> None:
        coursework.write(self.run / "manifest.json", {})
        self.assertIsNone(budget.check(self.run))
        budget.reserve_build(self.run)

    def test_budget_cannot_be_used_before_launch(self) -> None:
        with self.assertRaises(FileNotFoundError):
            budget.reserve_build(self.run)

    @patch("harness.builder_mcp.verify_bundle")
    def test_invalid_recipe_consumes_a_call_without_running_native_code(self, verify: object) -> None:
        coursework.write(self.run / "control/settings.json", {})
        budget.begin(self.run)
        with patch("harness.builder_mcp.Bridge.build") as execute:
            with self.assertRaises(ValueError):
                Bridge(self.run).call("build", {"recipe": {"steps": [{"op": "open", "id": "leak"}]}})
            execute.assert_not_called()
        self.assertEqual(coursework.read(self.run / "control/budget.json")["build_calls"], 1)

    def test_launcher_timeout_preserves_attempt_and_quarantines_inflight_cad(self) -> None:
        for name in ("raw", "builder", "artifacts"):
            (self.run / name).mkdir()
        (self.run / "builder/AGENTS.md").write_text("Build")
        coursework.write(
            self.run / "manifest.json",
            {"status": "prepared", "limits": {"max_seconds": 900, "max_builds": 15}},
        )
        coursework.write(self.run / "control/native-inflight.json", {"attempt": 1})
        process = MagicMock()
        process.__enter__.return_value = process
        process.communicate.side_effect = subprocess.TimeoutExpired("codex", 900)
        process.returncode = -15
        with (
            patch("harness.run.preflight"),
            patch("harness.run.native_check"),
            patch("harness.run.configuration", return_value={"model_reasoning_effort": "high"}),
            patch("harness.run.subprocess.Popen", return_value=process),
            patch("harness.budget.quarantine") as quarantine,
        ):
            launch(self.run, "gpt-6-astra")
        self.assertEqual(coursework.read(self.run / "manifest.json")["status"], "timeout")
        self.assertTrue((self.run / "control/exit.json").exists())
        process.terminate.assert_called_once()
        quarantine.assert_called_once()


class IntegrityTests(unittest.TestCase):
    def test_selected_dataset_and_snapshot(self) -> None:
        config = coursework.configuration(coursework.DEFAULT_CONFIG)
        inventory = coursework.check(config, reviewed=True)
        self.assertEqual(
            [i["pair_id"] for i in inventory["items"] if i["split"] == "val"], ["KF0017", "KF0027"]
        )

    def test_changed_artifact_and_escape_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / "model.json"
            p.write_text("original")
            files = {p.name: coursework.sha(p)}
            coursework.verify_files(root, files)
            p.write_text("changed")
            with self.assertRaises(ValueError):
                coursework.verify_files(root, files)
            with self.assertRaises(ValueError):
                coursework.inside(root, "../outside")

    def test_public_references_are_not_builder_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            coursework.write(run / "manifest.json", {})
            policy = configuration(run)["permissions.kf0011.filesystem"]
            self.assertEqual(policy["/Users"], "deny")
            self.assertEqual(policy[str(run / "builder")], "read")
            self.assertNotIn(str(coursework.ROOT / "data"), policy)

    def test_replay_requires_all_cases_and_complete_metrics(self) -> None:
        def result(value: float, rankable: bool = True) -> dict:
            return {
                "summary": {"rankable": rankable},
                "cases": [{"id": "x", "primary": {k: {"value": value} for k in ("solid_iou", "f_score")}}],
            }

        coursework.compare_metrics(result(0.5), result(0.5 + 1e-8))
        with self.assertRaises(ValueError):
            coursework.compare_metrics(result(0.5), result(0.6))
        with self.assertRaises(ValueError):
            coursework.compare_metrics(result(0.5), result(0.5, False))
        missing = result(0.5)
        missing["cases"] = []
        with self.assertRaises(ValueError):
            coursework.compare_metrics(result(0.5), missing)

    def test_unknown_and_duplicate_splits_are_rejected(self) -> None:
        config = coursework.configuration(coursework.DEFAULT_CONFIG)
        original_read = coursework.read

        def changed(path: Path) -> dict:
            data = original_read(path)
            if path.name == "manifest.json":
                data = json.loads(json.dumps(data))
                data["items"][1]["family"] = data["items"][0]["family"]
            return data

        with (
            patch("harness.coursework.read", side_effect=changed),
            self.assertRaisesRegex(ValueError, "Related"),
        ):
            coursework.check(config)


if __name__ == "__main__":
    unittest.main()
