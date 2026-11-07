"""Границы моста: в Windows проходит только ограниченный рецепт, без кода и путей."""

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from harness.builder_mcp import Bridge, connection_failed_before_execution, dispatch
from harness.native_worker import adapt_feature_diagnostics, count_model_objects, edge_order
from harness.port_budget import PacedLibrary, PortBudget, dynamic_range, occupied_ports
from harness.recipe import validate


def recipe() -> dict:
    return {
        "steps": [
            {
                "op": "sketch",
                "id": "profile",
                "plane": "XY",
                "items": [{"type": "circle", "args": [0, 0, 10]}],
            },
            {"op": "extrude", "id": "body", "sketch": "profile", "depth": 15},
        ]
    }


class NativeReadTests(unittest.TestCase):
    def test_count_does_not_materialize_objects_and_releases_vectors(self) -> None:
        api = Mock()
        api.IFeature_GetModelObjects.return_value = 3
        api.ObjectReferenceVector_Count.return_value = 550
        ks = Mock(ksapi=api)
        ks.ToInt32Vector.return_value = 2
        ks.GetReference.return_value = 4
        feature = Mock()
        feature.kompasApp.GetReference.return_value = 1
        self.assertEqual(count_model_objects(ks, feature, 9), 550)
        api.IFeature_GetModelObjects.assert_called_once_with(1, 4, 2)
        api.ObjectReferenceVector_Get.assert_not_called()
        api.ObjectReferenceVector_Delete.assert_called_once_with(1, 3)
        api.Int32Vector_Delete.assert_called_once_with(2)

    def test_edge_index_ignores_address_and_direction(self) -> None:
        edge = SimpleNamespace(
            kind="line",
            mid=(5.0, 19.3, 16.5),
            start=(5.0, 19.3, 0.0),
            end=(5.0, 19.3, 33.0),
            center=None,
            length=33.0,
            convexity="concave",
            key=7,
        )
        reverse = copy.copy(edge)
        reverse.start, reverse.end = edge.end, edge.start
        reverse.key = 8
        self.assertEqual(edge_order(edge), edge_order(reverse))


class PortBudgetTests(unittest.TestCase):
    def test_reads_configured_range_and_counts_unique_ipv4_ports(self) -> None:
        ports = dynamic_range(b"Start Port : 49152\r\nNumber of Ports : 16384\r\n")
        rows = (
            b"  TCP 127.0.0.1:49152 127.0.0.1:10705 TIME_WAIT\r\n"
            b"  TCP 127.0.0.1:49152 0.0.0.0:0 BOUND\r\n"
            b"  TCP 127.0.0.1:49153 127.0.0.1:10705 ESTABLISHED\r\n"
            b"  TCP 127.0.0.1:10705 127.0.0.1:49153 ESTABLISHED\r\n"
        )
        self.assertEqual(len(ports), 16384)
        self.assertEqual(occupied_ports(rows, ports), 2)

    @patch("harness.port_budget.sleep")
    def test_delays_native_call_until_ports_free_up(self, pause: Mock) -> None:
        sample = Mock(side_effect=[13_000, 12_000])
        method = Mock(return_value=42)
        dll = PacedLibrary(SimpleNamespace(query=method), PortBudget(range(49152, 65536), sample))
        self.assertEqual(dll.query(7), 42)
        self.assertEqual(dll.query(8), 42)
        self.assertEqual(sample.call_count, 2)
        pause.assert_called_once()
        self.assertEqual(method.call_count, 2)


class RecipeTests(unittest.TestCase):
    def test_valid_recipe(self) -> None:
        self.assertEqual(validate(recipe()), recipe())

    def test_disallow_escape_fields(self) -> None:
        for field, value in [
            ("path", "C:/Samples/model.m3d"),
            ("code", "import os"),
            ("method", "open"),
            ("__class__", "anything"),
        ]:
            with self.subTest(field=field):
                candidate = recipe()
                candidate["steps"][1][field] = value
                with self.assertRaises(ValueError):
                    validate(candidate)

    def test_no_open_or_eval_operation(self) -> None:
        for op in ("open", "exec", "eval", "save_as", "__getattribute__", "GetActiveDocument"):
            candidate = recipe()
            candidate["steps"][1]["op"] = op
            with self.subTest(op=op), self.assertRaises(ValueError):
                validate(candidate)

    def test_reject_foreign_references(self) -> None:
        for reference in ("../model.m3d", "body", "XY", "future", "__dict__", 3):
            candidate = recipe()
            candidate["steps"][1]["sketch"] = reference
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                validate(candidate)

    def test_reject_numbers_and_payloads(self) -> None:
        for value in (True, float("inf"), float("nan"), 1e20, "__import__('os')", {"path": "/tmp"}):
            candidate = recipe()
            candidate["steps"][1]["depth"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate(candidate)

    def test_contour_cannot_accept_extra_arguments(self) -> None:
        candidate = recipe()
        candidate["steps"][0]["items"][0]["filename"] = "/tmp/reference"
        with self.assertRaises(ValueError):
            validate(candidate)

    def test_typed_references_and_identifiers(self) -> None:
        for mutation in (
            lambda r: r["steps"][1].update(id="profile"),
            lambda r: r["steps"][1].update(id="../escape"),
            lambda r: r["steps"][0].update(plane="body"),
            lambda r: r.update(steps=r["steps"][::-1]),
        ):
            candidate = recipe()
            mutation(candidate)
            with self.assertRaises(ValueError):
                validate(candidate)


class FeatureErrorsTests(unittest.TestCase):
    def test_v25_uses_singular_error_code(self) -> None:
        class Feature:
            def __init__(self, code: int) -> None:
                self.code = code

            def GetObjectError(self) -> int:
                return self.code

        self.assertFalse(adapt_feature_diagnostics(Feature))
        for code in (0, 17):
            with self.subTest(code=code):
                feature = Feature(code)
                self.assertEqual(feature.IsHasError(), bool(code))
                self.assertEqual(feature.GetObjectErrors(), [code] if code else [])
                self.assertFalse(feature.IsHasWarning())

    def test_v26_keeps_native_checks(self) -> None:
        class Feature:
            def IsHasError(self) -> bool:
                return True

            def GetObjectErrors(self) -> list[int]:
                return [17, 25]

            def IsHasWarning(self) -> bool:
                return True

        self.assertTrue(adapt_feature_diagnostics(Feature))
        feature = Feature()
        self.assertTrue(feature.IsHasError())
        self.assertEqual(feature.GetObjectErrors(), [17, 25])
        self.assertTrue(feature.IsHasWarning())


class BridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        (self.run / "control").mkdir()
        (self.run / "control/settings.json").write_text("{}")
        self.bridge = Bridge(self.run)

    @patch("harness.builder_mcp.verify_bundle")
    @patch("harness.builder_mcp.Bridge.build")
    def test_bad_recipe_never_reaches_executor(self, build: object, verify: object) -> None:
        candidate = copy.deepcopy(recipe())
        candidate["steps"].append({"op": "open", "id": "foreign", "path": "reference.m3d"})
        with self.assertRaises(ValueError):
            self.bridge.call("build", {"recipe": candidate})
        build.assert_not_called()

    @patch("harness.builder_mcp.verify_bundle")
    @patch("harness.builder_mcp.Bridge.build")
    def test_timeout_quarantine_blocks_next_attempt(self, build: object, verify: object) -> None:
        (self.run / "control/uncertain.json").write_text("{}")
        with self.assertRaises(RuntimeError):
            self.bridge.call("build", {"recipe": recipe()})
        build.assert_not_called()

    def test_finish_is_not_a_path_interface(self) -> None:
        for attempt in (True, 0, -1, 31, "../reference", "/tmp", 1.5):
            with self.subTest(attempt=attempt), self.assertRaises(ValueError):
                self.bridge.finish(attempt)

    def test_mcp_exposes_only_expected_tools(self) -> None:
        response = dispatch(self.bridge, {"id": 1, "method": "tools/list"})
        self.assertEqual(
            {t["name"] for t in response["result"]["tools"]}, {"health", "build", "finish", "view_input"}
        )
        self.assertNotIn("resources", json.dumps(response))

    @patch("harness.builder_mcp.verify_bundle")
    @patch("harness.builder_mcp.Bridge.health", return_value={"content": []})
    def test_health_remains_available_during_quarantine(self, health: object, verify: object) -> None:
        (self.run / "control/uncertain.json").write_text("{}")
        self.bridge.call("health", {})
        health.assert_called_once()

    def test_only_unambiguous_ssh_errors_are_safe_to_retry(self) -> None:
        error = b"ssh: connect to host 10.211.55.3 port 22: Operation not permitted\r\n"
        self.assertTrue(connection_failed_before_execution(subprocess.CompletedProcess([], 255, b"", error)))
        for code, stdout, stderr in [
            (1, b"", error),
            (255, b"worker started", error),
            (255, b"", b"Connection closed by remote host"),
            (255, b"", error + b"remote worker log"),
        ]:
            with self.subTest(code=code, stdout=stdout, stderr=stderr):
                self.assertFalse(
                    connection_failed_before_execution(subprocess.CompletedProcess([], code, stdout, stderr))
                )


if __name__ == "__main__":
    unittest.main()
