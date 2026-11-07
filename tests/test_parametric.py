"""Безопасная арифметика, ограничения схемы и работа с native-дескрипторами."""

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from harness.expressions import expression, resolve, variables
from harness.parametric import bind_parameters, describe_variable
from harness.recipe import validate
from harness.run import configuration, require_unstarted


def parameterized_recipe() -> dict:
    return {
        "variables": [{"name": "diameter", "value": 20}, {"name": "height", "value": 10}],
        "steps": [
            {
                "op": "sketch",
                "id": "profile",
                "plane": "XY",
                "constrain": True,
                "items": [{"type": "circle", "args": [0, 0, {"expr": "diameter/2"}]}],
                "bindings": {"d0": "diameter"},
            },
            {
                "op": "extrude",
                "id": "blank",
                "sketch": "profile",
                "depth": {"expr": "height"},
                "bindings": {"Расстояние 1": "height"},
            },
        ],
    }


class ExpressionTests(unittest.TestCase):
    def test_declared_dependencies_and_geometry(self) -> None:
        values = variables(
            [{"name": "diameter", "value": 20}, {"name": "radius", "value": {"expr": "diameter/2"}}]
        )
        self.assertEqual(values, {"diameter": 20, "radius": 10})
        self.assertEqual(expression("-(diameter-4)/2", values), -8)
        self.assertEqual(
            resolve({"args": [0, {"expr": "radius"}], "id": "radius"}, values),
            {"args": [0, 10], "id": "radius"},
        )
        self.assertEqual(validate(parameterized_recipe()), parameterized_recipe())

    def test_reject_code_and_unbounded_computation(self) -> None:
        for expr in (
            "open('/tmp/a')",
            "x.real",
            "x[0]",
            "x**10",
            "x//2",
            "abs(x)",
            "x%2",
            "unknown",
            "x/0",
            "1e309",
            "100000*2",
            "x;exit()",
            "'hello'",
            "x+" * 90,
        ):
            with self.subTest(expr=expr), self.assertRaises(ValueError):
                expression(expr, {"x": 2})

    def test_no_forward_refs_cycles_or_reserved_names(self) -> None:
        for declarations in (
            [{"name": "x", "value": {"expr": "y"}}, {"name": "y", "value": 1}],
            [{"name": "x", "value": {"expr": "x"}}],
            [{"name": "v12", "value": 1}],
            [{"name": "x", "value": 1}, {"name": "x", "value": 2}],
        ):
            with self.subTest(declarations=declarations), self.assertRaises(ValueError):
                variables(declarations)

    def test_bindings_require_real_variable_and_constraints(self) -> None:
        for change in (
            {"constrain": False},
            {"bindings": {"d0": "20"}},
            {"bindings": {"d0": "unknown"}},
            {"bindings": {"d0": {"expr": "diameter"}}},
        ):
            candidate = parameterized_recipe()
            candidate["steps"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(candidate)

    def test_limit_complex_auto_constraints(self) -> None:
        candidate = parameterized_recipe()
        candidate["steps"][0]["items"] *= 17
        with self.assertRaises(ValueError):
            validate(candidate)

    def test_serialization_never_copies_native_handle(self) -> None:
        class Native:
            def __deepcopy__(self, memo: dict) -> None:
                raise AssertionError("Native handle must never be copied")

        @dataclasses.dataclass
        class Variable:
            name: str
            value: float
            raw: object

        self.assertEqual(describe_variable(Variable("d", 20, Native())), {"name": "d", "value": 20})

    def test_raw_offset_plane_is_queried_by_feature_name(self) -> None:
        api = Mock()
        api.list.return_value = []
        bind_parameters(SimpleNamespace(vars=api), object(), {"op": "offset_plane", "id": "base_plane"}, {})
        api.list.assert_called_once_with("base_plane")


class RunHistoryTests(unittest.TestCase):
    def test_mcp_authorization_is_scoped_and_preserves_sandbox(self) -> None:
        run = Path("/test/runs/e2e/kf0011-test")
        config = configuration(run)
        server = config["mcp_servers"]["kompas-builder"]
        self.assertEqual(set(server["tools"]), set(server["enabled_tools"]))
        self.assertEqual({rule["approval_mode"] for rule in server["tools"].values()}, {"approve"})
        self.assertFalse(config["permissions.kf0011.network.enabled"])
        self.assertEqual(config["approval_policy"], "never")
        self.assertEqual(config["permissions.kf0011.filesystem"]["/Users"], "deny")
        self.assertEqual(config["permissions.kf0011.filesystem"][str(run / "builder/work")], "write")

    def test_manual_runs_cannot_be_resealed_or_relaunched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "raw").mkdir()
            manifest = run / "manifest.json"
            manifest.write_text(json.dumps({"status": "prepared"}))
            require_unstarted(run)
            (run / "raw/attempt-0001").mkdir()
            with self.assertRaises(RuntimeError):
                require_unstarted(run)

    def test_evaluated_run_is_immutable_without_started_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "manifest.json").write_text(json.dumps({"status": "evaluated"}))
            with self.assertRaises(RuntimeError):
                require_unstarted(run)


if __name__ == "__main__":
    unittest.main()
