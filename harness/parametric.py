"""Проверяемые связи рецепта с настоящими параметрами сохранённой детали."""

import ast
import dataclasses
import math
from typing import Any

from harness.expressions import expression


def describe_variable(variable: Any) -> dict:
    # asdict сначала deepcopy(raw): это повреждает native-дескриптор KsAPI.
    return {f.name: getattr(variable, f.name) for f in dataclasses.fields(variable) if f.name != "raw"}


def add_variables(builder: Any, declarations: list[dict], values: dict[str, float]) -> None:
    for item in declarations:
        value = item["value"]
        if isinstance(value, dict):
            value = value["expr"].strip()
            # В библиотеке строка-литерал не устанавливается после AddVariable(0).
            if not any(isinstance(node, ast.Name) for node in ast.walk(ast.parse(value, mode="eval"))):
                value = values[item["name"]]
        builder.vars.add(item["name"], value, item.get("note", ""))
        actual = builder.vars.get(item["name"])
        if not math.isclose(actual.value, values[item["name"]], rel_tol=1e-8, abs_tol=1e-7):
            raise ValueError(f"Variable {item['name']}: CAD value differs from recipe")


def bind_parameters(builder: Any, owner: Any, step: dict, values: dict[str, float]) -> dict:
    """dN обозначает принятый авторазмер эскиза; vN — параметр текущего объекта."""
    constraints = getattr(owner, "constraints", None)
    aliases = {f"d{i}": dim[0] for i, dim in enumerate(constraints.dimensions)} if constraints else {}
    # offset_plane возвращает сырой IPlane3DByOffset, не Step с полем raw.
    parameter_owner = step["id"] if step["op"] == "offset_plane" else owner
    links = []
    for target, expr in step.get("bindings", {}).items():
        selected = aliases.get(target, target)
        parameter = builder.vars.parameter(parameter_owner, selected)
        if parameter.information:
            raise ValueError(f"{step['id']}.{target}: informational dimension cannot drive geometry")
        expected = expression(expr, values)
        # Не даём ошибочному селектору незаметно изменить исходную геометрию.
        if not math.isclose(parameter.value, expected, rel_tol=1e-8, abs_tol=1e-6):
            raise ValueError(
                f"{step['id']}.{target}: initial parameter {parameter.value:g} != expression {expected:g}; "
                "inspect parameter metadata and correct geometry or selector"
            )
        builder.vars.bind(parameter_owner, parameter.name, expr.strip())
        actual = builder.vars.parameter(parameter_owner, parameter.name)
        if not math.isclose(actual.value, expected, rel_tol=1e-8, abs_tol=1e-6):
            raise ValueError(f"{step['id']}.{target}: CAD did not apply binding")
        links.append({"selector": target, "variable": actual.name, "expression": actual.expression})
    result: dict[str, Any] = {"op": step["op"], "bindings": links}
    if step["op"] == "sketch":
        result["state"] = owner.state
        result["auto_constraints"] = dataclasses.asdict(constraints) if constraints else None
        result["dimensions"] = [
            {"selector": f"d{i}", "variable": name, "initial_value": value, "description": description}
            for i, (name, value, description) in enumerate(constraints.dimensions if constraints else [])
        ]
    result["parameters"] = [describe_variable(v) for v in builder.vars.list(parameter_owner)]
    return result
