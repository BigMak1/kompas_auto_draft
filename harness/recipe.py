"""Контракт ограниченного моста: данные вместо исполняемого кода и путей."""

import re
from typing import Any

from harness.expressions import expression, scalar, variables

OPS = {
    "offset_plane": ({"base", "offset"}, {"reverse"}),
    "sketch": ({"plane", "items"}, {"constrain"}),
    "extrude": ({"sketch", "depth"}, {"reverse", "depth2", "symmetric", "draft"}),
    "cut": ({"sketch"}, {"depth", "reverse", "depth2", "symmetric", "through", "draft"}),
    "revolve": ({"sketch"}, {"angle", "cut", "reverse", "angle2"}),
    "fillet": ({"edges", "radius"}, {"chain"}),
    "chamfer": ({"edges", "distance"}, {"distance2", "angle", "chain"}),
    "circular_pattern": ({"steps", "axis", "count"}, {"angle"}),
}
ITEMS = {"line": 4, "circle": 3, "arc": 5, "rect": 4, "polygon": None}
BOOLS = {"reverse", "symmetric", "through", "cut", "chain", "constrain"}
REFS = {"base": "plane", "plane": "plane", "sketch": "sketch"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(recipe: Any) -> dict:
    """Проверить весь запрос до любого обращения к КОМПАС."""
    require(
        isinstance(recipe, dict) and {"steps"} <= recipe.keys() <= {"steps", "variables"},
        "Expected steps and optional variables",
    )
    values = variables(recipe.get("variables", []))
    steps = recipe["steps"]
    require(isinstance(steps, list) and 1 <= len(steps) <= 100, "steps: 1..100")
    ids: dict[str, str] = {"XY": "plane", "XZ": "plane", "YZ": "plane"}
    for step in steps:
        require(isinstance(step, dict), "Step must be an object")
        op, name = step.get("op"), step.get("id")
        require(isinstance(op, str) and op in OPS, "Unsupported operation")
        require(isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,39}", name), "Invalid id")
        require(name not in ids, "Duplicate id")
        required, optional = OPS[op]
        require(required <= step.keys(), f"Missing fields for {op}")
        require(step.keys() <= required | optional | {"op", "id", "bindings"}, f"Unexpected fields for {op}")
        bindings = step.get("bindings", {})
        require(isinstance(bindings, dict) and len(bindings) <= 40, "Invalid bindings")
        if op == "sketch" and bindings:
            require(step.get("constrain") is True, "Sketch bindings require constrain: true")
        for target, expr in bindings.items():
            require(isinstance(target, str) and 1 <= len(target) <= 80, "Invalid parameter selector")
            expression(expr, values)
            require(
                any(re.search(r"\b" + re.escape(name) + r"\b", expr) for name in values),
                "Binding must use a declared variable",
            )
        for key, value in step.items():
            if key in {"op", "id", "items", "bindings"}:
                continue
            if key in REFS:
                require(isinstance(value, str) and ids.get(value) == REFS[key], f"Invalid {key}")
            elif key in BOOLS:
                require(type(value) is bool, f"{key} must be boolean")
            elif key == "axis":
                require(value in ("X", "Y", "Z"), "axis: X/Y/Z")
            elif key == "steps":
                require(isinstance(value, list) and 1 <= len(value) <= 100, "Invalid pattern steps")
                require(all(isinstance(v, str) and ids.get(v) == "solid" for v in value), "Invalid step ref")
            elif key == "edges":
                require(isinstance(value, list) and 1 <= len(value) <= 500, "Invalid edges")
                require(all(type(v) is int and 0 <= v < 10000 for v in value), "Invalid edge index")
            else:
                resolved = scalar(value, values)
                if key == "count":
                    require(float(resolved).is_integer() and 2 <= resolved <= 100, "count: integer 2..100")
        if op == "sketch":
            items = step["items"]
            require(isinstance(items, list) and 1 <= len(items) <= 1000, "items: 1..1000")
            if step.get("constrain"):
                expanded = sum(
                    4
                    if i.get("type") == "rect"
                    else len(i.get("args", []))
                    if i.get("type") == "polygon"
                    else 1
                    for i in items
                    if isinstance(i, dict)
                )
                require(expanded <= 16, "Auto constraints: at most 16 expanded sketch elements")
            for item in items:
                require(isinstance(item, dict), "Invalid contour item")
                require({"type", "args"} <= item.keys() <= {"type", "args", "style", "ccw"}, "Item fields")
                kind, args = item["type"], item["args"]
                require(isinstance(kind, str) and kind in ITEMS, "Unknown contour item")
                require(item.get("style", "normal") in ("normal", "axis", "construction"), "Invalid style")
                require("ccw" not in item or (kind == "arc" and type(item["ccw"]) is bool), "ccw: arc only")
                require(isinstance(args, list), "args must be a list")
                if kind == "polygon":
                    require(3 <= len(args) <= 1000, "polygon: 3..1000 vertices")
                    require(
                        all(isinstance(p, list) and len(p) == 2 for p in args),
                        "Invalid polygon points",
                    )
                    for point in args:
                        for value in point:
                            scalar(value, values)
                else:
                    require(len(args) == ITEMS[kind], "Invalid item args")
                    for value in args:
                        scalar(value, values)
        ids[name] = "plane" if op == "offset_plane" else "sketch" if op == "sketch" else "solid"
    require("solid" in ids.values(), "Recipe must contain a solid operation")
    return recipe
