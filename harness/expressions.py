"""Ограниченная арифметика рецепта: числа, известные переменные, + - * / и скобки.

Python eval и выполнение кода не используются. Выражения проверяются до обращения к CAD.
"""

import ast
import math
import re
from typing import Any

NAME = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")


def number(value: Any) -> bool:
    return type(value) in (int, float) and abs(value) <= 100_000 and math.isfinite(value)


def expression(text: Any, values: dict[str, float]) -> float:
    if not isinstance(text, str) or not 1 <= len(text) <= 160:
        raise ValueError("Expression must be 1..160 characters")
    if not re.fullmatch(r"[a-z0-9_+*/(). \-]+", text):
        raise ValueError("Only arithmetic on declared variables is allowed")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Invalid expression") from exc
    if len(list(ast.walk(tree))) > 64:
        raise ValueError("Expression too complex")

    def read(node: ast.AST, depth: int = 0) -> float:
        if depth > 12:
            raise ValueError("Expression too deep")
        if isinstance(node, ast.Constant) and number(node.value):
            result = float(node.value)
        elif isinstance(node, ast.Name) and node.id in values:
            result = values[node.id]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = read(node.operand, depth + 1) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = read(node.left, depth + 1), read(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                result = a + b
            elif isinstance(node.op, ast.Sub):
                result = a - b
            elif isinstance(node.op, ast.Mult):
                result = a * b
            else:
                if b == 0:
                    raise ValueError("Division by zero")
                result = a / b
        else:
            raise ValueError("Unsupported expression or undeclared variable")
        if not number(result):
            raise ValueError("Expression result outside allowed range")
        return result

    return read(tree.body)


def scalar(value: Any, values: dict[str, float]) -> float | int:
    if number(value):
        return value
    if isinstance(value, dict) and set(value) == {"expr"}:
        return expression(value["expr"], values)
    raise ValueError("Expected bounded number or {expr: arithmetic}")


def variables(items: Any) -> dict[str, float]:
    if not isinstance(items, list) or len(items) > 40:
        raise ValueError("variables: up to 40 declarations")
    values: dict[str, float] = {}
    for item in items:
        if not isinstance(item, dict) or not {"name", "value"} <= item.keys() <= {"name", "value", "note"}:
            raise ValueError("Variable fields: name, value, optional note")
        name = item["name"]
        if not isinstance(name, str) or not NAME.fullmatch(name) or re.fullmatch(r"[vd][0-9]+", name):
            raise ValueError("Invalid or reserved variable name")
        if name in values:
            raise ValueError("Duplicate variable name")
        if not isinstance(item.get("note", ""), str) or len(item.get("note", "")) > 160:
            raise ValueError("Variable note: up to 160 characters")
        values[name] = float(scalar(item["value"], values))
    return values


def resolve(value: Any, values: dict[str, float]) -> Any:
    """Разрешить только явные expr-узлы; структурные строки остаются строками."""
    if isinstance(value, dict):
        if set(value) == {"expr"}:
            return scalar(value, values)
        return {k: resolve(v, values) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, values) for v in value]
    return value
