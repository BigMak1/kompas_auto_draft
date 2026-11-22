"""Офлайн-тесты Pydantic-схем Recipe и Passport (включая регрессию реального ответа)."""

import copy
import math

import pytest
from agents import AgentOutputSchema
from pydantic import ValidationError

from src.core.utils import ROOT, read_json, sha256
from src.llm.client import diagnose_output
from src.schemas.passport import Passport
from src.schemas.recipe import Polygon, Recipe
from tests.conftest import PASSPORT, RECIPE


@pytest.mark.parametrize(
    "change",
    [
        {"action": "cut"},
        {"z": -1},
        {"depth": 0},
        {"depth": float("nan")},
        {"profile": {"kind": "polygon", "points": [{"x": 0, "y": 0}] * 3}},
    ],
)
def test_invalid_recipe_rejected(change: dict) -> None:
    with pytest.raises(ValidationError):
        Recipe.model_validate({"steps": [RECIPE["steps"][0] | change]})


def test_recipe_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Recipe.model_validate(RECIPE | {"extra_field": 123})


def test_recipe_schema_removes_polygon_max_items_and_normalizes_points() -> None:
    schema = AgentOutputSchema(Recipe, strict_json_schema=True).json_schema()
    assert schema["properties"]["steps"]["maxItems"] == 64
    assert "maxItems" not in schema["$defs"]["Polygon"]["properties"]["points"]

    many = [
        {
            "x": round(10.0 * math.cos(2 * math.pi * i / 513), 4),
            "y": round(10.0 * math.sin(2 * math.pi * i / 513), 4),
        }
        for i in range(513)
    ]
    with pytest.raises(ValidationError):
        Polygon.model_validate({"kind": "polygon", "points": many})

    normalized = Polygon.model_validate(
        {
            "kind": "polygon",
            "points": [
                {"x": 0, "y": 0},
                {"x": 1e-9, "y": 0},
                {"x": 10, "y": 0},
                {"x": 5, "y": 10},
                {"x": 0, "y": 0},
            ],
        }
    )
    assert [(p.x, p.y) for p in normalized.points] == [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"unit": None}, "unit"),
        ({"unit": "kg"}, "unit"),
        ({"reading_status": "unresolved"}, "value"),
        ({"feature_id": None}, "feature_id"),
        ({"binding_status": "unresolved"}, "feature_id"),
    ],
)
def test_passport_dimension_evidence_rules(change: dict, field: str) -> None:
    value = copy.deepcopy(PASSPORT)
    value["dimensions"][0].update(change)
    with pytest.raises(ValidationError) as caught:
        Passport.model_validate(value)
    assert any(err["loc"] == ("dimensions", 0, field) for err in caught.value.errors())


def test_passport_checks_references_and_general_notes() -> None:
    bad_note = {**PASSPORT, "general_notes": [{"raw": "H14, h14", "view_id": "absent", "status": "observed"}]}
    with pytest.raises(ValidationError) as caught:
        Passport.model_validate(bad_note)
    assert caught.value.errors()[0]["loc"] == ("general_notes", 0, "view_id")
    assert caught.value.errors()[0]["type"] == "passport_reference"


def test_actual_saved_passport_fixture_regression() -> None:
    import json

    path = ROOT / "tests/fixtures/passport-response-v1.json"
    checksum = sha256(path)
    payload = read_json(path)
    output = json.loads(payload["output"][-1]["content"][0]["text"])
    output.update(schema_version="drawing-passport-2", general_notes=[])
    adapted = copy.deepcopy(payload)
    adapted["output"][-1]["content"][0]["text"] = json.dumps(output)

    details = diagnose_output(adapted, Passport)
    assert details is not None and details["code"] == "schema_validation"
    fields = {tuple(v["loc"]): v["msg"] for v in details["validation"]}
    assert "dimensionless" in fields[("dimensions", 34, "unit")]
    assert "general_notes" in fields[("dimensions", 45, "feature_id")]

    output["dimensions"][34]["unit"] = "dimensionless"
    general = output["dimensions"].pop(45)
    output["general_notes"] = [{"raw": general["raw"], "view_id": general["view_id"], "status": "observed"}]
    Passport.model_validate(output)
    assert sha256(path) == checksum
