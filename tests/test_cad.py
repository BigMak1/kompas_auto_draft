"""Офлайн-тесты построения CAD (CadQuery) и изолированного воркера с таймаутом."""

import asyncio
import math
from pathlib import Path

import pytest
import trimesh

from src.cad import builder, worker
from src.core.config import CadSettings
from src.schemas.recipe import Recipe
from tests.conftest import RECIPE


def test_real_cad_outputs_only_valid_stl(tmp_path: Path) -> None:
    recipe = Recipe.model_validate(
        {
            "steps": [
                RECIPE["steps"][0],
                RECIPE["steps"][0]
                | {"action": "cut", "profile": {"kind": "circle", "x": 0, "y": 0, "radius": 0.5}},
            ]
        }
    )
    settings = CadSettings(stl_tolerance_mm=0.01, stl_angle_rad=0.05)
    result = builder.build(recipe, tmp_path, settings)
    expected_vol = (4.0 - math.pi * 0.25) * 2.0
    assert result["volume_mm3"] == pytest.approx(expected_vol, rel=1e-6)
    assert trimesh.load_mesh(tmp_path / "model.stl").is_volume
    assert [p.name for p in tmp_path.iterdir()] == ["model.stl"]


@pytest.mark.parametrize(
    "steps",
    [
        # Разъединяющее добавление (2 отдельных тела).
        [
            RECIPE["steps"][0],
            RECIPE["steps"][0]
            | {"action": "add", "profile": {"kind": "circle", "x": 20, "y": 0, "radius": 1}},
        ],
        # Сквозной разрез, делящий деталь на 2 тела.
        [
            RECIPE["steps"][0],
            {
                "action": "cut",
                "profile": {"kind": "rectangle", "x": -0.2, "y": -2, "width": 0.4, "height": 4},
                "z": 0,
                "depth": 2,
                "direction": "+Z",
            },
        ],
        # Вырез, полностью уничтожающий объём.
        [RECIPE["steps"][0], RECIPE["steps"][0] | {"action": "cut"}],
        # Выход материала в отрицательную область Z < 0.
        [RECIPE["steps"][0] | {"direction": "-Z", "z": 1.0, "depth": 2.0}],
        # Самопересекающийся полигон.
        [
            {
                "action": "add",
                "profile": {
                    "kind": "polygon",
                    "points": [
                        {"x": 0, "y": 0},
                        {"x": 20, "y": 10},
                        {"x": 20, "y": 0},
                        {"x": 5, "y": 5},
                        {"x": 0, "y": 10},
                    ],
                },
                "z": 0,
                "depth": 5,
                "direction": "+Z",
            }
        ],
        # Касательный вырез по наружной стенке, дающий незамкнутый STL.
        [
            {
                "action": "add",
                "profile": {"kind": "circle", "x": 0, "y": 0, "radius": 10},
                "z": 0,
                "depth": 10,
                "direction": "+Z",
            },
            {
                "action": "cut",
                "profile": {"kind": "circle", "x": 5, "y": 0, "radius": 5},
                "z": 0,
                "depth": 10,
                "direction": "+Z",
            },
        ],
    ],
)
def test_invalid_topology_and_nonmanifold_rejected_as_generation(tmp_path: Path, steps: list[dict]) -> None:
    settings = CadSettings(stl_tolerance_mm=0.01, stl_angle_rad=0.05)
    recipe = Recipe.model_validate({"steps": steps})
    with pytest.raises(ValueError):
        builder.build(recipe, tmp_path / "direct", settings)
    res = asyncio.run(
        worker.build_recipe(recipe.model_dump(), tmp_path / "dest", tmp_path / "tech", settings, timeout=30.0)
    )
    assert res["status"] == "failed"
    assert res["failure_kind"] == "generation"


@pytest.mark.parametrize(
    ("action", "profile"),
    [
        ("add", {"kind": "rectangle", "x": -0.5, "y": -0.5, "width": 1, "height": 1}),
        ("cut", {"kind": "circle", "x": 20, "y": 0, "radius": 1}),
    ],
)
def test_overlapping_noop_steps_succeed_and_record_noop_steps(
    tmp_path: Path, action: str, profile: dict
) -> None:
    settings = CadSettings(stl_tolerance_mm=0.01, stl_angle_rad=0.05)
    recipe = Recipe.model_validate(
        {"steps": [RECIPE["steps"][0], RECIPE["steps"][0] | {"action": action, "profile": profile}]}
    )
    info = builder.build(recipe, tmp_path, settings)
    assert info["noop_steps"] == [2]
    assert info["volume_mm3"] == pytest.approx(8.0, rel=1e-6)
    assert trimesh.load_mesh(tmp_path / "model.stl").is_volume


def test_polygon_micro_duplicate_and_repeated_start_build_valid_stl(tmp_path: Path) -> None:
    settings = CadSettings(stl_tolerance_mm=0.01, stl_angle_rad=0.05)
    recipe = Recipe.model_validate(
        {
            "steps": [
                {
                    "action": "add",
                    "profile": {
                        "kind": "polygon",
                        "points": [
                            {"x": 0, "y": 0},
                            {"x": 1e-9, "y": 0},
                            {"x": 10, "y": 0},
                            {"x": 5, "y": 10},
                            {"x": 0, "y": 0},
                        ],
                    },
                    "z": 0,
                    "depth": 4,
                    "direction": "+Z",
                }
            ]
        }
    )
    info = builder.build(recipe, tmp_path, settings)
    assert info["noop_steps"] == []
    assert info["volume_mm3"] == pytest.approx(200.0, rel=1e-6)
    assert trimesh.load_mesh(tmp_path / "model.stl").is_volume


def test_cad_worker_builds_cleans_intermediates_and_obeys_timeout(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    tech = tmp_path / "debug/cad"
    dest.mkdir(parents=True)
    settings = CadSettings(stl_tolerance_mm=0.01, stl_angle_rad=0.05)

    res = asyncio.run(worker.build_recipe(RECIPE, dest, tech, settings, timeout=30.0))
    assert res["status"] == "success"
    assert trimesh.load_mesh(dest / "model.stl").is_volume
    assert not (tech / "model.stl").exists()
    assert not (tech / "build-input.json").exists()
    assert not (tech / "build-result.json").exists()
    assert (tech / "cad.log").is_file()

    with pytest.raises(TimeoutError):
        asyncio.run(worker.build_recipe(RECIPE, dest, tmp_path / "debug/timeout", settings, timeout=0.001))
