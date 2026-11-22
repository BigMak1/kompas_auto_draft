"""Числовой рецепт → одно тело CadQuery → замкнутая сетка STL."""

import math
import tempfile
from pathlib import Path

import cadquery as cq
import trimesh
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.Standard import Standard_Failure

from src.core.config import CadSettings
from src.core.utils import sha256
from src.schemas.recipe import Circle, Recipe, Rectangle


def bounds(shape: cq.Shape) -> list[list[float]]:
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape.wrapped, box, False, False)
    values = box.Get()
    return [list(values[:3]), list(values[3:])]


def _boolean_op(shape: cq.Shape, tool: cq.Shape, action: str) -> cq.Shape:
    raw = shape.fuse(tool) if action == "add" else shape.cut(tool)
    try:
        cleaned = raw.clean()
        volume = cleaned.Volume()
    except Standard_Failure:
        return raw
    if cleaned.isValid() and len(cleaned.Solids()) == 1 and math.isfinite(volume) and volume > 0:
        return cleaned
    return raw


def _build_shape_with_info(recipe: Recipe) -> tuple[cq.Shape, list[int]]:
    shape: cq.Shape | None = None
    previous_volume = 0.0
    noop_steps: list[int] = []
    for index, step in enumerate(recipe.steps, 1):
        try:
            workplane = cq.Workplane("XY", origin=(0, 0, step.z))
            profile = step.profile
            if isinstance(profile, Circle):
                workplane = workplane.center(profile.x, profile.y).circle(profile.radius)
            elif isinstance(profile, Rectangle):
                workplane = workplane.center(
                    profile.x + profile.width / 2, profile.y + profile.height / 2
                ).rect(profile.width, profile.height)
            else:
                workplane = workplane.polyline([(p.x, p.y) for p in profile.points]).close()
            tool = workplane.extrude(step.depth if step.direction == "+Z" else -step.depth).val()
            shape = tool if shape is None else _boolean_op(shape, tool, step.action)
            volume = shape.Volume()
        except (Standard_Failure, ZeroDivisionError) as error:
            raise ValueError(f"Шаг {index}: ошибка геометрии CadQuery/OCCT: {error}") from error
        if not shape.isValid() or len(shape.Solids()) != 1 or not math.isfinite(volume) or volume <= 0:
            raise ValueError(f"Шаг {index}: требуется одно корректное объёмное тело")
        change = volume - previous_volume if step.action == "add" else previous_volume - volume
        if change <= max(1e-8, previous_volume * 1e-10):
            noop_steps.append(index)
        previous_volume = volume
    assert shape is not None
    if bounds(shape)[0][2] < -1e-6:
        raise ValueError("Деталь должна лежать в положительной полуплоскости Z>=0")
    return shape, noop_steps


def build_shape(recipe: Recipe) -> cq.Shape:
    return _build_shape_with_info(recipe)[0]


def build(recipe: Recipe, directory: Path, settings: CadSettings) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    shape, noop_steps = _build_shape_with_info(recipe)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        mesh_path = Path(temporary) / "model.stl"
        try:
            exported = shape.exportStl(
                str(mesh_path),
                tolerance=settings.stl_tolerance_mm,
                angularTolerance=settings.stl_angle_rad,
                relative=False,
                parallel=False,
            )
        except Standard_Failure as error:
            raise ValueError(f"Ошибка экспорта STL: {error}") from error
        if not exported:
            raise ValueError("Не удалось экспортировать STL")
        mesh = trimesh.load_mesh(mesh_path, process=True)
        if not mesh.is_volume:
            raise ValueError("STL не является замкнутой ориентированной объёмной сеткой")
        mesh_path.replace(directory / "model.stl")
    return {
        "solid_valid": True,
        "body_count": 1,
        "operations": len(recipe.steps),
        "noop_steps": noop_steps,
        "volume_mm3": shape.Volume(),
        "bounds_mm": bounds(shape),
        "triangles": len(mesh.faces),
        "files": {"model.stl": sha256(directory / "model.stl")},
    }
