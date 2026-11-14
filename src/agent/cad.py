"""Числовой рецепт → одно тело CadQuery → STL. Без LLM и внешних CAD-приложений."""

import math
import tempfile
from pathlib import Path

import cadquery as cq
import trimesh
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

from src.agent.schema import Circle, Recipe, Rectangle
from src.config import CadSettings
from src.utils import sha256


def bounds(shape: cq.Shape) -> list[list[float]]:
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape.wrapped, box, False, False)
    values = box.Get()
    return [list(values[:3]), list(values[3:])]


def build_shape(recipe: Recipe) -> cq.Shape:
    shape = None
    previous_volume = 0.0
    for index, step in enumerate(recipe.steps, 1):
        workplane = cq.Workplane("XY", origin=(0, 0, step.z))
        profile = step.profile
        if isinstance(profile, Circle):
            workplane = workplane.center(profile.x, profile.y).circle(profile.radius)
        elif isinstance(profile, Rectangle):
            workplane = workplane.center(profile.x + profile.width / 2, profile.y + profile.height / 2).rect(
                profile.width, profile.height
            )
        else:
            workplane = workplane.polyline([(p.x, p.y) for p in profile.points]).close()
        tool = workplane.extrude(step.depth if step.direction == "+Z" else -step.depth).val()
        shape = (
            tool
            if shape is None
            else (shape.fuse(tool).clean() if step.action == "add" else shape.cut(tool).clean())
        )
        volume = shape.Volume()
        if not shape.isValid() or len(shape.Solids()) != 1 or not math.isfinite(volume) or volume <= 0:
            raise ValueError(f"Шаг {index}: требуется одно корректное объёмное тело")
        change = volume - previous_volume if step.action == "add" else previous_volume - volume
        if change <= max(1e-8, previous_volume * 1e-10):
            raise ValueError(f"Шаг {index} не изменяет материал в указанном направлении")
        previous_volume = volume
    assert shape is not None
    if bounds(shape)[0][2] < -1e-6:
        raise ValueError("Деталь должна лежать в положительной полуплоскости Z")
    return shape


def build(recipe: Recipe, directory: Path, settings: CadSettings) -> dict:
    shape = build_shape(recipe)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        mesh_path = Path(temporary) / "model.stl"
        if not shape.exportStl(
            str(mesh_path),
            tolerance=settings.stl_tolerance_mm,
            angularTolerance=settings.stl_angle_rad,
            relative=False,
            parallel=False,
        ):
            raise ValueError("Не удалось сохранить STL")
        mesh = trimesh.load_mesh(mesh_path, process=True)
        if not mesh.is_volume:
            raise ValueError("STL не является замкнутой ориентированной объёмной сеткой")
        mesh_path.replace(directory / "model.stl")
    return {
        "solid_valid": True,
        "body_count": 1,
        "operations": len(recipe.steps),
        "volume_mm3": shape.Volume(),
        "bounds_mm": bounds(shape),
        "triangles": len(mesh.faces),
        "files": {"model.stl": sha256(directory / "model.stl")},
    }
