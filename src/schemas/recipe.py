"""Схема рецепта CAD: замкнутый профиль XY и выдавливание вдоль Z."""

import math
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from src.schemas.base import StrictModel

Coordinate = Annotated[float, Field(ge=-1000, le=1000, allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, le=1000, allow_inf_nan=False)]
MAX_POLYGON_POINTS = 512
POLYGON_MERGE_EPS_MM = 1e-5
MAX_RECIPE_STEPS = 64


class Point(StrictModel):
    x: Coordinate
    y: Coordinate


class Circle(StrictModel):
    kind: Literal["circle"]
    x: Coordinate
    y: Coordinate
    radius: Positive


class Rectangle(StrictModel):
    kind: Literal["rectangle"]
    x: Coordinate
    y: Coordinate
    width: Positive
    height: Positive


class Polygon(StrictModel):
    kind: Literal["polygon"]
    points: Annotated[
        list[Point],
        Field(
            min_length=3,
            description=(
                "Вершины простого замкнутого контура по обходу (3..512 точек); "
                "для мелких зубьев/фасок используй circle или компактную полную аппроксимацию."
            ),
        ),
    ]

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        if len(self.points) > MAX_POLYGON_POINTS:
            raise ValueError(f"Многоугольник не может содержать более {MAX_POLYGON_POINTS} вершин")
        cleaned: list[Point] = []
        for point in self.points:
            if (
                not cleaned
                or math.hypot(point.x - cleaned[-1].x, point.y - cleaned[-1].y) >= POLYGON_MERGE_EPS_MM
            ):
                cleaned.append(point)
        while (
            len(cleaned) > 1
            and math.hypot(cleaned[-1].x - cleaned[0].x, cleaned[-1].y - cleaned[0].y) < POLYGON_MERGE_EPS_MM
        ):
            cleaned.pop()
        if len(cleaned) < 3:
            raise ValueError("Многоугольник должен иметь ненулевую площадь и не менее 3 различных вершин")
        pairs = list(zip(cleaned, cleaned[1:] + cleaned[:1], strict=True))
        if abs(sum(a.x * b.y - b.x * a.y for a, b in pairs)) < 1e-8:
            raise ValueError("Многоугольник должен иметь ненулевую площадь и не менее 3 различных вершин")
        self.points = cleaned
        return self


class Step(StrictModel):
    action: Literal["add", "cut"]
    profile: Circle | Rectangle | Polygon
    z: Annotated[float, Field(ge=0, le=1000, allow_inf_nan=False)]
    depth: Positive
    direction: Literal["+Z", "-Z"]


class Recipe(StrictModel):
    steps: Annotated[list[Step], Field(min_length=1, max_length=MAX_RECIPE_STEPS)]

    @model_validator(mode="after")
    def starts_with_material(self) -> Self:
        if self.steps[0].action != "add":
            raise ValueError("Первый шаг рецепта должен добавлять материал (action='add')")
        return self
