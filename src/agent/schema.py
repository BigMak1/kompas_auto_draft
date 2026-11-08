"""Схема ответа агента: замкнутый профиль XY и выдавливание вдоль Z."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Coordinate = Annotated[float, Field(ge=-1000, le=1000, allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, le=1000, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


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
    points: Annotated[list[Point], Field(min_length=3, max_length=128)]

    @model_validator(mode="after")
    def nonzero_area(self) -> Self:
        pairs = list(zip(self.points, self.points[1:] + self.points[:1], strict=True))
        if any(a == b for a, b in pairs) or abs(sum(a.x * b.y - b.x * a.y for a, b in pairs)) < 1e-8:
            raise ValueError("Многоугольник должен иметь площадь и различные соседние вершины")
        return self


class Step(StrictModel):
    action: Literal["add", "cut"]
    profile: Circle | Rectangle | Polygon
    z: Annotated[float, Field(ge=0, le=1000, allow_inf_nan=False)]
    depth: Positive
    direction: Literal["+Z", "-Z"]


class Recipe(StrictModel):
    steps: Annotated[list[Step], Field(min_length=1, max_length=40)]

    @model_validator(mode="after")
    def starts_with_material(self) -> Self:
        if self.steps[0].action != "add":
            raise ValueError("Первый шаг должен добавлять материал")
        return self
