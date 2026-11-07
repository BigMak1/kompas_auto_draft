"""Вход оценки и нормализованный адаптер к снимкам R2 (не замена facts.json)."""

from typing import Annotated, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[Finite, Field(gt=0)]
Nonnegative = Annotated[Finite, Field(ge=0)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Vec3 = tuple[Finite, Finite, Finite]
Matrix3 = tuple[Vec3, Vec3, Vec3]


class DTO(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Config(DTO):
    surface_samples: int = Field(default=8192, ge=100)
    registration_samples: int = Field(default=2048, ge=100)
    edge_samples: int = Field(default=4096, ge=100)
    seed: int = Field(default=20261107, ge=0)
    tau_relative: Positive = 0.01
    tau_mm: Positive | None = None
    alignment: Literal["rigid", "fixed"] = "rigid"
    icp_iterations: int = Field(default=40, ge=1)
    icp_candidates: int = Field(default=8, ge=1, le=100)
    symmetry_steps: int = Field(default=12, ge=4, le=72)
    inertia_relative_gap: Positive = 0.01
    edge_angle_degrees: Positive = Field(default=30, le=180)
    expected_bodies: int = Field(default=1, ge=1)
    dimension_tolerance_mm: Positive = 0.01


class Facts(DTO):
    """Все длины в мм; оси — столбцы в системе координат исходной сетки."""

    source: Literal["kompas", "analytic_fixture"]
    source_version: str = Field(min_length=1)
    mesh_sha256: Digest
    mesh_units: Literal["mm", "cm", "m", "inch"]
    native_sha256: Digest | None = None
    units: Literal["mm"] = "mm"
    solid_valid: bool | None = None
    body_count: int | None = Field(default=None, ge=0)
    volume_mm3: Nonnegative | None = None
    area_mm2: Nonnegative | None = None
    mass_g: Nonnegative | None = None
    material: str | None = None
    center_mm: Vec3 | None = None
    principal_axes: Matrix3 | None = None
    principal_moments: Vec3 | None = None
    inertia_unit: Literal["g_mm2", "mm5"] | None = None
    tree_fingerprint: str | None = Field(default=None, min_length=1)
    document_fingerprint: str | None = Field(default=None, min_length=1)
    topology: dict[str, int] = Field(default_factory=dict)
    dimensions_mm: dict[str, list[Nonnegative]] = Field(default_factory=dict)
    measurements: dict[str, Finite] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_axes(self) -> Self:
        if self.principal_axes is not None:
            axes = np.array(self.principal_axes)
            if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6, rtol=0):
                raise ValueError("principal_axes должны быть ортонормальными")
        if self.principal_moments is not None and min(self.principal_moments) <= 0:
            raise ValueError("Главные моменты должны быть положительными")
        if any(value < 0 for value in self.topology.values()):
            raise ValueError("Числа топологии не могут быть отрицательными")
        return self


class ModelInput(DTO):
    mesh: str
    units: Literal["mm", "cm", "m", "inch"]  # STL не хранит единицы: поле обязательно.
    facts: str | None = None
    native: str | None = None
    edge_points: str | None = None  # JSON: точки B-rep, равномерные по длине, в единицах mesh.
    export_manifest: str | None = None  # Проверяемая запись выгрузки вычислителем.


class ExportConfig(DTO):
    chord_mm: Positive = 0.01
    angle: Positive = 0.05
    max_cells: int = Field(default=2_000_000, ge=1)


class Requirement(DTO):
    id: str = Field(min_length=1)
    kind: Literal["dimension", "constraint"]
    target: Finite
    tolerance: Nonnegative
    unit: Literal["mm", "deg", "ratio"]


class Case(DTO):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    family: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    reference: ModelInput
    prediction: ModelInput | None = None
    generation_status: Literal["success", "failed", "timeout", "invalid"] = "success"
    requirements: list[Requirement] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_requirements(self) -> Self:
        if self.generation_status == "success" and self.prediction is None:
            raise ValueError("У успешной попытки нужен prediction")
        ids = [r.id for r in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("Повтор id требования")
        return self


class Evaluation(DTO):
    config: Config = Field(default_factory=Config)
    export: ExportConfig = Field(default_factory=ExportConfig)
    cases: list[Case] = Field(min_length=1)
    boolean_results: str | None = None
    purpose: Literal["benchmark", "metric_validation"] = "benchmark"

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        ids = [c.id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Повтор id детали")
        return self


class BooleanResult(DTO):
    request_sha256: Digest
    backend: Literal["kompas_v25", "analytic_fixture"]
    backend_version: str = Field(min_length=1)
    status: Literal["ok", "error"]
    prediction_volume_mm3: Positive | None = None
    reference_volume_mm3: Positive | None = None
    intersection_volume_mm3: Nonnegative | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def complete_result(self) -> Self:
        if self.status == "ok" and any(
            v is None
            for v in (self.prediction_volume_mm3, self.reference_volume_mm3, self.intersection_volume_mm3)
        ):
            raise ValueError("Успешная булева операция должна вернуть все объёмы")
        if self.status == "error" and not self.reason:
            raise ValueError("Нужна причина ошибки ядра")
        return self
