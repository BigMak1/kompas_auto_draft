"""Параметры эксперимента из единственного YAML-файла."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AgentSettings(Settings):
    model: str = Field(min_length=1)
    reasoning_effort: Literal["low", "medium", "high"]
    timeout_seconds: float = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    image_detail: Literal["low", "high", "auto"]
    image_max_side: int | None = Field(default=None, ge=1)
    prompt_version: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class CadSettings(Settings):
    stl_tolerance_mm: float = Field(gt=0)
    stl_angle_rad: float = Field(gt=0)


class EvaluationSettings(Settings):
    surface_samples: int = Field(ge=32)
    seed: int = Field(ge=0)
    tau_relative: float = Field(gt=0, le=1)
    intersection_roundoff_relative: float = Field(gt=0, le=1e-4)


class Config(Settings):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    agent: AgentSettings
    cad: CadSettings
    evaluation: EvaluationSettings


def load_config(path: Path = ROOT / "configs/baseline.yaml") -> Config:
    return Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
