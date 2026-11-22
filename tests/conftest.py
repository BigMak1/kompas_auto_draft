"""Общие фикстуры для полностью офлайн-тестирования (без обращений к OpenAI API)."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import trimesh
from PIL import Image

from src.core.config import ExperimentConfig, load_experiment, resolve_prompts
from src.core.utils import ROOT, sha256, write_json

RECIPE = {
    "steps": [
        {
            "action": "add",
            "profile": {"kind": "rectangle", "x": -1, "y": -1, "width": 2, "height": 2},
            "z": 0,
            "depth": 2,
            "direction": "+Z",
        }
    ]
}

PASSPORT = {
    "schema_version": "drawing-passport-2",
    "views": [{"id": "V1", "label": "Главный вид", "kind": "view", "description": "Прямоугольник"}],
    "coordinates": {"origin": "Центр", "axis_z": "Толщина", "front_view_id": "V1", "status": "observed"},
    "features": [
        {
            "id": "F1",
            "kind": "брусок",
            "description": "Основное тело",
            "view_ids": ["V1"],
            "status": "observed",
        }
    ],
    "dimensions": [
        {
            "id": "D1",
            "name": "Ширина",
            "raw": "2",
            "value": 2.0,
            "unit": "mm",
            "view_id": "V1",
            "feature_id": "F1",
            "reading_status": "observed",
            "binding_status": "observed",
        }
    ],
    "relations": [],
    "general_notes": [],
    "unresolved": [],
}


@pytest.fixture(autouse=True)
def block_real_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Блокирует любые случайные обращения к реальному OpenAI API в тестах."""
    import src.llm.client as llm_client

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Обращение к настоящему OpenAI API в офлайн-тестах запрещено")

    monkeypatch.setattr(llm_client, "AsyncOpenAI", forbidden)


@pytest.fixture
def sandbox(tmp_path: Path) -> tuple[Path, ExperimentConfig]:
    """Изолированный мини-репозиторий с 1 dev и 1 val парой и копией промптов/кода оценщика."""
    (tmp_path / "data/input").mkdir(parents=True)
    (tmp_path / "data/target").mkdir(parents=True)
    (tmp_path / "prompts").mkdir(parents=True)
    for rel in ("prompts/cad_builder.md", "prompts/drawing_passport.md", "prompts/drawing_notes.md"):
        (tmp_path / rel).write_bytes((ROOT / rel).read_bytes())

    # Копируем проверяемые файлы оценщика, чтобы verify_evaluator работал внутри sandbox.
    for rel in (
        "src/evaluation/metrics.py",
        "src/evaluation/dataset.py",
        "src/core/utils.py",
    ):
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / rel).read_bytes())

    items = []
    for pair_id, split, color, extents, z_shift in (
        ("DEV1", "dev", "white", [3, 3, 3], 1.5),
        ("VAL1", "val", "black", [2, 2, 2], 1.0),
    ):
        png_path = tmp_path / f"data/input/{pair_id}.png"
        stl_path = tmp_path / f"data/target/{pair_id}.stl"
        Image.new("RGB", (16, 16), color).save(png_path)
        mesh = trimesh.creation.box(
            extents=extents,
            transform=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, z_shift], [0, 0, 0, 1]],
        )
        mesh.export(stl_path)
        items.append(
            {
                "pair_id": pair_id,
                "name": pair_id,
                "family": f"family_{split}",
                "split": split,
                "sheet_count": 1,
                "orientation": f"Ось Z, база Z=0, паз к +Y для {pair_id}.",
                "input_png": {"path": f"input/{pair_id}.png", "sha256": sha256(png_path)},
                "target_mesh": {"path": f"target/{pair_id}.stl", "sha256": sha256(stl_path)},
                "review": {"status": "accepted"},
            }
        )
    write_json(
        tmp_path / "data/manifest.json",
        {
            "dataset_id": "unit-sandbox-v1",
            "status": "ready",
            "coordinates": {"units": "mm"},
            "items": items,
        },
    )

    cfg = load_experiment(ROOT / "configs/experiments/01-drawing-passport")
    cfg = cfg.model_copy(
        update={
            "execution": cfg.execution.model_copy(
                update={"repeats": 1, "validation_size": 1, "concurrency": 4}
            )
        }
    )
    cfg = resolve_prompts(tmp_path, cfg)
    return tmp_path, cfg


class FakePipelineModel:
    """Детерминированная заглушка LLM-этапа с поддержкой имитации задержки и сбоев."""

    def __init__(self, *, delay: float = 0.0, reading_seconds: float | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.delay = delay
        self.reading_seconds = reading_seconds
        self.fail_mode: str | None = None

    async def __call__(
        self,
        *,
        settings: Any,
        instructions: str,
        image: str | None,
        text: str | None,
        output_type: Any,
        max_tokens: int,
        timeout: float,
        raw_directory: Path,
        trace_meta: dict | None = None,
        tracing: bool = False,
    ) -> dict:
        if self.delay:
            await asyncio.sleep(self.delay)
        self.calls.append(
            {
                "instructions": instructions,
                "image": image,
                "text": text,
                "output_type": output_type,
                "max_tokens": max_tokens,
                "timeout": timeout,
                "trace_meta": trace_meta,
                "tracing": tracing,
            }
        )
        raw_directory.mkdir(parents=True, exist_ok=True)
        write_json(raw_directory / "request.json", {"text": text, "has_image": image is not None})

        if self.fail_mode == "infrastructure":
            return {
                "status": "failed",
                "failure_kind": "infrastructure",
                "error": {"stage": "llm", "type": "ConnectError", "message": "API unavailable"},
                "usage": None,
                "seconds": 0.1,
            }
        if self.fail_mode == "generation":
            return {
                "status": "failed",
                "failure_kind": "generation",
                "failure_reason": "max_output_tokens",
                "error": {"stage": "llm", "type": "ModelBehaviorError", "message": "Truncated"},
                "usage": {"input_tokens": 100, "output_tokens": max_tokens},
                "seconds": 0.2,
            }

        is_reading = output_type is None or getattr(output_type, "__name__", "") == "Passport"
        out_tokens = 900 if is_reading else 700
        secs = self.reading_seconds if (is_reading and self.reading_seconds is not None) else 0.05
        if output_type is None:
            output: Any = "Заметки: ширина 2 мм, высота 2 мм, база Z=0."
        elif output_type.__name__ == "Passport":
            output = json.loads(json.dumps(PASSPORT))
        else:
            output = json.loads(json.dumps(RECIPE))
        return {
            "status": "success",
            "output": output,
            "usage": {"input_tokens": 200, "output_tokens": out_tokens},
            "seconds": secs,
        }


async def fake_cad_builder(
    recipe: dict, destination: Path, technical: Path, settings: Any, timeout: float
) -> dict:
    technical.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(
        extents=[2, 2, 2], transform=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
    )
    mesh.export(destination / "model.stl")
    return {
        "status": "success",
        "cad": {
            "solid_valid": True,
            "body_count": 1,
            "operations": len(recipe["steps"]),
            "volume_mm3": float(mesh.volume),
            "triangles": len(mesh.faces),
            "files": {"model.stl": sha256(destination / "model.stl")},
        },
    }
