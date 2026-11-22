"""Офлайн-тесты конфигурации эксперимента, проверки промптов и защиты от утечек данных."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.core.config import ExperimentConfig, load_experiment, resolve_prompts
from src.core.utils import ROOT, read_json, write_json
from src.evaluation.dataset import check_manifest
from src.experiments.runner import run_matrix, validate_config_pipelines
from src.llm.client import messages
from src.schemas.recipe import Recipe
from tests.conftest import FakePipelineModel, fake_cad_builder


def test_config_and_prompts_validation(sandbox: tuple[Path, ExperimentConfig]) -> None:
    root, cfg = sandbox
    validate_config_pipelines(cfg)
    assert len(cfg.variants) == 4
    assert len(cfg.comparisons) == 4
    assert len(cfg.prompt_snapshot) == 3

    (root / "prompts/cad_builder.md").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="пустой"):
        resolve_prompts(root, cfg)


def test_dataset_rejects_family_overlap_and_tampered_file(sandbox: tuple[Path, ExperimentConfig]) -> None:
    root, cfg = sandbox
    check_manifest(root, cfg.manifest, validation_size=1, require_val=True)
    manifest = read_json(root / cfg.manifest)
    manifest["items"][1]["family"] = manifest["items"][0]["family"]
    write_json(root / cfg.manifest, manifest)
    with pytest.raises(ValueError, match="Родственные"):
        check_manifest(root, cfg.manifest, validation_size=1, require_val=True)

    manifest["items"][1]["family"] = "family_val"
    write_json(root / cfg.manifest, manifest)
    (root / "data/input/VAL1.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="изменён"):
        check_manifest(root, cfg.manifest, validation_size=1, require_val=True)


@pytest.mark.parametrize("orientation", [None, "", "   ", {"bad": 1}])
def test_missing_orientation_stops_before_api(
    sandbox: tuple[Path, ExperimentConfig], orientation: object
) -> None:
    import asyncio

    root, cfg = sandbox
    manifest = read_json(root / cfg.manifest)
    manifest["items"][0]["orientation"] = orientation
    write_json(root / cfg.manifest, manifest)
    model = FakePipelineModel()
    with pytest.raises(ValueError, match="ориентация"):
        asyncio.run(
            run_matrix(
                root,
                cfg,
                "bad-orient",
                "dev",
                None,
                1,
                tracking=False,
                auto_evaluate=False,
                requester=model,
                builder=fake_cad_builder,
            )
        )
    assert not model.calls and not (root / "runs").exists()


@pytest.mark.parametrize("sheet_count", [None, 2, True])
def test_non_single_sheet_stops_before_api(
    sandbox: tuple[Path, ExperimentConfig], sheet_count: object
) -> None:
    import asyncio

    root, cfg = sandbox
    manifest = read_json(root / cfg.manifest)
    manifest["items"][0]["sheet_count"] = sheet_count
    write_json(root / cfg.manifest, manifest)
    model = FakePipelineModel()
    with pytest.raises(ValueError, match="ровно один лист"):
        asyncio.run(
            run_matrix(
                root,
                cfg,
                "bad-sheet",
                "dev",
                None,
                1,
                tracking=False,
                auto_evaluate=False,
                requester=model,
                builder=fake_cad_builder,
            )
        )
    assert not model.calls


@pytest.mark.parametrize("scale", [-1.0, 1.1])
def test_preparation_rejects_reflection_and_scaling(
    sandbox: tuple[Path, ExperimentConfig], scale: float
) -> None:
    root, cfg = sandbox
    manifest = read_json(root / cfg.manifest)
    matrix = np.eye(4)
    matrix[0, 0] = scale
    manifest["items"][0]["preparation"] = {"transform_to_target": matrix.tolist()}
    write_json(root / cfg.manifest, manifest)
    with pytest.raises(ValueError, match="только поворот и перенос"):
        check_manifest(root, cfg.manifest, validation_size=1, require_val=True)


def test_orientation_does_not_leak_target_metadata(sandbox: tuple[Path, ExperimentConfig]) -> None:
    import asyncio

    root, cfg = sandbox
    manifest = read_json(root / cfg.manifest)
    hidden = "TARGET-ONLY-SECRET-META"
    manifest["items"][0]["limitations"] = [hidden]
    manifest["items"][0]["review"]["evidence"] = hidden
    manifest["items"][0]["preparation"] = {"transform_to_target": np.eye(4).tolist(), "source_mesh": hidden}
    write_json(root / cfg.manifest, manifest)

    model = FakePipelineModel()
    asyncio.run(
        run_matrix(
            root,
            cfg,
            "leak-check",
            "dev",
            None,
            1,
            tracking=False,
            auto_evaluate=False,
            requester=model,
            builder=fake_cad_builder,
        )
    )
    for call in model.calls:
        serialized = json.dumps(messages(call["image"], call["text"], "high"), ensure_ascii=False)
        assert hidden not in serialized
        assert manifest["items"][0]["target_mesh"]["sha256"] not in serialized
        if call["image"] is None:
            assert call["output_type"] is Recipe
            assert "input_image" not in serialized


def test_modular_experiment_folder_configs_load_and_validate(tmp_path: Path) -> None:
    import shutil

    exp_dir = ROOT / "configs/experiments/01-drawing-passport"
    variant_names = [
        "01_baseline_direct",
        "02_passport_png",
        "03_passport_only",
        "04_notes_png",
    ]
    assert (exp_dir / "study.yaml").is_file()
    for variant in variant_names:
        for filename in ("pipeline.yaml", "llm.yaml", "cad.yaml", "evaluation.yaml"):
            assert (exp_dir / variant / filename).is_file()

    cfg_dir = load_experiment(exp_dir)
    cfg_entry = load_experiment(exp_dir / "study.yaml")
    assert cfg_dir.model_dump() == cfg_entry.model_dump()
    assert list(cfg_dir.variants) == variant_names
    assert cfg_dir.variants["01_baseline_direct"].pipeline == "direct"
    assert cfg_dir.variants["03_passport_only"].ablation_of == "02_passport_png"

    # Нарушение единых условий (llm или evaluation) между папками экспериментов отклоняется.
    llm_diff_dir = tmp_path / "01-llm-diff"
    shutil.copytree(exp_dir, llm_diff_dir)
    llm_text = (llm_diff_dir / "02_passport_png/llm.yaml").read_text(encoding="utf-8")
    (llm_diff_dir / "02_passport_png/llm.yaml").write_text(
        llm_text.replace("gpt-6.1-sol", "gpt-other"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Нарушено честное сравнение"):
        load_experiment(llm_diff_dir)

    eval_diff_dir = tmp_path / "01-eval-diff"
    shutil.copytree(exp_dir, eval_diff_dir)
    eval_text = (eval_diff_dir / "03_passport_only/evaluation.yaml").read_text(encoding="utf-8")
    (eval_diff_dir / "03_passport_only/evaluation.yaml").write_text(
        eval_text.replace("20261107", "20261108"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Нарушено честное сравнение"):
        load_experiment(eval_diff_dir)

    # Дублирование ключей между YAML-файлами одного варианта или symlink отклоняются.
    copy_dir = tmp_path / "01-dup"
    shutil.copytree(exp_dir, copy_dir)
    (copy_dir / "01_baseline_direct/cad.yaml").write_text(
        "agent: {}\ncad:\n  stl_tolerance_mm: 0.01\n  stl_angle_rad: 0.05\n", "utf-8"
    )
    with pytest.raises(ValueError, match="Дублирующиеся ключи"):
        load_experiment(copy_dir)

    sym_dir = tmp_path / "01-sym"
    shutil.copytree(exp_dir, sym_dir)
    target = tmp_path / "outside.yaml"
    target.write_text("cad:\n  stl_tolerance_mm: 0.01\n  stl_angle_rad: 0.05\n", encoding="utf-8")
    (sym_dir / "01_baseline_direct/cad.yaml").unlink()
    (sym_dir / "01_baseline_direct/cad.yaml").symlink_to(target)
    with pytest.raises(ValueError, match="символической ссылкой"):
        load_experiment(sym_dir)

    sym_var_dir = tmp_path / "01-sym-dir"
    shutil.copytree(exp_dir, sym_var_dir)
    outside_var = tmp_path / "outside-variant"
    shutil.copytree(exp_dir / "01_baseline_direct", outside_var)
    shutil.rmtree(sym_var_dir / "04_notes_png")
    (sym_var_dir / "04_notes_png").symlink_to(outside_var)
    with pytest.raises(ValueError, match="символической ссылкой"):
        load_experiment(sym_var_dir)
