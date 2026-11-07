"""Аналитические контрпримеры: шкала, знаменатели, локальные дефекты, пропуски."""

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from eval.geometry.alignment import align, rigid_candidates, validate_rigid
from eval.geometry.diagnostics import compare_facts, match_dimensions, mesh_diagnostics, requirement_metrics
from eval.geometry.io import atomic_json, read_json, sha256
from eval.geometry.models import BooleanResult, Case, Config, Evaluation, Facts, ModelInput, Requirement
from eval.geometry.runner import (
    copy_guard,
    evaluate_case,
    finalize_solid_results,
    load_model,
    metric,
    run_evaluation,
    summarize,
)
from eval.geometry.smoke import create_smoke, write_box
from eval.geometry.solid import evaluate_solid, iou_from_volumes, solid_request
from eval.geometry.surface import cloud_distances, coverage, distance_metrics, evaluate_surface


def facts(axes: np.ndarray | None = None, center: list | None = None) -> Facts:
    return Facts(
        source="analytic_fixture",
        source_version="test",
        mesh_sha256="a" * 64,
        mesh_units="mm",
        center_mm=center or [0, 0, 0],
        principal_axes=(axes if axes is not None else np.eye(3)).tolist(),
        principal_moments=[1, 2, 3],
        solid_valid=True,
        volume_mm3=30,
        body_count=1,
    )


def test_fscore_does_not_alias_siou() -> None:
    result = coverage(np.array([0] * 9 + [2]), np.array([0] + [2] * 9), 1)
    assert result["siou"] == pytest.approx(0.5)
    assert result["f_score"] == pytest.approx(0.18)


def test_distance_units_and_tail() -> None:
    result = distance_metrics(np.array([2.0]), np.array([3.0]))
    assert result["chamfer_l1_mm"] == 5
    assert result["chamfer_l2_mm2"] == 13  # без множителя 1/2
    outlier = np.array([0.0] * 99 + [10.0])
    result = distance_metrics(outlier, outlier)
    assert result["hausdorff_mm"] == 10
    assert result["hausdorff95_mm"] == 0
    points = np.array([[1.0, 2, 3], [4.0, 0, 6]])
    assert distance_metrics(*cloud_distances(points, points))["chamfer_l2_mm2"] == 0


@pytest.mark.parametrize("values", [np.array([]), np.array([np.nan]), np.array([-1.0])])
def test_invalid_distances_rejected(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        coverage(values, np.array([0.0]), 1)


def test_point_to_surface_identity_and_scale_error() -> None:
    mesh = trimesh.creation.box(extents=[2, 3, 5])
    config = Config(surface_samples=256, tau_mm=0.01)
    same, _ = evaluate_surface(mesh, mesh, config, 6)
    assert same["f_score"] == 1
    assert same["point_to_surface"]["chamfer_l2_mm2"] < 1e-20
    scaled = mesh.copy()
    scaled.apply_scale(1.2)
    different, _ = evaluate_surface(scaled, mesh, config, 6)
    assert different["f_score"] < 0.1


def test_rotation_translation_removed_without_scale() -> None:
    reference = trimesh.creation.box(extents=[2, 3, 5])
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", [32, 17, 41], degrees=True).as_matrix()
    matrix[:3, 3] = [50, -20, 9]
    prediction = reference.copy()
    prediction.apply_transform(matrix)
    config = Config(registration_samples=128, surface_samples=128, icp_candidates=4, icp_iterations=10)
    transform, _ = align(
        prediction, reference, facts(matrix[:3, :3], matrix[:3, 3].tolist()), facts(), config
    )
    validate_rigid(transform)
    prediction.apply_transform(transform)
    result, _ = evaluate_surface(prediction, reference, config, 6)
    assert result["f_score"] == 1
    assert result["point_to_surface"]["hausdorff_mm"] < 1e-8


def test_degenerate_axes_and_improper_transforms() -> None:
    standard = rigid_candidates(facts(), facts(), Config())
    symmetric = rigid_candidates(
        facts(), facts().model_copy(update={"principal_moments": (1, 1, 3)}), Config()
    )
    assert len(standard) == 24
    assert len(symmetric) > len(standard)
    for matrix in symmetric:
        validate_rigid(matrix)
    for bad in (np.diag([-1.0, 1, 1, 1]), np.diag([2.0, 2, 2, 1])):
        with pytest.raises(ValueError):
            validate_rigid(bad)


def test_rigid_never_guesses_axes_from_points() -> None:
    mesh = trimesh.creation.box()
    with pytest.raises(ValueError, match="R2"):
        align(mesh, mesh, None, None, Config())


def test_solid_iou_and_invalid_kernel_results() -> None:
    assert iou_from_volumes(8, 8, 8)["iou"] == 1
    assert iou_from_volumes(8, 8, 0)["iou"] == 0
    assert iou_from_volumes(8, 8, 4)["iou"] == pytest.approx(1 / 3)
    for values in ((8, 8, 9), (0, 8, 0), (8, 8, -1), (np.nan, 8, 0)):
        with pytest.raises(ValueError):
            iou_from_volumes(*values)


def test_boolean_evidence_binds_pose_and_models() -> None:
    request = solid_request("a", "aligned", "a" * 64, "b" * 64, "c" * 64, "d" * 64, np.eye(4))
    evidence = BooleanResult(
        request_sha256=request["request_sha256"],
        backend="kompas_v25",
        backend_version="25",
        status="ok",
        prediction_volume_mm3=8,
        reference_volume_mm3=8,
        intersection_volume_mm3=4,
    )
    results = {evidence.request_sha256: evidence}
    assert evaluate_solid(request, results, None, None, False)["value"] == pytest.approx(1 / 3)
    transform = np.eye(4)
    transform[0, 3] = 2
    stale = solid_request("a", "aligned", "a" * 64, "b" * 64, "c" * 64, "d" * 64, transform)
    assert evaluate_solid(stale, results, None, None, False)["status"] == "not_computed"
    assert evaluate_solid(request, results, facts(), None, False)["status"] == "error"


def test_mesh_defects_and_self_intersection() -> None:
    box = trimesh.creation.box()
    closed = mesh_diagnostics(box)
    assert closed["dangling_edge_length_mm"] == 0
    assert closed["flux_enclosure_error_mm2"] == 0
    assert closed["self_intersection_ratio"]["value"] == 0
    opened = box.copy()
    opened.update_faces(np.arange(len(opened.faces)) != 0)
    assert mesh_diagnostics(opened)["dangling_edge_length_mm"] > 0
    other = box.copy()
    other.apply_translation([0.25, 0.25, 0.25])
    intersection = mesh_diagnostics(trimesh.util.concatenate([box, other]))
    assert intersection["connected_segments"] == 2
    assert intersection["self_intersection_ratio"]["value"] > 0


def test_missing_dimensions_remain_in_denominator() -> None:
    checks = [
        Requirement(id="hole", kind="dimension", target=5, tolerance=0.1, unit="mm"),
        Requirement(id="parallel", kind="constraint", target=0, tolerance=0.01, unit="deg"),
    ]
    result = requirement_metrics(checks, {"parallel": 0})
    assert result["specification_compliance"] == 0.5
    assert result["g_csr_project"] == 1
    assert result["checks"][0]["absolute_error"] is None
    assert match_dimensions([10, 10], [10, 10, 10], 0)["recall"] == pytest.approx(2 / 3)
    assert match_dimensions([10], [10, 10], 0)["matched"] == 1


def test_auxiliary_progress_identifies_shared_work_and_dimension_groups() -> None:
    stages: list[str] = []
    mesh = trimesh.creation.box()
    config = Config(surface_samples=100)
    surface, _ = evaluate_surface(mesh, mesh, config, 1, stages.append)
    mesh_diagnostics(mesh, stages.append)
    pred = facts().model_copy(update={"dimensions_mm": {"linear": [1, 2], "radius": [3]}})
    ref = facts().model_copy(update={"dimensions_mm": {"linear": [1], "radius": [3, 4]}})
    comparison = compare_facts(pred, ref, np.eye(4), config, stages.append)
    assert surface["f_score"] == 1
    assert comparison == compare_facts(pred, ref, np.eye(4), config)
    assert any("CD, HD, SIoU" in stage for stage in stages)
    assert any("Normal Consistency" in stage for stage in stages)
    assert any("Self-Intersection Ratio" in stage for stage in stages)
    dimensions = [stage for stage in stages if stage.startswith("Размеры")]
    assert dimensions == [
        "Размеры 1/2 · линейные: 2 ↔ 1 значений",
        "Размеры 2/2 · радиусы: 1 ↔ 2 значений",
    ]


def test_copy_guard_and_incomplete_fingerprints() -> None:
    assert copy_guard(None, None, {"native": "a"}, {"native": "a"})["status"] == "rejected"
    assert copy_guard(None, None, {"native": "a"}, {"native": "b"})["status"] == "incomplete"
    pred = facts().model_copy(update={"tree_fingerprint": "a", "document_fingerprint": "a"})
    ref = facts().model_copy(update={"tree_fingerprint": "b", "document_fingerprint": "b"})
    assert copy_guard(pred, ref, {"native": "a"}, {"native": "b"})["status"] == "clear"


def test_aggregate_never_hides_missing_iou_or_failed_generation() -> None:
    rows = [
        {
            "eligible": True,
            "status": "evaluated",
            "valid_shape": True,
            "primary": {"solid_iou": metric(0.8), "f_score": metric(1.0)},
            "secondary": {},
        },
        {
            "eligible": True,
            "status": "invalid",
            "valid_shape": False,
            "primary": {"solid_iou": metric(0.0), "f_score": metric(0.0)},
            "secondary": {},
        },
    ]
    summary = summarize(rows)
    assert summary["primary"]["solid_iou"]["mean"] == 0.4
    assert summary["valid_shape_rate"] == 0.5
    assert summary["rankable"]
    rows[0]["primary"]["solid_iou"] = metric(None)
    summary = summarize(rows)
    assert summary["primary"]["solid_iou"]["mean"] is None
    assert not summary["rankable"]


def test_dtos_reject_ambiguous_inputs() -> None:
    with pytest.raises(ValidationError):
        ModelInput(mesh="input.stl")
    with pytest.raises(ValidationError):
        Config(tau_mm=float("nan"))
    with pytest.raises(ValidationError):
        Evaluation(cases=[])


def test_snapshot_hash_and_units(tmp_path: Path) -> None:
    model = write_box(tmp_path, "box", (2, 3, 5), np.eye(4))
    mesh, _, _ = load_model(ModelInput.model_validate(model), tmp_path, True)
    cm_mesh, _, _ = load_model(
        ModelInput.model_validate({**model, "units": "cm", "facts": None}), tmp_path, True
    )
    assert cm_mesh.extents == pytest.approx(mesh.extents * 10)
    with pytest.raises(ValueError, match="Единицы сетки"):
        load_model(ModelInput.model_validate({**model, "units": "cm"}), tmp_path, True)
    path = tmp_path / model["facts"]
    data = read_json(path)
    data["mesh_sha256"] = "b" * 64
    atomic_json(path, data)
    with pytest.raises(ValueError, match="другой сетке"):
        load_model(ModelInput.model_validate(model), tmp_path, True)


def test_invalid_solid_needs_no_registration(tmp_path: Path) -> None:
    ref = write_box(tmp_path, "ref", (2, 3, 5), np.eye(4))
    pred = write_box(tmp_path, "pred", (2, 3, 5), np.eye(4))
    data = read_json(tmp_path / pred["facts"])
    data.update(solid_valid=False, principal_axes=None)
    atomic_json(tmp_path / pred["facts"], data)
    case = Case(id="bad", family="box", reference=ModelInput(**ref), prediction=ModelInput(**pred))
    result, _ = evaluate_case(case, tmp_path, Config(), {}, True, tmp_path / "raw")
    assert result["status"] == "invalid"
    assert result["primary"]["solid_iou"]["value"] == 0


def test_missing_self_intersection_check_blocks_rating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = write_box(tmp_path, "ref", (2, 3, 5), np.eye(4))
    pred = write_box(tmp_path, "pred", (2, 3, 5), np.eye(4))
    case = Case(id="unverified", family="box", reference=ModelInput(**ref), prediction=ModelInput(**pred))
    config = Config(alignment="fixed", surface_samples=100, edge_samples=100)
    monkeypatch.setitem(sys.modules, "igl.copyleft.cgal", None)
    result, requests = evaluate_case(case, tmp_path, config, {}, True, tmp_path / "raw")
    evidence = {
        request["request_sha256"]: BooleanResult(
            request_sha256=request["request_sha256"],
            backend="analytic_fixture",
            backend_version="test",
            status="ok",
            prediction_volume_mm3=30,
            reference_volume_mm3=30,
            intersection_volume_mm3=30,
        )
        for request in requests
    }
    result, _ = evaluate_case(case, tmp_path, config, evidence, True, tmp_path / "raw")
    assert result["primary"]["solid_iou"]["value"] == 1
    assert result["primary"]["f_score"]["value"] == 1
    assert result["status"] == "evaluator_error"
    assert "проверка самопересечений" in result["error"]
    assert not summarize([result])["rankable"]


def test_full_pipeline_and_reproducibility(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    final = create_smoke(Path("smoke"))
    report = read_json(final / "metrics.json")
    assert report["summary"]["rankable"]
    cases = {r["id"]: r for r in report["cases"]}
    assert cases["rigid_box"]["primary"]["f_score"]["value"] == 1
    assert cases["rigid_box"]["primary"]["solid_iou"]["value"] == pytest.approx(1, abs=2e-5)
    assert cases["rigid_box"]["unaligned"]["solid"]["value"] == 0
    assert cases["wrong_width"]["primary"]["solid_iou"]["value"] < 0.84
    assert cases["wrong_width"]["primary"]["f_score"]["value"] < 1
    assert cases["failed_generation"]["primary"]["solid_iou"]["value"] == 0
    before = sha256(final / "metrics.json")
    again = run_evaluation(tmp_path / "smoke")
    assert sha256(again / "metrics.json") == before
    for name in ("manifest.json", "notes.md", "raw/solid_requests.json", "report.md"):
        assert (final / name).exists()


def test_finalization_only_updates_solids_and_preserves_preliminary(tmp_path: Path, monkeypatch) -> None:
    final = create_smoke(tmp_path / "smoke")
    preliminary = Path(read_json(final / "manifest.json")["reused_geometry"]["path"])
    previous_hash = sha256(preliminary / "metrics.json")
    evaluation = Evaluation.model_validate(read_json(tmp_path / "smoke/evaluation.json"))

    def forbid_geometry(*args, **kwargs) -> None:
        raise AssertionError("Повторная геометрия не нужна для новых ответов ядра")

    for name in ("align", "evaluate_surface", "mesh_diagnostics", "compare_facts", "load_model"):
        monkeypatch.setattr(f"eval.geometry.runner.{name}", forbid_geometry)
    updated = finalize_solid_results(tmp_path / "smoke", preliminary, evaluation)
    assert sha256(updated / "metrics.json") == sha256(final / "metrics.json")
    assert sha256(preliminary / "metrics.json") == previous_hash
    for original in (preliminary / "raw").glob("*/deviations.json"):
        assert sha256(updated / original.relative_to(preliminary)) == sha256(original)


def test_finalization_rejects_changed_inputs_or_settings(tmp_path: Path) -> None:
    final = create_smoke(tmp_path / "smoke")
    preliminary = Path(read_json(final / "manifest.json")["reused_geometry"]["path"])
    evaluation = Evaluation.model_validate(read_json(tmp_path / "smoke/evaluation.json"))
    changed = evaluation.model_copy(update={"config": evaluation.config.model_copy(update={"tau_mm": 0.1})})
    with pytest.raises(ValueError, match="Настройки"):
        finalize_solid_results(tmp_path / "smoke", preliminary, changed)
    mesh = tmp_path / "smoke" / evaluation.cases[0].reference.mesh
    mesh.write_bytes(mesh.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="изменился"):
        finalize_solid_results(tmp_path / "smoke", preliminary, evaluation)


def test_missing_metrics_explain_requirements_material_and_edges(tmp_path: Path) -> None:
    inputs = []
    for role in ("ref", "pred"):
        model = write_box(tmp_path, role, (2, 3, 5), np.eye(4))
        path = tmp_path / model["facts"]
        data = read_json(path)
        data.update(mass_g=30, inertia_unit="g_mm2", material=None)
        atomic_json(path, data)
        inputs.append(ModelInput(**model))
    case = Case(id="no_spec", family="box", reference=inputs[0], prediction=inputs[1])
    config = Config(alignment="fixed", surface_samples=100, edge_samples=100)
    report, _ = evaluate_case(
        case, tmp_path, config, {}, True, tmp_path / "raw"
    )
    metrics = report["secondary"]
    assert "requirements" in metrics["dimension_absolute_error_mm_mean"]["reason"]
    assert "материал" in metrics["facts.mass_g_relative_error"]["reason"]
    assert (
        metrics["facts.mass_g_relative_error"]["reason"]
        == metrics["facts.principal_moments_relative_error"]["reason"]
    )
    assert "B-rep" in metrics["edge_chamfer_brep_mm2"]["reason"]
    assert metrics["edge_chamfer_mesh_proxy_mm2"]["value"] is not None
