"""Офлайн-тесты MLflow 3.18: Parent/Child runs, concurrent Agents SDK tracing, feedback и переносимость."""

import asyncio
import json
import shutil
import sqlite3
from pathlib import Path

import httpx2 as httpx
import pytest
from mlflow.entities import ViewType
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from src.core.config import ExperimentConfig
from src.core.utils import read_json
from src.experiments.runner import run_matrix
from src.experiments.store import purge_study_files
from src.llm import client as llm_client
from src.tracking.mlflow import (
    backfill_trace_sessions,
    database_path,
    local_tracking,
    normalize_portable_paths,
    purge_study_from_mlflow,
    tracking_uri,
)
from tests.conftest import PASSPORT, RECIPE, fake_cad_builder


def test_mlflow_parent_child_concurrent_tracing_feedback_and_portability(
    sandbox: tuple[Path, ExperimentConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-proj-offline-tracing-secret"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    def respond(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        is_passport = "drawing-passport-2" in body.get("instructions", "")
        is_notes = "обычные текстовые заметки" in body.get("instructions", "")
        text = (
            "Заметки для построения"
            if is_notes
            else json.dumps(PASSPORT if is_passport else RECIPE, ensure_ascii=False)
        )
        return httpx.Response(
            200,
            json={
                "id": "resp_trace",
                "object": "response",
                "created_at": 0,
                "model": "offline",
                "status": "completed",
                "tools": [],
                "output": [
                    {
                        "id": "msg_trace",
                        "role": "assistant",
                        "type": "message",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 120, "output_tokens": 45, "total_tokens": 165},
            },
        )

    monkeypatch.setattr(llm_client, "AsyncOpenAI", AsyncOpenAI)
    monkeypatch.setattr(
        llm_client,
        "DefaultAsyncHttpxClient",
        lambda **kw: DefaultAsyncHttpxClient(transport=httpx.MockTransport(respond), **kw),
    )

    root, cfg = sandbox
    orig_cwd = Path.cwd()
    monkeypatch.chdir(root)
    try:
        directory = asyncio.run(
            run_matrix(
                root,
                cfg,
                "mlflow-e2e",
                "val",
                None,
                repeats=1,
                concurrency=4,
                tracking=True,
                auto_evaluate=True,
                builder=fake_cad_builder,
            )
        )
        assert not (root / "logs/mlflow-e2e/tracking-error.log").exists()
        assert not list(root.glob("mlflow/*.db-wal"))
        assert not list(root.glob("mlflow/*.db-shm"))

        # Проверяем, что в tracking.db нет абсолютных путей к root.
        with sqlite3.connect(database_path(root)) as conn:
            exp_locs = conn.execute(
                "SELECT experiment_id, artifact_location FROM experiments WHERE experiment_id != 0"
            ).fetchall()
            run_uris = conn.execute("SELECT artifact_uri FROM runs").fetchall()
            trace_locs = conn.execute(
                "SELECT value FROM trace_tags WHERE key='mlflow.artifactLocation'"
            ).fetchall()
            assert all(loc == "mlflow/artifacts" for _, loc in exp_locs)
            assert all(uri_val.startswith("mlflow/artifacts/") for (uri_val,) in run_uris)
            assert all(not val.startswith("/") for (val,) in trace_locs)

        # Переносим sandbox в новое место (старый путь физически исчезает).
        monkeypatch.chdir(orig_cwd)
        relocated = root.with_name(root.name + "-moved")
        shutil.move(str(root), str(relocated))
    except BaseException:
        monkeypatch.chdir(orig_cwd)
        raise
    try:
        monkeypatch.chdir(relocated)
        normalize_portable_paths(relocated)
        with local_tracking(relocated):
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=tracking_uri(relocated), registry_uri=tracking_uri(relocated))
            exp = client.get_experiment_by_name(f"{cfg.name} · {cfg.title}")
            assert exp is not None
            active_exps = client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
            assert [e.experiment_id for e in active_exps] == [exp.experiment_id]

            # 1 Parent Run + 4 Child Runs = 5 runs всего.
            runs = client.search_runs([exp.experiment_id])
            assert len(runs) == 5
            parent = next(r for r in runs if r.data.tags.get("mlflow_role") == "study")
            children = [r for r in runs if r.data.tags.get("mlflow_role") == "variant"]
            assert len(children) == 4
            assert all(c.data.tags.get("mlflow.parentRunId") == parent.info.run_id for c in children)
            ordered_children = client.search_runs(
                [exp.experiment_id],
                filter_string="tags.mlflow_role = 'variant'",
                order_by=["attributes.start_time DESC"],
            )
            assert [r.data.tags["mlflow.runName"] for r in ordered_children] == [
                "01_baseline_direct",
                "02_passport_png",
                "03_passport_only",
                "04_notes_png",
            ]
            assert all(c.data.metrics["iou_mean"] == pytest.approx(1.0) for c in children)
            assert all(c.data.metrics["f_score_mean"] == pytest.approx(1.0) for c in children)

            # На Child Runs есть eval_results.json и prompts/*.md; на Parent Run есть report.md и графики.
            parent_arts = {a.path for a in client.list_artifacts(parent.info.run_id)}
            assert {"report.md", "runs.csv", "summary.json", "charts", "geometry"} <= parent_arts
            report_local = Path(client.download_artifacts(parent.info.run_id, "report.md"))
            assert "Таблица экспериментов" in report_local.read_text(encoding="utf-8")

            for child in children:
                child_arts = {a.path for a in client.list_artifacts(child.info.run_id)}
                assert "eval_results.json" in child_arts
                assert "prompts" in child_arts
                eval_tbl = read_json(Path(client.download_artifacts(child.info.run_id, "eval_results.json")))
                assert len(eval_tbl["data"]) == 1

            # Проверяем трассы (4 trial-трассы на child runs + 2 shared reading-трассы на parent run = 6).
            traces = client.search_traces(locations=[exp.experiment_id])
            assert len(traces) == 6
            child_ids = {c.info.run_id for c in children}
            trial_traces = [t for t in traces if t.info.trace_metadata.get("mlflow.sourceRun") in child_ids]
            reading_traces = [
                t for t in traces if t.info.trace_metadata.get("mlflow.sourceRun") == parent.info.run_id
            ]
            assert len(trial_traces) == 4 and len(reading_traces) == 2

            # На каждой trial-трассе есть assessments (iou, f_score, valid), промпт, session и нет утечки.
            for tr in traces:
                assert tr.info.trace_metadata.get("mlflow.trace.session") == "mlflow-e2e/VAL1-r01"
            session_traces = client.search_traces(
                locations=[exp.experiment_id],
                filter_string="metadata.`mlflow.trace.session` = 'mlflow-e2e/VAL1-r01'",
            )
            assert len(session_traces) == 6
            for tr in trial_traces:
                names = {a.name for a in tr.info.assessments}
                assert {"iou", "f_score", "valid"} <= names
                assert tr.info.tags.get("mlflow.linkedPrompts")
                span_names = {s.name for s in tr.data.spans}
                assert "recipe_generation" in span_names and "cad_build" in span_names
                serialized = json.dumps([s.to_dict() for s in tr.data.spans], ensure_ascii=False)
                assert secret not in serialized and "data:image/png;base64," not in serialized

            # Проверяем промпты в Registry.
            for p_name in ("cad-builder", "drawing-passport", "drawing-notes"):
                versions = client.search_prompt_versions(p_name)
                assert len(versions) == 1

            # Проверяем, что повторная синхронизация не плодит дубликаты промптов и оценок.
            from src.experiments.report import observations
            from src.experiments.store import verify_study
            from src.tracking.mlflow import sync_study_results

            moved_dir = relocated / "runs" / directory.name
            study = verify_study(moved_dir)
            rows = observations(relocated, moved_dir, study)
            summary = read_json(relocated / "results" / directory.name / "summary.json")
            sync_study_results(
                relocated, moved_dir, study, rows, summary, relocated / "results" / directory.name
            )
            assert len(client.search_prompt_versions("cad-builder")) == 1
            assert len(client.get_trace(trial_traces[0].info.trace_id).info.assessments) == 3

        # Проверяем реальный WSGI-сервер MLflow UI после переноса директории.
        from mlflow.server import app, handlers

        monkeypatch.setattr(handlers, "_tracking_store", None)
        monkeypatch.setattr(handlers, "_model_registry_store", None)
        handlers.initialize_backend_stores(
            backend_store_uri=tracking_uri(relocated),
            registry_store_uri=tracking_uri(relocated),
            default_artifact_root="mlflow/artifacts",
        )
        web = app.test_client()
        resp_exp = web.get("/ajax-api/2.0/mlflow/experiments/search?max_results=10")
        assert resp_exp.status_code == 200
        resp_report = web.get(f"/get-artifact?path=report.md&run_id={parent.info.run_id}")
        assert resp_report.status_code == 200
        assert "Таблица экспериментов" in resp_report.data.decode("utf-8")
        resp_trace = web.get(f"/ajax-api/3.0/mlflow/traces/{trial_traces[0].info.trace_id}")
        assert resp_trace.status_code == 200
    finally:
        monkeypatch.chdir(orig_cwd)
        shutil.rmtree(relocated, ignore_errors=True)


def test_purge_study_removes_only_target_study_and_keeps_default_hidden(
    sandbox: tuple[Path, ExperimentConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cfg = sandbox
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-offline-purge-secret")
    from tests.conftest import FakePipelineModel

    model = FakePipelineModel()
    orig_cwd = Path.cwd()
    monkeypatch.chdir(root)
    try:
        for study_name in ("study-old", "study-keep"):
            asyncio.run(
                run_matrix(
                    root,
                    cfg,
                    study_name,
                    "val",
                    None,
                    repeats=1,
                    concurrency=4,
                    tracking=True,
                    auto_evaluate=True,
                    requester=model,
                    builder=fake_cad_builder,
                )
            )
        stats = purge_study_from_mlflow(root, "study-old")
        removed = purge_study_files(root, "study-old")
        assert stats["runs"] == 5 and stats["traces"] == 6
        assert "runs/study-old" in removed and "results/study-old" in removed
        assert not (root / "runs/study-old").exists()
        assert not (root / "results/study-old").exists()
        assert not (root / "logs/study-old").exists()
        assert not (root / "logs/.locks/study-old.lock").exists()
        assert (root / "runs/study-keep").is_dir() and (root / "results/study-keep").is_dir()

        with sqlite3.connect(database_path(root)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 5
            assert conn.execute("SELECT COUNT(*) FROM trace_info").fetchone()[0] == 6
            assert conn.execute("SELECT COUNT(*) FROM entity_associations").fetchone()[0] == 6
            assert (
                conn.execute("SELECT lifecycle_stage FROM experiments WHERE experiment_id = 0").fetchone()[0]
                == "deleted"
            )
            conn.execute("UPDATE trace_info SET session_id = NULL")
            conn.commit()

        assert backfill_trace_sessions(root) == 6
        with local_tracking(root):
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=tracking_uri(root), registry_uri=tracking_uri(root))
            assert "0" not in [
                e.experiment_id for e in client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
            ]
            found = client.search_traces(
                locations=[client.get_experiment_by_name(f"{cfg.name} · {cfg.title}").experiment_id],
                filter_string="metadata.`mlflow.trace.session` = 'study-keep/VAL1-r01'",
            )
            assert len(found) == 6
    finally:
        monkeypatch.chdir(orig_cwd)
