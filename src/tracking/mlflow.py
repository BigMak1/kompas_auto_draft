"""Интеграция с MLflow 3.18: Parent/Child runs, трассировка OpenAI Agents SDK, feedback и промпты."""

import asyncio
import contextvars
import os
import shutil
import sqlite3
from collections.abc import Awaitable, Callable, Iterator
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
from typing import Any

from src.core.config import ExperimentConfig
from src.core.logging import log, log_error
from src.core.utils import digest, error_details, redact

ARTIFACTS_RELATIVE = "mlflow/artifacts"


def database_path(root: Path) -> Path:
    return root / "mlflow/tracking.db"


def tracking_uri(root: Path) -> str:
    return "sqlite:///" + database_path(root).resolve().as_posix()


def prompt_registry_name(relative_path: str) -> str:
    return Path(relative_path).stem.replace("_", "-")


def trace_session_id(study_name: str, pair_id: str, repeat: int) -> str:
    return f"{study_name}/{pair_id}-r{repeat:02d}"


def hide_default_experiment(client: Any) -> bool:
    """Мягко скрывает пустой эксперимент Default (id=0), чтобы не загромождать MLflow UI."""
    from mlflow.entities import LifecycleStage, ViewType
    from mlflow.exceptions import MlflowException

    try:
        default_exp = client.get_experiment("0")
    except MlflowException as error:
        if error.error_code == "RESOURCE_DOES_NOT_EXIST":
            return False
        raise
    if default_exp is None or default_exp.lifecycle_stage != LifecycleStage.ACTIVE:
        return False
    if client.search_runs(["0"], run_view_type=ViewType.ALL, max_results=1):
        return False
    if client.search_traces(locations=["0"], max_results=1):
        return False
    client.delete_experiment("0")
    return True


def backfill_trace_sessions(root: Path) -> int:
    """Проставляет mlflow.trace.session для существующих трасс серии без перезаписи."""
    db = database_path(root)
    if not db.is_file():
        return 0
    updated = 0
    with closing(sqlite3.connect(db)) as conn, conn:
        if not conn.execute("PRAGMA table_info(trace_info)").fetchall():
            return 0
        rows = conn.execute(
            """
            SELECT t.request_id,
                   (SELECT value FROM trace_tags WHERE request_id = t.request_id AND key = 'study'),
                   (SELECT value FROM trace_tags WHERE request_id = t.request_id AND key = 'pair_id'),
                   (SELECT value FROM trace_tags WHERE request_id = t.request_id AND key = 'repeat')
            FROM trace_info t
            WHERE t.session_id IS NULL OR t.session_id = ''
            """
        ).fetchall()
        for trace_id, study_name, pair_id, repeat_raw in rows:
            if not (study_name and pair_id and repeat_raw):
                continue
            try:
                session = trace_session_id(study_name, pair_id, int(repeat_raw))
            except ValueError:
                continue
            conn.execute("UPDATE trace_info SET session_id = ? WHERE request_id = ?", (session, trace_id))
            updated += 1
    return updated


def sanitize_span(span: Any) -> None:
    """Фильтр перед сохранением span: убирает секреты и base64-изображения."""
    from mlflow.entities import SpanStatus

    span.set_inputs(redact(span.inputs))
    span.set_outputs(redact(span.outputs))
    for key, value in span.attributes.items():
        span.set_attribute(key, redact(value))
    for event in span.events:
        for key, value in event.attributes.items():
            event.attributes[key] = redact(value)
    if span.status.description:
        span.set_status(SpanStatus(span.status.status_code, redact(span.status.description)))


def normalize_portable_paths(root: Path) -> None:
    """Нормализует пути артефактов в SQLite до относительных и фиксирует WAL в tracking.db."""
    db = database_path(root)
    if not db.is_file():
        return
    root_resolved = root.resolve()
    root_str = root_resolved.as_posix().rstrip("/") + "/"
    root_uri = root_resolved.as_uri().rstrip("/") + "/"

    def to_relative(value: str | None) -> str | None:
        if not value:
            return value
        for prefix in (root_uri, root_str):
            if value.startswith(prefix):
                return value.removeprefix(prefix)
        return value

    with closing(sqlite3.connect(db)) as conn, conn:
        for table, column, where in (
            ("experiments", "artifact_location", "experiment_id != 0"),
            ("runs", "artifact_uri", "1=1"),
            ("trace_tags", "value", "key='mlflow.artifactLocation'"),
        ):
            if not conn.execute(f"PRAGMA table_info({table})").fetchall():
                continue
            rows = conn.execute(f"SELECT rowid, {column} FROM {table} WHERE {where}").fetchall()
            for rowid, val in rows:
                updated = to_relative(val)
                if updated != val:
                    conn.execute(f"UPDATE {table} SET {column}=? WHERE rowid=?", (updated, rowid))
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


@contextmanager
def local_tracking(root: Path) -> Iterator[Any]:
    """Нацеливает MLflow на локальную переносимую БД проекта и синхронный экспорт трасс."""
    (root / "mlflow/artifacts").mkdir(parents=True, exist_ok=True)
    saved_env = {
        key: os.environ.get(key)
        for key in (
            "MLFLOW_DISABLE_AGENT_HINT",
            "MLFLOW_DISABLE_TELEMETRY",
            "MLFLOW_TRACE_EXTRACT_ATTACHMENTS",
            "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING",
        )
    }
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["MLFLOW_TRACE_EXTRACT_ATTACHMENTS"] = "false"
    os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"

    import mlflow

    prev_tracking = mlflow.get_tracking_uri()
    prev_registry = mlflow.get_registry_uri()
    uri = tracking_uri(root)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    try:
        yield mlflow
    finally:
        mlflow.set_tracking_uri(prev_tracking)
        mlflow.set_registry_uri(prev_registry)
        for key, prev in saved_env.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def ensure_experiment(client: Any, root: Path, name: str) -> str:
    from mlflow.entities import LifecycleStage

    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        exp_id = client.create_experiment(name, artifact_location=ARTIFACTS_RELATIVE)
        with closing(sqlite3.connect(database_path(root))) as conn, conn:
            conn.execute(
                "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
                (ARTIFACTS_RELATIVE, int(exp_id)),
            )
    else:
        if experiment.lifecycle_stage == LifecycleStage.DELETED:
            client.restore_experiment(experiment.experiment_id)
        exp_id = experiment.experiment_id
    if name != "Default":
        hide_default_experiment(client)
    return exp_id


def sync_prompts(client: Any, config: ExperimentConfig) -> dict[str, dict[str, str]]:
    """Зеркалирует prompts/*.md в MLflow Prompt Registry без дублирования неизменённых текстов."""
    from mlflow.exceptions import MlflowException

    updated: dict[str, dict[str, str]] = {}
    for rel_path, info in config.prompt_snapshot.items():
        name = prompt_registry_name(rel_path)
        text = info["text"]
        checksum = info["sha256"]
        try:
            versions = client.search_prompt_versions(name, max_results=1000)
        except MlflowException as error:
            if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
            versions = []
        matched = next((v for v in versions if v.template == text), None)
        if matched is None:
            matched = client.register_prompt(
                name,
                text,
                commit_message=f"Sync {rel_path} ({checksum[:12]})",
                tags={"sha256": checksum, "source_path": rel_path},
            )
        updated[rel_path] = {
            **info,
            "mlflow_name": name,
            "mlflow_version": str(matched.version),
            "mlflow_uri": f"prompts:/{name}/{matched.version}",
        }
    return updated


def _common_params(study: dict) -> dict[str, str]:
    config = study["config"]
    params: dict[str, str] = {
        "experiment_name": config["name"],
        "split": study["split"],
        "repeats": str(study["repeats"]),
        "order_seed": str(config["execution"]["order_seed"]),
        "concurrency": str(config["execution"]["concurrency"]),
        "dataset_version": config["dataset_version"],
        "manifest_sha256": digest(study["manifest"])[:16],
        "reading.max_output_tokens": str(config["reading"]["max_output_tokens"]),
        "study_fingerprint": study["fingerprint"][:16],
    }
    for section in ("agent", "cad", "evaluation"):
        for key, value in config.get(section, {}).items():
            params[f"{section}.{key}"] = str(value)
    return params


def _log_missing_params(client: Any, run_id: str, params: dict[str, str]) -> None:
    existing = client.get_run(run_id).data.params
    for key, value in params.items():
        if key not in existing:
            client.log_param(run_id, key, value[:6000])


def ensure_study_and_variant_runs(
    client: Any,
    root: Path,
    experiment_id: str,
    study: dict,
    prompt_snapshot: dict[str, dict[str, str]],
) -> tuple[str, dict[str, str]]:
    """Создаёт или находит Parent Run серии и Child Runs для каждого варианта."""
    study_name = study["name"]
    config = study["config"]
    found_parent = client.search_runs(
        [experiment_id],
        filter_string=f"tags.study = '{study_name}' AND tags.mlflow_role = 'study'",
        max_results=2,
    )
    if found_parent:
        study_run_id = found_parent[0].info.run_id
    else:
        tags = {
            "mlflow.runName": study_name,
            "study": study_name,
            "mlflow_role": "study",
            "experiment_name": config["name"],
            "split": study["split"],
            "repeats": str(study["repeats"]),
            "formal_validation": str(study["formal_validation"]).lower(),
            "dataset_version": config["dataset_version"],
            "dataset_id": str(study["manifest"].get("dataset_id", "")),
            "git_commit": str(study.get("source_commit") or ""),
            "hypothesis": (config.get("hypothesis") or "")[:4000],
        }
        if config.get("number") is not None:
            tags["experiment_number"] = str(config["number"])
        study_run_id = client.create_run(experiment_id, tags=tags).info.run_id
    _log_missing_params(client, study_run_id, _common_params(study))

    parent_start = client.get_run(study_run_id).info.start_time
    variants_list = list(config["variants"].items())
    variant_runs: dict[str, str] = {}
    for idx, (variant_key, variant) in enumerate(variants_list):
        found_child = client.search_runs(
            [experiment_id],
            filter_string=(
                f"tags.study = '{study_name}' AND tags.mlflow_role = 'variant' "
                f"AND tags.variant = '{variant_key}'"
            ),
            max_results=2,
        )
        if found_child:
            variant_run_id = found_child[0].info.run_id
        else:
            child_tags = {
                "mlflow.runName": variant_key,
                "mlflow.parentRunId": study_run_id,
                "study": study_name,
                "mlflow_role": "variant",
                "variant": variant_key,
                "pipeline": variant["pipeline"],
                "role": variant["role"],
                "ablation_of": str(variant.get("ablation_of") or ""),
                "label": variant["label"],
                "hypothesis": (variant.get("hypothesis") or "")[:4000],
                "split": study["split"],
                "dataset_version": config["dataset_version"],
            }
            variant_run_id = client.create_run(
                experiment_id,
                start_time=parent_start + (len(variants_list) - idx),
                tags=child_tags,
            ).info.run_id

        variant_params = {
            **_common_params(study),
            "variant": variant_key,
            "pipeline": variant["pipeline"],
            "role": variant["role"],
            "ablation_of": str(variant.get("ablation_of") or "none"),
        }
        for role, rel_path in variant["prompts"].items():
            snap = prompt_snapshot[rel_path]
            variant_params[f"prompt.{role}.path"] = rel_path
            variant_params[f"prompt.{role}.sha256"] = snap["sha256"][:16]
            if snap.get("mlflow_uri"):
                variant_params[f"prompt.{role}.uri"] = snap["mlflow_uri"]
                client.link_prompt_version_to_run(variant_run_id, snap["mlflow_uri"])
        _log_missing_params(client, variant_run_id, variant_params)

        # Записываем тексты промптов варианта один раз в артефакты MLflow.
        child_tags_now = client.get_run(variant_run_id).data.tags
        prompts_digest = digest({r: prompt_snapshot[p]["sha256"] for r, p in variant["prompts"].items()})
        if child_tags_now.get("prompts_artifact_sha256") != prompts_digest:
            for rel_path in sorted(set(variant["prompts"].values())):
                prompt_file = root / rel_path
                if prompt_file.is_file():
                    client.log_artifact(variant_run_id, str(prompt_file), artifact_path="prompts")
            client.set_tag(variant_run_id, "prompts_artifact_sha256", prompts_digest)

        variant_runs[variant_key] = variant_run_id

    return study_run_id, variant_runs


class StudyTracker:
    """Управляет жизненным циклом MLflow (runs + concurrent tracing) на время выполнения серии."""

    def __init__(
        self,
        *,
        mlflow_mod: Any,
        client: Any,
        root: Path,
        study_dir: Path,
        study: dict,
        experiment_id: str,
        study_run_id: str,
        variant_runs: dict[str, str],
        prompt_snapshot: dict[str, dict[str, str]],
    ) -> None:
        self.mlflow = mlflow_mod
        self.client = client
        self.root = root
        self.study_dir = study_dir
        self.study = study
        self.experiment_id = experiment_id
        self.study_run_id = study_run_id
        self.variant_runs = variant_runs
        self.prompt_snapshot = prompt_snapshot

    @contextmanager
    def start_trial_trace(
        self,
        *,
        variant_key: str,
        pair_id: str,
        repeat: int,
        attempt: str,
        has_png: bool,
    ) -> Iterator[Any]:
        run_id = self.variant_runs[variant_key]
        with self.mlflow.start_span(
            name=f"{variant_key}/{pair_id}-r{repeat:02d}",
            span_type="CHAIN",
            run_id=run_id,
        ) as span:
            span.set_inputs(
                {
                    "variant": variant_key,
                    "pair_id": pair_id,
                    "repeat": repeat,
                    "attempt": attempt,
                    "has_png": has_png,
                }
            )
            self.mlflow.update_current_trace(
                tags={
                    "study": self.study["name"],
                    "variant": variant_key,
                    "pair_id": pair_id,
                    "repeat": str(repeat),
                    "attempt": attempt,
                    "stage": "trial",
                },
                metadata={"mlflow.trace.session": trace_session_id(self.study["name"], pair_id, repeat)},
            )
            yield span

    def finish_trial_trace(self, trace_ctx: Any, span: Any, record: dict) -> None:
        from mlflow.entities import SpanStatus, SpanStatusCode

        try:
            if span is not None:
                span.set_outputs(
                    {
                        "status": record.get("status"),
                        "failure_kind": record.get("failure_kind"),
                        "seconds": round(float(record.get("seconds", 0.0)), 3),
                        "usage": record.get("usage"),
                        "cad_builds": record.get("cad_builds"),
                        "reading_trace_id": record.get("reading_trace_id"),
                    }
                )
                if record.get("status") != "success":
                    err = record.get("error") or {}
                    span.set_status(
                        SpanStatus(
                            SpanStatusCode.ERROR,
                            redact(err.get("summary") or err.get("message") or "Ошибка маршрута"),
                        )
                    )
        except Exception as error:  # noqa: BLE001 — сбой трассировки не отменяет результат попытки.
            self._diagnose(error)
        finally:
            try:
                trace_ctx.__exit__(None, None, None)
            except Exception as error:  # noqa: BLE001
                self._diagnose(error)

    async def trace_cad_step(self, coro: Awaitable[dict], *, steps: int) -> dict:
        with self.mlflow.start_span(name="cad_build", span_type="TOOL") as span:
            span.set_inputs({"operations": steps})
            result = await coro
            span.set_outputs(
                {
                    "status": result.get("status"),
                    "cad": result.get("cad"),
                    "error": result.get("error"),
                }
            )
            return result

    def wrap_requester(self, raw_requester: Callable[..., Awaitable[dict]]) -> Callable[..., Awaitable[dict]]:
        async def traced_request(**kwargs: Any) -> dict:
            meta = kwargs.pop("trace_meta", None) or {}
            kind = meta.get("kind", "recipe")
            prompt_uri = meta.get("prompt_uri")

            if kind == "reading":
                # Общее чтение пишем в чистый изолированный контекст под Parent Run серии.
                async def run_reading() -> dict:
                    span_name = (
                        f"reading/{meta.get('stage')}/{meta.get('pair_id')}-r{meta.get('repeat', 0):02d}"
                    )
                    with self.mlflow.start_span(
                        name=span_name,
                        span_type="CHAIN",
                        run_id=self.study_run_id,
                    ) as span:
                        if prompt_uri:
                            self.mlflow.genai.load_prompt(
                                prompt_uri, link_to_model=False, cache_ttl_seconds=0
                            )
                        span.set_inputs(
                            {
                                "stage": meta.get("stage"),
                                "pair_id": meta.get("pair_id"),
                                "repeat": meta.get("repeat"),
                                "attempt": meta.get("attempt"),
                                "max_output_tokens": kwargs.get("max_tokens"),
                            }
                        )
                        result = await raw_requester(**kwargs, tracing=True)
                        span.set_outputs(
                            {
                                "status": result.get("status"),
                                "usage": result.get("usage"),
                                "error": result.get("error"),
                            }
                        )
                        pair_val = str(meta.get("pair_id", ""))
                        repeat_val = int(meta.get("repeat", 0))
                        self.mlflow.update_current_trace(
                            tags={
                                "study": self.study["name"],
                                "stage": "reading",
                                "reading_kind": str(meta.get("stage", "")),
                                "pair_id": pair_val,
                                "repeat": str(meta.get("repeat", "")),
                                "attempt": str(meta.get("attempt", "")),
                            },
                            metadata={
                                "mlflow.trace.session": trace_session_id(
                                    self.study["name"], pair_val, repeat_val
                                )
                            },
                        )
                        result["trace_id"] = span.trace_id
                        return result

                return await asyncio.create_task(run_reading(), context=contextvars.Context())

            # Этап построения рецепта выполняется внутри активной trial-трассы варианта.
            with self.mlflow.start_span(name="recipe_generation", span_type="CHAIN") as span:
                if prompt_uri:
                    self.mlflow.genai.load_prompt(prompt_uri, link_to_model=False, cache_ttl_seconds=0)
                span.set_inputs(
                    {
                        "variant": meta.get("variant"),
                        "pair_id": meta.get("pair_id"),
                        "max_output_tokens": kwargs.get("max_tokens"),
                        "has_png": kwargs.get("image") is not None,
                        "reading_trace_id": meta.get("reading_trace_id"),
                    }
                )
                result = await raw_requester(**kwargs, tracing=True)
                span.set_outputs(
                    {
                        "status": result.get("status"),
                        "usage": result.get("usage"),
                        "error": result.get("error"),
                    }
                )
                result["trace_id"] = span.trace_id
                return result

        return traced_request

    def _diagnose(self, error: Exception) -> None:
        log_error(
            error_details(error, "трассировка MLflow"),
            self.root / "logs" / self.study["name"] / "tracking-error.log",
        )


@contextmanager
def track_study_run(
    root: Path, study_dir: Path, study: dict
) -> Iterator[tuple[StudyTracker, ExperimentConfig]]:
    """Открывает контекст MLflow на время выполнения матрицы (runs + OpenAI autolog + checkpoint)."""
    config = ExperimentConfig.model_validate(study["config"])
    with ExitStack() as stack:
        mlflow_mod = stack.enter_context(local_tracking(root))
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri(root), registry_uri=tracking_uri(root))
        exp_name = f"{config.name} · {config.title}"
        experiment_id = ensure_experiment(client, root, exp_name)
        mlflow_mod.set_experiment(experiment_id=experiment_id)
        synced_prompts = sync_prompts(client, config)
        study_run_id, variant_runs = ensure_study_and_variant_runs(
            client, root, experiment_id, study, synced_prompts
        )
        stack.enter_context(mlflow_mod.tracing.configure(span_processors=[sanitize_span]))
        mlflow_mod.openai.autolog(log_traces=True, disable_openai_agent_tracer=True)
        tracker = StudyTracker(
            mlflow_mod=mlflow_mod,
            client=client,
            root=root,
            study_dir=study_dir,
            study=study,
            experiment_id=experiment_id,
            study_run_id=study_run_id,
            variant_runs=variant_runs,
            prompt_snapshot=synced_prompts,
        )
        log("MLFLOW", f"Parent run {study_run_id[:8]}… · эксперимент №{experiment_id}.")
        try:
            yield tracker, config.model_copy(update={"prompt_snapshot": synced_prompts})
        finally:
            try:
                mlflow_mod.flush_trace_async_logging()
            finally:
                mlflow_mod.openai.autolog(disable=True)
                normalize_portable_paths(root)


def _log_metric_safe(client: Any, run_id: str, key: str, value: float | int | None) -> None:
    if value is not None:
        client.log_metric(run_id, key, float(value))


def sync_study_results(
    root: Path,
    study_dir: Path,
    study: dict,
    rows: list[dict],
    summary: dict,
    results_dir: Path,
) -> None:
    """Идемпотентно записывает агрегаты, таблицы, графики и assessments трасс в MLflow."""
    from mlflow.entities import AssessmentSource
    from mlflow.exceptions import MlflowException

    from src.experiments.store import read_record

    with local_tracking(root) as mlflow_mod:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri(root), registry_uri=tracking_uri(root))
        config = ExperimentConfig.model_validate(study["config"])
        experiment_id = ensure_experiment(client, root, f"{config.name} · {config.title}")
        synced_prompts = sync_prompts(client, config)
        study_run_id, variant_runs = ensure_study_and_variant_runs(
            client, root, experiment_id, study, synced_prompts
        )

        # 1. Обновляем Parent Run (Study) сводными тегами, метриками и артефактами отчёта.
        client.set_tag(study_run_id, "complete", str(summary["complete"]).lower())
        client.set_tag(study_run_id, "measured_cases", f"{summary['measured']}/{summary['expected']}")
        _log_metric_safe(client, study_run_id, "measured_cases", summary["measured"])
        _log_metric_safe(client, study_run_id, "expected_cases", summary["expected"])
        actual = summary.get("physical_usage") or {}
        _log_metric_safe(client, study_run_id, "physical_seconds", actual.get("seconds"))
        _log_metric_safe(client, study_run_id, "physical_input_tokens", actual.get("input_tokens"))
        _log_metric_safe(client, study_run_id, "physical_output_tokens", actual.get("output_tokens"))

        report_digest = digest(summary)
        parent_tags = client.get_run(study_run_id).data.tags
        if parent_tags.get("report_digest") != report_digest and results_dir.is_dir():
            for filename in ("report.md", "runs.csv", "summary.json", "evaluations.json"):
                file = results_dir / filename
                if file.is_file():
                    client.log_artifact(study_run_id, str(file))
            for png in sorted(results_dir.glob("*.png")):
                dest = "geometry" if png.name.startswith("geometry-") else "charts"
                client.log_artifact(study_run_id, str(png), artifact_path=dest)
            client.set_tag(study_run_id, "report_digest", report_digest)
        client.set_terminated(study_run_id, status="FINISHED" if summary["complete"] else "RUNNING")

        # 2. Обновляем Child Runs по каждому варианту и прикрепляем assessments к трассам.
        source = AssessmentSource(source_type="CODE", source_id="offline_evaluator")
        for variant_key, variant_run_id in variant_runs.items():
            metrics = summary["variants"][variant_key]
            _log_metric_safe(client, variant_run_id, "iou_mean", metrics.get("iou"))
            _log_metric_safe(client, variant_run_id, "iou_std", metrics.get("iou_part_std"))
            _log_metric_safe(client, variant_run_id, "f_score_mean", metrics.get("f_score"))
            _log_metric_safe(client, variant_run_id, "f_score_std", metrics.get("f_score_part_std"))
            _log_metric_safe(client, variant_run_id, "validity", metrics.get("validity"))
            _log_metric_safe(client, variant_run_id, "seconds_mean", metrics.get("seconds_mean"))
            _log_metric_safe(client, variant_run_id, "input_tokens_mean", metrics.get("input_tokens_mean"))
            _log_metric_safe(client, variant_run_id, "output_tokens_mean", metrics.get("output_tokens_mean"))
            _log_metric_safe(client, variant_run_id, "measured_cases", metrics.get("measured"))
            _log_metric_safe(client, variant_run_id, "expected_cases", metrics.get("expected"))
            _log_metric_safe(
                client, variant_run_id, "generation_failures", metrics.get("generation_failures")
            )
            _log_metric_safe(client, variant_run_id, "infra_failures", metrics.get("infra_failures"))

            for comp in summary.get("comparisons", []):
                if comp["candidate"] == variant_key:
                    _log_metric_safe(
                        client, variant_run_id, f"delta_iou_vs_{comp['control']}", comp.get("delta_iou")
                    )
                    _log_metric_safe(
                        client,
                        variant_run_id,
                        f"delta_f_score_vs_{comp['control']}",
                        comp.get("delta_f_score"),
                    )

            variant_rows = [r for r in rows if r["variant"] == variant_key]
            table_digest = digest(variant_rows)
            v_tags = client.get_run(variant_run_id).data.tags
            if variant_rows and v_tags.get("eval_table_digest") != table_digest:
                table_data = {
                    col: [row[col] for row in variant_rows]
                    for col in (
                        "pair_id",
                        "repeat",
                        "status",
                        "outcome_category",
                        "iou",
                        "f_score",
                        "valid",
                        "seconds",
                        "input_tokens",
                        "output_tokens",
                        "reason",
                    )
                }
                mlflow_mod.log_table(table_data, artifact_file="eval_results.json", run_id=variant_run_id)
                client.set_tag(variant_run_id, "eval_table_digest", table_digest)

            client.set_terminated(
                variant_run_id,
                status="FINISHED" if metrics["measured"] == metrics["expected"] else "RUNNING",
            )

            # Прикрепляем оценки к трассам без дублирования.
            for row in variant_rows:
                if not row.get("attempt"):
                    continue
                trial_dir = study_dir / row["attempt"]
                if not (trial_dir / "record.json").is_file():
                    continue
                try:
                    record = read_record(trial_dir)
                except (OSError, ValueError):
                    continue
                trace_id = record.get("trace_id")
                if not trace_id:
                    continue
                try:
                    trace = client.get_trace(trace_id)
                except MlflowException:
                    continue
                existing_names = {a.name for a in trace.info.assessments}
                if row["iou"] is not None and "iou" not in existing_names:
                    mlflow_mod.log_feedback(
                        trace_id=trace_id,
                        name="iou",
                        value=float(row["iou"]),
                        source=source,
                        rationale=row["reason"] or f"Valid STL ({row['outcome_category']})",
                    )
                if row["f_score"] is not None and "f_score" not in existing_names:
                    mlflow_mod.log_feedback(
                        trace_id=trace_id,
                        name="f_score",
                        value=float(row["f_score"]),
                        source=source,
                    )
                if row["valid"] is not None and "valid" not in existing_names:
                    mlflow_mod.log_feedback(
                        trace_id=trace_id,
                        name="valid",
                        value=bool(row["valid"]),
                        source=source,
                        rationale=row["reason"] or None,
                    )

        backfill_trace_sessions(root)
        normalize_portable_paths(root)
        log("MLFLOW", f"Синхронизировано: серия {study['name']} (Parent {study_run_id[:8]}…).")


def _remove_artifact_dir(artifacts_root: Path, candidate: Path) -> bool:
    if not candidate.exists() or candidate.is_symlink():
        return False
    if not candidate.resolve().is_relative_to(artifacts_root.resolve()):
        raise ValueError(f"Путь артефакта вне mlflow/artifacts: {candidate}")
    shutil.rmtree(candidate)
    return True


def _delete_entity_associations(db: Path, entity_type: str, entity_ids: list[str]) -> None:
    if not entity_ids or not db.is_file():
        return
    with closing(sqlite3.connect(db)) as conn, conn:
        if not conn.execute("PRAGMA table_info(entity_associations)").fetchall():
            return
        for start in range(0, len(entity_ids), 400):
            batch = entity_ids[start : start + 400]
            placeholders = ",".join("?" for _ in batch)
            conn.execute(
                f"DELETE FROM entity_associations "
                f"WHERE (source_type = ? AND source_id IN ({placeholders})) "
                f"   OR (destination_type = ? AND destination_id IN ({placeholders}))",
                [entity_type, *batch, entity_type, *batch],
            )


def purge_study_from_mlflow(root: Path, study_name: str) -> dict[str, int]:
    """Навсегда удаляет runs, traces, оценки и артефакты выбранной серии из MLflow."""
    from mlflow.entities import LifecycleStage, ViewType
    from mlflow.tracking import MlflowClient

    db = database_path(root)
    if not db.is_file():
        return {"runs": 0, "traces": 0, "artifact_dirs": 0}

    escaped = study_name.replace("'", "\\'")
    artifacts_root = root / ARTIFACTS_RELATIVE
    with local_tracking(root):
        client = MlflowClient(tracking_uri=tracking_uri(root), registry_uri=tracking_uri(root))
        exp_ids = [e.experiment_id for e in client.search_experiments(view_type=ViewType.ALL)]

        runs_by_id: dict[str, Any] = {}
        if exp_ids:
            for run in client.search_runs(
                exp_ids,
                filter_string=f"tags.study = '{escaped}'",
                run_view_type=ViewType.ALL,
                max_results=1000,
            ):
                runs_by_id[run.info.run_id] = run
            for parent_id in list(runs_by_id):
                for child in client.search_runs(
                    exp_ids,
                    filter_string=f"tags.`mlflow.parentRunId` = '{parent_id}'",
                    run_view_type=ViewType.ALL,
                    max_results=1000,
                ):
                    runs_by_id.setdefault(child.info.run_id, child)

        traces_by_exp: dict[str, set[str]] = {}
        for exp_id in exp_ids:
            page_token = None
            while True:
                page = client.search_traces(
                    locations=[exp_id],
                    filter_string=f"tags.study = '{escaped}'",
                    max_results=500,
                    page_token=page_token,
                )
                for trace in page:
                    traces_by_exp.setdefault(exp_id, set()).add(trace.info.trace_id)
                page_token = getattr(page, "token", None)
                if not page_token:
                    break

        run_ids = list(runs_by_id)
        if run_ids:
            with closing(sqlite3.connect(db)) as conn:
                if conn.execute("PRAGMA table_info(trace_request_metadata)").fetchall():
                    for start in range(0, len(run_ids), 400):
                        batch = run_ids[start : start + 400]
                        placeholders = ",".join("?" for _ in batch)
                        extra = conn.execute(
                            f"SELECT DISTINCT t.experiment_id, t.request_id "
                            f"FROM trace_request_metadata m "
                            f"JOIN trace_info t ON t.request_id = m.request_id "
                            f"WHERE m.key = 'mlflow.sourceRun' AND m.value IN ({placeholders})",
                            batch,
                        ).fetchall()
                        for eid, tid in extra:
                            traces_by_exp.setdefault(str(eid), set()).add(tid)

        deleted_traces = 0
        deleted_dirs = 0
        all_trace_ids: list[str] = []
        for exp_id, tid_set in traces_by_exp.items():
            tids = sorted(tid_set)
            all_trace_ids.extend(tids)
            for start in range(0, len(tids), 500):
                deleted_traces += client.delete_traces(
                    experiment_id=exp_id, trace_ids=tids[start : start + 500]
                )
            for tid in tids:
                if _remove_artifact_dir(artifacts_root, artifacts_root / "traces" / tid):
                    deleted_dirs += 1

        ordered_runs = sorted(
            runs_by_id.values(),
            key=lambda r: 0 if r.data.tags.get("mlflow.parentRunId") else 1,
        )
        for run in ordered_runs:
            rid = run.info.run_id
            if run.info.lifecycle_stage == LifecycleStage.ACTIVE:
                client.delete_run(rid)
            client._tracking_client.store._hard_delete_run(rid)
            if _remove_artifact_dir(artifacts_root, artifacts_root / rid):
                deleted_dirs += 1

        _delete_entity_associations(db, "trace", all_trace_ids)
        _delete_entity_associations(db, "run", run_ids)
        hide_default_experiment(client)
        normalize_portable_paths(root)

        try:
            with closing(sqlite3.connect(db)) as conn:
                conn.isolation_level = None
                conn.execute("VACUUM")
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError:
            pass

        return {"runs": len(ordered_runs), "traces": deleted_traces, "artifact_dirs": deleted_dirs}
