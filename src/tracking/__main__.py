"""CLI запуска локального интерфейса MLflow."""

import argparse
import os
import subprocess
import sys

from filelock import FileLock

from src.core.logging import log, log_error
from src.core.utils import ROOT, error_details
from src.tracking.mlflow import (
    database_path,
    hide_default_experiment,
    local_tracking,
    normalize_portable_paths,
    tracking_uri,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Запуск локального переносимого MLflow UI")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    try:
        (ROOT / "mlflow/artifacts").mkdir(parents=True, exist_ok=True)
        lock_path = ROOT / "logs/.locks/mlflow-server.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        normalize_portable_paths(ROOT)
        uri = tracking_uri(ROOT)
        if database_path(ROOT).is_file():
            with local_tracking(ROOT):
                from mlflow.tracking import MlflowClient

                hide_default_experiment(MlflowClient(tracking_uri=uri, registry_uri=uri))
        env = {
            **os.environ,
            "MLFLOW_DISABLE_AGENT_HINT": "1",
            "MLFLOW_DISABLE_TELEMETRY": "true",
        }
        with FileLock(lock_path, timeout=0):
            if not database_path(ROOT).is_file():
                log("MLFLOW", "База tracking.db будет создана при открытии сервера.")
            log(
                "MLFLOW",
                f"Открывай http://{args.host}:{args.port} · Experiments, Runs, Charts, Traces, Prompts.",
            )
            return subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "mlflow",
                    "server",
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                    "--backend-store-uri",
                    uri,
                    "--default-artifact-root",
                    "mlflow/artifacts",
                    "--no-serve-artifacts",
                ],
                cwd=ROOT,
                env=env,
            )
    except Exception as error:  # noqa: BLE001 — диагностируем ошибку запуска сервера.
        log_error(error_details(error, "запуск MLflow UI"), ROOT / "logs/tracking/server-error.log")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
