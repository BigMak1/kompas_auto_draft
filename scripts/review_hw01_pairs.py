"""Выгрузить проверочные M3D на копиях для аудита пар до генерации.

uv run --group eval python -m scripts.review_hw01_pairs --settings <локальный JSON>
Вывод — новая папка runs/hw01-qa; исходные файлы не меняются.
"""

import argparse
from pathlib import Path

from eval.geometry.backend import BackendSettings, WorkerClient
from eval.geometry.io import atomic_json, create_artifact_directory, read_json
from eval.geometry.models import ExportConfig
from eval.geometry.normalize import normalize_export
from harness.coursework import cad_session

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    settings = read_json(args.settings)
    backend = BackendSettings(
        transport="ssh",
        ssh_host=settings["ssh_host"],
        remote_repo=settings["windows_repo"],
        python=settings["windows_python"],
        bin_dir=settings["kompas_bin_dir"],
    )
    output = create_artifact_directory(ROOT / "runs/hw01-qa", "review")
    client = WorkerClient(backend, output)
    print(client.health(), flush=True)
    inventory = read_json(ROOT / "data/hw01/v1/manifest.json")
    evidence = {}
    for item in inventory["items"]:
        if item["split"] != "val":
            continue
        print("Export", item["pair_id"], flush=True)
        job, _ = client.invoke(
            "export",
            {"profile": ExportConfig().model_dump()},
            {"model": ROOT / "data/hw01/v1" / item["reference_m3d"]["path"]},
        )
        normalize_export(job)
        evidence[item["pair_id"]] = str(job.relative_to(ROOT))
        atomic_json(output / "index.json", evidence)
    print(output, flush=True)


if __name__ == "__main__":
    with cad_session():
        main()
