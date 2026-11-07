"""Собрать публичную производную копию, сохранив локальный экспорт и его хеши.

После live evaluate: переместить его комплект в .hw01-work, затем передать как --source.
Числа, модели, сетки и ответы КОМПАС не меняются; диагностические пути становятся относительными.
"""

import argparse
import shutil
from pathlib import Path

from harness.coursework import ROOT, copy_generation, read, sha, verify_files, write


def publish(source: Path, output: Path, ledger_path: Path) -> None:
    verify_files(source, read(source / "SHA256.json"))
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.name + ".publishing")
    shutil.copytree(source, temporary)
    original_hashes = read(source / "SHA256.json")
    evaluation = read(temporary / "evaluation.json")
    metrics = read(temporary / "expected-metrics.json")
    cases = {case["id"]: case for case in evaluation["cases"]}
    for case in metrics["cases"]:
        for role, fields in case.get("inputs", {}).items():
            for key in ("mesh", "native"):
                if key + "_path" in fields:
                    fields[key + "_path"] = cases[case["id"]][role][key]
    write(temporary / "expected-metrics.json", metrics)
    # История транспорта остаётся в оригинале. Действующий переносимый вход един для всех записей.
    (temporary / "metric-evidence/preparation.json").unlink()
    write(temporary / "metric-evidence/evaluation-input.json", evaluation)
    ledger = read(ledger_path)
    for split, runs in ledger.items():
        for pair_id, relative in runs.items():
            run = ROOT / relative
            destination = temporary / "generation" / pair_id
            if split == "dev":
                copy_generation(run, destination)
            record = read(destination / "generation.json")
            sealed = read(run / "control/bundle.json")
            prefix = run.relative_to(ROOT).as_posix() + "/builder/"
            write(
                destination / "builder-files.json",
                {
                    path.removeprefix(prefix): digest
                    for path, digest in sealed.items()
                    if path.startswith(prefix)
                },
            )
            for original, published in (
                ("AGENTS.md", "builder.md"),
                ("API.md", "api.md"),
                ("CONNECTION.md", "connection.md"),
                ("library/AGENTS.md", "library.md"),
            ):
                path = run / "builder" / original
                if path.exists():
                    verify_files(run / "builder", {original: sealed[prefix + original]})
                    target = destination / "instructions" / published
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
            for attempt in sorted((run / "raw").glob("attempt-*")):
                for name in ("recipe.json", "result.json"):
                    path = attempt / name
                    if not path.is_file():
                        continue
                    expected = record["raw_sha256"][path.relative_to(run).as_posix()]
                    verify_files(attempt, {name: expected})
                    target = destination / "attempts" / attempt.name / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
            selection = record.get("selection")
            if selection:
                attempt = run / "raw" / f"attempt-{selection['attempt']:04d}"
                for name in ("preview.png", "result.json"):
                    path = attempt / name
                    verify_files(attempt, {name: record["raw_sha256"][path.relative_to(run).as_posix()]})
                    shutil.copyfile(
                        path, destination / ("build_evidence.json" if name == "result.json" else name)
                    )
                if split == "dev":
                    verify_files(attempt, {"model.m3d": selection["model_sha256"]})
                    shutil.copyfile(attempt / "model.m3d", destination / "model.m3d")
    write(
        temporary / "publication.json",
        {
            "source_files_sha256": original_hashes,
            "packager_sha256": sha(Path(__file__)),
            "changes": [
                "expected-metrics.json: только mesh_path/native_path заменены относительными путями",
                "metric-evidence/evaluation-input.json: пути приведены к публичному evaluation.json",
                "metric-evidence/preparation.json: локальная диагностика транспорта не публикуется",
                "generation: добавлены исходные рецепты/ответы попыток и отдельные доказательства dev",
                "generation: сохранены точные инструкции и опись изолированной рабочей области",
            ],
            "baseline_changed": False,
            "local_originals_preserved": True,
        },
    )
    for path in temporary.rglob("*"):
        if path.suffix not in {".json", ".md"}:
            continue
        content = path.read_text()
        if str(ROOT) in content or '"ssh_host"' in content or '"remote_repo"' in content:
            raise ValueError(f"Local connection/path found: {path.relative_to(temporary)}")
    write(
        temporary / "SHA256.json",
        {
            path.relative_to(temporary).as_posix(): sha(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file() and path != temporary / "SHA256.json"
        },
    )
    temporary.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    args = parser.parse_args()
    publish(args.source.resolve(), args.output.resolve(), args.ledger.resolve())


if __name__ == "__main__":
    main()
