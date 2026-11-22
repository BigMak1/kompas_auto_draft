"""Изолированный процесс CadQuery с жёстким таймаутом и очисткой временных файлов."""

import asyncio
import os
import sys
from pathlib import Path

from OCP.Standard import Standard_Failure

from src.cad.builder import build
from src.core.config import CadSettings
from src.core.utils import ROOT, atomic, error_details, read_json, redact, write_json
from src.schemas.recipe import Recipe


async def build_recipe(
    recipe: dict,
    destination: Path,
    technical: Path,
    settings: CadSettings,
    timeout: float,
) -> dict:
    """Запускает CadQuery в отдельном процессе, чтобы при таймауте можно было его прервать."""
    technical.mkdir(parents=True, exist_ok=True)
    write_json(technical / "build-input.json", {"recipe": recipe, "settings": settings.model_dump()})
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))
    }
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "src.cad.worker",
        str(technical.resolve()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=environment,
        cwd=ROOT,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.communicate()
        raise
    try:
        atomic(technical / "cad.log", redact((stdout + stderr).decode(errors="replace")).encode("utf-8"))
        result_path = technical / "build-result.json"
        if not result_path.is_file():
            raise RuntimeError(f"Процесс CAD завершился без результата, код {process.returncode}")
        result = read_json(result_path)
        if result["status"] == "success":
            atomic(destination / "model.stl", (technical / "model.stl").read_bytes())
        return result
    finally:
        for name in ("model.stl", "build-input.json", "build-result.json"):
            (technical / name).unlink(missing_ok=True)


def main() -> int:
    directory = Path(sys.argv[1])
    request = read_json(directory / "build-input.json")
    try:
        result = build(
            Recipe.model_validate(request["recipe"]),
            directory,
            CadSettings(**request["settings"]),
        )
        write_json(directory / "build-result.json", {"status": "success", "cad": result})
        return 0
    except Exception as error:  # noqa: BLE001 — передаём исходную ошибку родительскому процессу.
        write_json(
            directory / "build-result.json",
            {
                "status": "failed",
                "failure_kind": "generation"
                if isinstance(error, (ValueError, RuntimeError, Standard_Failure))
                else "infrastructure",
                "error": error_details(error, "построение CAD"),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
