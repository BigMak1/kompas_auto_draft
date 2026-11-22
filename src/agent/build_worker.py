"""Изолированный процесс CadQuery для принудительного ограничения времени построения."""

import sys
from pathlib import Path

from src.agent.cad import build
from src.agent.schemas.recipe import Recipe
from src.config import CadSettings
from src.utils import error_details, read_json, write_json


def main() -> int:
    directory = Path(sys.argv[1])
    request = read_json(directory / "build-input.json")
    try:
        result = build(
            Recipe.model_validate(request["recipe"]), directory, CadSettings(**request["settings"])
        )
        write_json(directory / "build-result.json", {"status": "success", "cad": result})
        return 0
    except Exception as error:  # noqa: BLE001 — передаём исходную ошибку родителю.
        write_json(
            directory / "build-result.json",
            {
                "status": "failed",
                "failure_kind": "generation"
                if isinstance(error, (ValueError, RuntimeError))
                else "infrastructure",
                "error": error_details(error, "построение CAD"),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
