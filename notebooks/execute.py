"""Выполнить EDA в свежем ядре и атомарно сохранить ноутбук с результатами."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="По умолчанию обновляет hw01_eda.ipynb")
    args = parser.parse_args()
    source = Path(__file__).resolve().with_name("hw01_eda.ipynb")
    target = args.output.resolve() if args.output else source
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    with TemporaryDirectory(prefix="hw01-eda-") as cache:
        env = os.environ | {
            "MPLCONFIGDIR": str(Path(cache) / "matplotlib"),
            "IPYTHONDIR": str(Path(cache) / "ipython"),
            "JUPYTER_RUNTIME_DIR": str(Path(cache) / "jupyter"),
        }
        NotebookClient(
            notebook,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(source.parent)}},
            record_timing=False,
        ).execute(env=env)
    nbformat.validate(notebook)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(target)
    count = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Executed {count} code cells: {target.name}")


if __name__ == "__main__":
    main()
