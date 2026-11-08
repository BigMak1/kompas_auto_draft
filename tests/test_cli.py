"""Команды запуска проверяются на временном наборе без запросов к OpenAI."""

import asyncio
from pathlib import Path

import pytest
from PIL import Image

from src.agent import agent, main
from src.config import ROOT, Config
from src.utils import Run, read_json, sha256, write_json

PARTS = ["KF0011", "KF0017", "KF0027", *[f"EXTRA{i}" for i in range(7)]]


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Настоящий OpenAI API запрещён в тестах")

    monkeypatch.setattr(agent, "AsyncOpenAI", forbidden)


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/baseline.yaml").write_bytes((ROOT / "configs/baseline.yaml").read_bytes())
    (tmp_path / "data/input").mkdir(parents=True)
    items = []
    for index, part in enumerate(PARTS):
        source = tmp_path / f"data/input/drawing_{index}.png"
        Image.new("RGB", (8, 8), "white").save(source)
        items.append(
            {
                "pair_id": part,
                "input_png": {"path": f"input/{source.name}", "sha256": sha256(source)},
            }
        )
    write_json(tmp_path / "data/manifest.json", {"items": items})
    # Посторонний файл не должен незаметно расширять набор.
    Image.new("RGB", (8, 8), "white").save(tmp_path / "data/input/extra.png")
    monkeypatch.setattr(main, "ROOT", tmp_path)
    return tmp_path


def test_no_command_shows_help_without_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main.main([]) == 0
    output = capsys.readouterr()
    assert "uv run poe help" in output.out and "run" in output.out
    assert not output.err and not list(tmp_path.iterdir())


@pytest.mark.parametrize("failed", [False, True])
def test_run_without_drawing_processes_every_case_once(
    dataset: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, failed: bool
) -> None:
    calls = []
    running = False

    async def generate(source: Path, run: Run, config: Config) -> dict:
        nonlocal running
        assert not running
        running = True
        await asyncio.sleep(0)
        calls.append((source, run))
        assert run.directory == dataset / "run/baseline" / run.pair_id
        running = False
        return {"status": "failed" if failed and run.pair_id == "KF0017" else "success"}

    monkeypatch.setattr(agent, "generate", generate)
    assert main.main(["run"]) == int(failed)
    assert [run.pair_id for _, run in calls] == PARTS
    assert [source.name for source, _ in calls] == [f"drawing_{i}.png" for i in range(10)]
    assert f"Построено {9 if failed else 10}/10" in capsys.readouterr().out


@pytest.mark.parametrize("custom_png", [False, True])
def test_run_single_drawing(dataset: Path, monkeypatch: pytest.MonkeyPatch, custom_png: bool) -> None:
    calls = []
    source = dataset / "data/input/drawing_1.png"

    async def generate(source: Path, run: Run, config: Config) -> dict:
        calls.append((source, run))
        return {"status": "success"}

    monkeypatch.setattr(agent, "generate", generate)
    assert main.main(["run", str(source) if custom_png else "kf0017"]) == 0
    assert len(calls) == 1 and calls[0][0] == source
    assert calls[0][1].pair_id == ("DRAWING_1" if custom_png else "KF0017")


def test_changed_input_stops_batch_before_first_generation(
    dataset: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    async def forbidden(*args: object) -> dict:
        raise AssertionError("Прогон не должен начинаться до проверки входов")

    monkeypatch.setattr(agent, "generate", forbidden)
    (dataset / "data/input/drawing_1.png").write_bytes(b"changed")
    assert main.main(["run"]) == 1
    assert "SHA256 не совпал" in " ".join(capsys.readouterr().out.split())
    diagnostic = (dataset / "artifacts/command-error.log").read_text()
    assert "проверка входных файлов" in diagnostic and "SHA256 не совпал" in diagnostic
    assert not (dataset / "run").exists()


@pytest.mark.parametrize("invalid", ["empty", "duplicate", "unknown"])
def test_invalid_selection_does_not_start_generation(
    dataset: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    async def forbidden(*args: object) -> dict:
        raise AssertionError("Некорректный набор не должен запускаться")

    monkeypatch.setattr(agent, "generate", forbidden)
    manifest = read_json(dataset / "data/manifest.json")
    if invalid == "empty":
        manifest["items"] = []
    elif invalid == "duplicate":
        manifest["items"].append(manifest["items"][0])
    write_json(dataset / "data/manifest.json", manifest)
    assert main.main(["run", "UNKNOWN"] if invalid == "unknown" else ["run"]) == 1
    assert not (dataset / "run").exists()


def test_batch_without_key_preserves_previous_models(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    previous = []
    for part in PARTS:
        run = Run(part, root=dataset)
        run.create()
        model = run.directory / "model.stl"
        model.write_bytes(b"previous")
        previous.append(model)
    assert main.main(["run"]) == 1
    assert all(model.read_bytes() == b"previous" for model in previous)
    assert "OPENAI_API_KEY" in (dataset / "artifacts/command-error.log").read_text()
