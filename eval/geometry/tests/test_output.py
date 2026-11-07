"""Читаемые имена не перезаписывают результаты; прогресс не загрязняет stdout."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from eval.geometry.io import create_artifact_directory, read_json
from eval.geometry.progress import progress_display
from eval.geometry.smoke import create_smoke
from eval.geometry.terminal import duration, print_summary


@pytest.mark.parametrize("prefix", ["eval", "pipeline", "mesh-probe", "import", "launch", "export"])
def test_same_second_outputs_keep_local_time_and_do_not_overwrite(tmp_path: Path, prefix: str) -> None:
    started = datetime(2026, 11, 7, 17, 52, 30, 123456, tzinfo=timezone(timedelta(hours=3)))
    first = create_artifact_directory(tmp_path, prefix, started)
    marker = first / "metrics.json"
    marker.write_text("original")
    with ThreadPoolExecutor(max_workers=4) as pool:
        others = list(pool.map(lambda _: create_artifact_directory(tmp_path, prefix, started), range(4)))
    assert first.name == f"{prefix}_2026-11-07_17-52-30"
    assert {path.name for path in others} == {f"{first.name}_{i}" for i in range(2, 6)}
    assert marker.read_text() == "original"


def test_redirected_progress_uses_stderr_without_ansi(capsys: pytest.CaptureFixture[str]) -> None:
    with progress_display(no_color=True) as progress:
        progress("KF0011 · Совмещение: положение 2/8")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[00:00]" in captured.err
    assert "положение 2/8" in captured.err
    assert "\x1b" not in captured.err


def test_quiet_progress_is_silent(capsys: pytest.CaptureFixture[str]) -> None:
    with progress_display(quiet=True) as progress:
        progress("Не печатать")
    assert capsys.readouterr() == ("", "")


def test_total_duration_printed_without_changing_quiet_mode(tmp_path: Path, capsys) -> None:
    output = create_smoke(tmp_path / "smoke")
    report = read_json(output / "metrics.json")
    print_summary(report, output, no_color=True, elapsed_seconds=218.9)
    assert "Общее время прогона: 3 мин 38 с" in capsys.readouterr().out
    print_summary(report, output, quiet=True, elapsed_seconds=218.9)
    assert capsys.readouterr().out == f"{output}\n"
    assert duration(3661) == "1 ч 01 мин 01 с"
