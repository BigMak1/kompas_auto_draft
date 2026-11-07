"""Запуск доверенного оценщика на Windows через существующий SSH-профиль."""

import argparse
import base64
import json
import subprocess
from pathlib import Path


def run(settings: dict, script: str, arguments: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    def quote(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    command = (
        "$ProgressPreference='SilentlyContinue'; & "
        + " ".join(
            quote(s) for s in [settings["windows_python"], "-I", "-B", "-X", "utf8", script, *arguments]
        )
        + "; exit $LASTEXITCODE"
    )
    encoded = base64.b64encode(command.encode("utf-16-le")).decode()
    return subprocess.run(
        [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            settings["ssh_host"],
            "powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand " + encoded,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", type=Path)
    parser.add_argument("script")
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    result = run(json.loads(args.settings.read_text()), args.script, args.arguments)
    print(result.stdout)
    print(result.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
