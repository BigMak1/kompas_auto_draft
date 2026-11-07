"""Run the portable Windows ksapi-expert MCP over stdio (for SSH clients).

Use the Python bundled with C:\\MCP\\ksapi-expert-mcp. The portable distribution
provides the model, index, dependencies and its native stdio launch command.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"C:\MCP\ksapi-expert-mcp"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / "config" / "server.json").read_text(encoding="utf-8"))
    wrappers = config["ksapi_wrappers_dir"]
    wrappers_dir = root / wrappers if wrappers is not None else root / "data" / "ksapi-wrappers"

    # The bundled Python loads the portable libs through its site-packages .pth.
    from ksapi_expert_mcp.deployment.cli import run

    return run(
        argv=(
            "launch",
            "--corpus",
            "ksapi",
            "--artifact-root",
            str(root / "data" / "snapshots"),
            "--embedding-cache-dir",
            str(root / "data" / "models"),
            "--runtime-work-root",
            str(root / "work"),
            "--query-log-file",
            str(root / "logs" / "queries.jsonl"),
            "--ksapi-wrappers-dir",
            str(wrappers_dir),
            "--wrapper-max-matches",
            str(config["wrapper_max_matches"]),
            "--validate-max-code-chars",
            str(config["validate_max_code_chars"]),
            "--validate-max-concurrent",
            str(config["validate_max_concurrent"]),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
