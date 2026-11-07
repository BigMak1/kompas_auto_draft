#!/bin/bash
# Mac client: carry MCP JSON-RPC over SSH, with no pseudo-terminal or stdout logs.
set -euo pipefail

remote_script=$(cat <<'POWERSHELL'
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
& 'C:\MCP\ksapi-expert-mcp\python\python.exe' -I -B -X utf8 'C:\Mac\Home\Documents\VS_code_projects\ITMO\ASCON\kompas_auto_draft\scripts\ksapi_mcp_stdio.py'
exit $LASTEXITCODE
POWERSHELL
)
encoded_command=$(printf '%s' "$remote_script" | /usr/bin/iconv -f UTF-8 -t UTF-16LE | /usr/bin/base64 | tr -d '\n')

exec /usr/bin/ssh -T \
    -o BatchMode=yes \
    -o ConnectTimeout=10 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    windows_server \
    "powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand $encoded_command"
