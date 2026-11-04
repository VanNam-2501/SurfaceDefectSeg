$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:STRAW_WEB_CONFIG = 'straw_realtime_web.json'
& '.\.venv\Scripts\python.exe' -m uvicorn apps.web_demo.backend.b1_app:app --host 127.0.0.1 --port 8034 --workers 1
