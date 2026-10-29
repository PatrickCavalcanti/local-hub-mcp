@echo off
rem Inicia o local-hub-mcp (gateway /v1 + MCP na porta da config). Fechar esta janela derruba os modelos.
setlocal
cd /d "%~dp0"
title local-hub-mcp

if not exist config.json (
    echo [erro] falta config.json nesta pasta.
    echo        Copie config.exemplo.json para config.json e edite llama_cpp_dir e models_dir.
    pause
    exit /b 1
)

where uv >nul 2>nul
if errorlevel 1 (
    echo [erro] uv nao encontrado no PATH. Instale: https://docs.astral.sh/uv/
    pause
    exit /b 1
)

uv run local-hub-mcp --config config.json
if errorlevel 1 pause
