#!/usr/bin/env sh
# Inicia o local-hub-mcp (gateway /v1 + MCP na porta da config). Ctrl+C derruba os modelos.
cd "$(dirname "$0")" || exit 1

if [ ! -f config.json ]; then
    echo "[erro] falta config.json nesta pasta."
    echo "       Copie config.exemplo.json para config.json e edite llama_cpp_dir e models_dir."
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "[erro] uv nao encontrado no PATH. Instale: https://docs.astral.sh/uv/"
    exit 1
fi

exec uv run local-hub-mcp --config config.json
