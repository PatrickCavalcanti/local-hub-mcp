"""Ponto de entrada: python -m local_hub_mcp --config config.json"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import uvicorn

from . import jobobject, recursos
from .adaptador import ffmpeg
from .app import criar_app
from .config import ErroConfig, carregar
from .mcp_server import Contexto
from .perfis import Catalogo
from .supervisor import Supervisor
from .tarefas import Tarefas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="local-hub-mcp", description="Gateway OpenAI + MCP para modelos llama.cpp locais")
    ap.add_argument("--config", default="config.json")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = carregar(Path(a.config))
    except ErroConfig as e:
        print(f"[erro] {e}", file=sys.stderr)
        return 2
    if not cfg.perfis_arquivo.exists():
        exemplo = Path(a.config).resolve().parent / "perfis.exemplo.json"
        if exemplo.exists():
            shutil.copyfile(exemplo, cfg.perfis_arquivo)
            print(f"[info] {cfg.perfis_arquivo.name} criado a partir de {exemplo.name}")
    modo = jobobject.entrar()
    caminho_ffmpeg = ffmpeg.achar(cfg.ffmpeg_dir)
    ffmpeg.configurar(caminho_ffmpeg)
    catalogo = Catalogo(cfg.perfis_arquivo, cfg.models_dir)
    supervisor = Supervisor(cfg, catalogo, ffmpeg_dir=caminho_ffmpeg.parent if caminho_ffmpeg else None)
    ctx = Contexto(cfg, catalogo, supervisor, Tarefas(), caminho_ffmpeg, recursos.versao_llama(cfg.llama_server))
    print("=" * 66)
    print(f"  local-hub-mcp  ·  llama.cpp {ctx.versao_llama}")
    print(f"  /v1 (OpenAI) : {cfg.base_url}/v1")
    print(f"  /mcp         : {cfg.base_url}/mcp")
    print(f"  perfis       : {len(catalogo.listar())} em {cfg.perfis_arquivo}")
    print(f"  ffmpeg       : {caminho_ffmpeg or 'NÃO ENCONTRADO (anexos de áudio/vídeo e voz desligados)'}")
    print(f"  job object   : {modo}  (fechar o serviço derruba os llama-server)")
    print("=" * 66, flush=True)
    uvicorn.run(criar_app(ctx), host=cfg.bind, port=cfg.porta, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
