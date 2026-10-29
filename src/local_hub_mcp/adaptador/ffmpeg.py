"""Localiza e roda o ffmpeg (anexos de áudio/vídeo, STT, TTS e vídeo do llama-server)."""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path

from . import ErroApi

SEM_JANELA = 0x08000000 if os.name == "nt" else 0
_exe: str | None = None


def achar(ffmpeg_dir: Path | None) -> Path | None:
    nome = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    if ffmpeg_dir:
        p = Path(ffmpeg_dir) / nome
        return p if p.is_file() else None
    w = shutil.which("ffmpeg")
    if w:
        return Path(w)
    if os.name == "nt":
        # o winget instala aqui, e terminal aberto antes da instalação não enxerga o PATH novo
        base = os.environ.get("LOCALAPPDATA", "")
        achados = glob.glob(os.path.join(base, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*", "**", "ffmpeg.exe"),
                            recursive=True)
        if achados:
            return Path(achados[0])
    return None


def configurar(caminho: Path | None) -> None:
    global _exe
    _exe = str(caminho) if caminho else None


def disponivel() -> bool:
    return _exe is not None


def rodar(args: list[str], entrada: bytes | None = None) -> subprocess.CompletedProcess:
    if not _exe:
        raise ErroApi(503, "ffmpeg não encontrado: instale (winget install Gyan.FFmpeg) ou defina ffmpeg_dir na config",
                      "server_error")
    return subprocess.run([_exe, *args], input=entrada, capture_output=True, creationflags=SEM_JANELA)
