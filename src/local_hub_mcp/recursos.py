"""Leitura de recursos da máquina: VRAM (total, livre, por processo), RAM, versão do llama.cpp."""
from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import psutil

GB = 1024 ** 3
SEM_JANELA = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW
PID_RE = re.compile(r"pid_(\d+)_luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)", re.I)


@dataclass
class Gpu:
    total_gb: float
    livre_gb: float


@dataclass
class EstadoRecursos:
    gpu: Gpu | None
    ram_total_gb: float
    ram_livre_gb: float


@dataclass
class MemoriaProcesso:
    dedicada_gb: float = 0.0
    compartilhada_gb: float = 0.0


def _sistema(exe: str) -> str:
    """Caminho em System32 (o Git Bash põe outros find/timeout na frente no PATH)."""
    return os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", exe)


def _rodar(cmd: list[str], timeout: float = 15) -> str:
    r = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=SEM_JANELA)
    return (r.stdout or b"").decode("utf-8", "replace") + (r.stderr or b"").decode("utf-8", "replace")


def parse_nvidia_smi(texto: str) -> Gpu | None:
    for linha in texto.splitlines():
        campos = [x.strip() for x in linha.split(",")]
        if len(campos) == 2 and all(re.fullmatch(r"\d+(\.\d+)?", x) for x in campos):
            total, livre = (float(x) for x in campos)
            return Gpu(total_gb=total / 1024, livre_gb=livre / 1024)
    return None


def ler_gpu() -> Gpu | None:
    exe = shutil.which("nvidia-smi") or (_sistema("nvidia-smi.exe") if os.name == "nt" else None)
    if not exe or not os.path.exists(exe):
        return None
    try:
        return parse_nvidia_smi(_rodar([exe, "--query-gpu=memory.total,memory.free",
                                        "--format=csv,noheader,nounits"]))
    except (OSError, subprocess.TimeoutExpired):
        return None


def ler_estado() -> EstadoRecursos:
    vm = psutil.virtual_memory()
    return EstadoRecursos(gpu=ler_gpu(), ram_total_gb=vm.total / GB, ram_livre_gb=vm.available / GB)


def parse_typeperf(texto: str) -> dict[int, MemoriaProcesso]:
    """CSV do typeperf (cabeçalho + uma amostra) -> memória de GPU por PID, selecionando o adaptador com mais dedicada."""
    linhas = [l for l in texto.splitlines() if l.startswith('"')]
    if len(linhas) < 2:
        return {}
    cab, val = list(csv.reader(io.StringIO("\n".join(linhas[:2]))))[:2]
    adaptadores: dict[tuple[int, str], MemoriaProcesso] = {}
    for nome, valor in zip(cab[1:], val[1:]):
        m = PID_RE.search(nome)
        if not m:
            continue
        try:
            gb = float(valor.strip() or 0) / GB
        except ValueError:
            continue
        mp = adaptadores.setdefault((int(m.group(1)), m.group(2).lower()), MemoriaProcesso())
        baixo = nome.lower()
        if baixo.endswith("\\dedicated usage"):
            mp.dedicada_gb += gb
        elif baixo.endswith("\\shared usage"):
            mp.compartilhada_gb += gb
    out: dict[int, MemoriaProcesso] = {}
    for (pid, _), mp in adaptadores.items():
        if pid not in out or (mp.dedicada_gb, mp.compartilhada_gb) > (out[pid].dedicada_gb, out[pid].compartilhada_gb):
            out[pid] = mp
    return out


def memoria_gpu_processos() -> dict[int, MemoriaProcesso]:
    """Contadores do Windows. O nvidia-smi não mostra o transbordo para a RAM (WDDM); estes mostram."""
    if os.name != "nt":
        return {}
    try:
        texto = _rodar([_sistema("typeperf.exe"), r"\GPU Process Memory(*)\Dedicated Usage",
                        r"\GPU Process Memory(*)\Shared Usage", "-sc", "1"], timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    return parse_typeperf(texto)


def ram_processo_gb(pid: int) -> float:
    try:
        return psutil.Process(pid).memory_info().rss / GB
    except psutil.Error:
        return 0.0


def parse_versao(texto: str) -> str:
    m = re.search(r"version:\s*(.+)", texto)
    return m.group(1).strip() if m else "desconhecida"


def versao_llama(llama_server: Path) -> str:
    try:
        return parse_versao(_rodar([str(llama_server), "--version"], timeout=30))
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"desconhecida ({e})"
