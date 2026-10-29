"""Mede velocidade com o llama-bench variando -ncmoe e -ub; devolve a tabela e o melhor.

Medido: -ncmoe e -ub são as flags que mais importam e o ótimo muda por modelo (Qwen3.6: -ncmoe 32;
MiniCPM 35B: -ncmoe 40; -ub 2048 dá +88% de prefill com experts na CPU, mas num 2B todo na GPU só
gasta 1,2 GB de buffer).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .perfis import Perfil, caminho_seguro

SEM_JANELA = 0x08000000 if os.name == "nt" else 0
# flag do llama-server -> flag equivalente do llama-bench
TRANSFERIVEIS = {"-ngl": "-ngl", "--n-gpu-layers": "-ngl", "-ncmoe": "-ncmoe", "--n-cpu-moe": "-ncmoe",
                 "-ub": "-ub", "--ubatch-size": "-ub", "-b": "-b", "--batch-size": "-b",
                 "-ctk": "-ctk", "--cache-type-k": "-ctk", "-ctv": "-ctv", "--cache-type-v": "-ctv",
                 "-fa": "-fa", "--flash-attn": "-fa", "-t": "-t", "--threads": "-t",
                 "--device": "-dev", "-dev": "-dev"}


class ErroMedir(Exception):
    pass


def flags_bench(flags: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    i = 0
    while i < len(flags):
        if flags[i] in TRANSFERIVEIS and i + 1 < len(flags):
            out[TRANSFERIVEIS[flags[i]]] = flags[i + 1]
            i += 2
        else:
            i += 1
    return out


def montar_argv(llama_bench: Path, models_dir: Path, perfil: Perfil, ncmoe: list[int] | None = None,
                ub: list[int] | None = None, n_prompt: int = 512, n_gen: int = 64, repeticoes: int = 2) -> list[str]:
    base = flags_bench(perfil.flags)
    if perfil.onde == "cpu":
        base["-dev"] = "none"
    if ncmoe:
        base["-ncmoe"] = ",".join(str(x) for x in ncmoe)
    if ub:
        base["-ub"] = ",".join(str(x) for x in ub)
    cmd = [str(llama_bench), "-m", str(caminho_seguro(models_dir, perfil.arquivo)), "-p", str(n_prompt),
           "-n", str(n_gen), "-r", str(repeticoes), "-o", "json"]
    for k, v in base.items():
        cmd += [k, v]
    return cmd


def interpretar(texto: str) -> list[dict]:
    ini, fim = texto.find("["), texto.rfind("]")
    if ini < 0 or fim < ini:
        raise ErroMedir("a saída do llama-bench não tem JSON")
    linhas: dict[tuple, dict] = {}
    for r in json.loads(texto[ini:fim + 1]):
        chave = (r.get("n_cpu_moe", 0), r.get("n_ubatch"))
        l = linhas.setdefault(chave, {"ncmoe": chave[0], "ub": chave[1], "pp_ts": None, "tg_ts": None})
        if r.get("n_gen", 0) == 0 and r.get("n_prompt", 0) > 0:
            l["pp_ts"] = round(r["avg_ts"], 1)
        elif r.get("n_prompt", 0) == 0 and r.get("n_gen", 0) > 0:
            l["tg_ts"] = round(r["avg_ts"], 1)
    return sorted(linhas.values(), key=lambda l: (l["ncmoe"], l["ub"] or 0))


def melhor(linhas: list[dict]) -> dict:
    candidatas = [l for l in linhas if l["tg_ts"]]
    if not candidatas:
        raise ErroMedir("nenhuma combinação gerou tokens (todas falharam?)")
    return max(candidatas, key=lambda l: (l["tg_ts"], l["pp_ts"] or 0))


def aplicar(flags: list[str], escolhido: dict, ncmoe_variado: bool, ub_variado: bool) -> list[str]:
    """Troca (ou acrescenta) -ncmoe/-ub nas flags do perfil pelo valor escolhido."""
    novo = list(flags)
    for nome, sinonimos, valor, variado in (("-ncmoe", ("-ncmoe", "--n-cpu-moe"), escolhido["ncmoe"], ncmoe_variado),
                                            ("-ub", ("-ub", "--ubatch-size"), escolhido["ub"], ub_variado)):
        if not variado:
            continue
        for i, f in enumerate(novo):
            if f in sinonimos and i + 1 < len(novo):
                novo[i + 1] = str(valor)
                break
        else:
            novo += [nome, str(valor)]
    return novo


def rodar(cmd: list[str], timeout_s: int = 900) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout_s, creationflags=SEM_JANELA)
    except subprocess.TimeoutExpired as e:
        raise ErroMedir(f"o llama-bench passou de {timeout_s} s; use uma grade menor") from e
    saida = (r.stdout or b"").decode("utf-8", "replace")
    if r.returncode != 0:
        raise ErroMedir("o llama-bench falhou: " + (r.stderr or r.stdout or b"").decode("utf-8", "replace")[-400:])
    return saida
