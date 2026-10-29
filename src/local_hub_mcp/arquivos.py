"""Apagar arquivos baixados de models_dir, em duas etapas: listar (com um código) e confirmar com o código.

O código amarra a confirmação à lista exata que foi mostrada: se algo mudou entre as duas chamadas, recusa.
Nunca apaga arquivo usado por um perfil nem nada fora de models_dir.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .perfis import ErroPerfil, Perfil, caminho_seguro, partes

GB = 1024 ** 3


def _em_uso(models_dir: Path, perfis: list[Perfil]) -> dict[Path, str]:
    usados: dict[Path, str] = {}
    for p in perfis:
        try:
            for a in partes(caminho_seguro(models_dir, p.arquivo)):
                usados[a.resolve()] = p.alias
            if p.mmproj:
                usados[caminho_seguro(models_dir, p.mmproj).resolve()] = p.alias
        except ErroPerfil:
            continue
    return usados


def _alvos(models_dir: Path, caminhos: list[str]) -> tuple[list[Path], list[Path]]:
    """(arquivos, pastas) a apagar. Uma pasta leva tudo dentro dela; um GGUF dividido leva todas as partes."""
    base = Path(models_dir).resolve()
    arquivos: list[Path] = []
    pastas: list[Path] = []
    if not caminhos:
        raise ErroPerfil("diga o que apagar: caminhos relativos a models_dir (arquivo .gguf ou pasta)")
    for rel in caminhos:
        alvo = caminho_seguro(models_dir, rel)
        if alvo == base:
            raise ErroPerfil("aponte um arquivo ou uma pasta dentro de models_dir, não a pasta de modelos inteira")
        if alvo.is_dir():
            pastas.append(alvo)
            arquivos += [f for f in alvo.rglob("*") if f.is_file()]
        elif alvo.is_file():
            arquivos += [a for a in partes(alvo) if a.is_file()]
        else:
            raise ErroPerfil(f"não existe em models_dir: {rel}")
    vistos: dict[Path, None] = {}
    for f in arquivos:
        real = f.resolve()
        if base not in real.parents:   # link/junction dentro da pasta apontando para fora
            raise ErroPerfil(f"'{f}' aponta para fora de models_dir; não apago por aqui")
        vistos.setdefault(real, None)
    return list(vistos), pastas


def _codigo(base: Path, arquivos: list[Path]) -> str:
    assinatura = "\n".join(f"{f.relative_to(base).as_posix()}:{f.stat().st_size}" for f in sorted(arquivos))
    return hashlib.sha256(assinatura.encode()).hexdigest()[:10]


def _conferir_uso(models_dir: Path, arquivos: list[Path], perfis: list[Perfil],
                  em_uso: dict[Path, str] | None) -> None:
    usados = {**_em_uso(models_dir, perfis), **{Path(k).resolve(): v for k, v in (em_uso or {}).items()}}
    presos = sorted({usados[f] for f in arquivos if f in usados})
    if presos:
        raise ErroPerfil(f"há arquivos usados pelo(s) perfil(is) {', '.join(presos)} (no catálogo ou no ar); "
                         f"derrube/apague o perfil antes ou escolha outros arquivos")


def planejar(models_dir: Path, caminhos: list[str], perfis: list[Perfil],
             em_uso: dict[Path, str] | None = None) -> dict:
    """em_uso: arquivos de processos no ar (podem diferir do catálogo depois de um editar_perfil)."""
    base = Path(models_dir).resolve()
    arquivos, _ = _alvos(models_dir, caminhos)
    _conferir_uso(models_dir, arquivos, perfis, em_uso)
    total = sum(f.stat().st_size for f in arquivos)
    return {"arquivos": [{"caminho": f.relative_to(base).as_posix(), "tamanho_gb": round(f.stat().st_size / GB, 3)}
                         for f in sorted(arquivos)],
            "total_gb": round(total / GB, 2), "confirmacao": _codigo(base, arquivos)}


def _remover_pastas_vazias(pasta: Path) -> None:
    """Só pastas que ficaram vazias, de baixo para cima (nunca apaga arquivo que não estava na lista)."""
    for raiz, _, _ in sorted(os.walk(pasta), key=lambda x: -len(x[0])):
        try:
            os.rmdir(raiz)
        except OSError:
            pass


def remover(models_dir: Path, caminhos: list[str], perfis: list[Perfil], confirmacao: str,
            em_uso: dict[Path, str] | None = None) -> dict:
    base = Path(models_dir).resolve()
    arquivos, pastas = _alvos(models_dir, caminhos)   # uma única enumeração: é ESTA lista que se confere e apaga
    _conferir_uso(models_dir, arquivos, perfis, em_uso)
    if confirmacao != _codigo(base, arquivos):
        raise ErroPerfil("o código de confirmação não bate com a lista atual de arquivos; chame de novo sem "
                         "confirmacao, mostre a lista ao usuário e confirme com o código novo")
    apagados, falhas, liberado = [], [], 0
    for f in sorted(arquivos):
        try:
            tamanho = f.stat().st_size
            f.unlink()
            apagados.append(f.relative_to(base).as_posix())
            liberado += tamanho
        except OSError as e:
            falhas.append({"caminho": f.relative_to(base).as_posix(), "erro": str(e)})
    for pasta in pastas:
        _remover_pastas_vazias(pasta)
    return {"apagados": apagados, "liberado_gb": round(liberado / GB, 2), "falhas": falhas}
