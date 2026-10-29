"""Configuração do serviço: um arquivo JSON, sem caminho fixo no código."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

LOCAIS = {"127.0.0.1", "localhost", "::1"}


class ErroConfig(Exception):
    pass


def _exe(nome: str) -> str:
    return nome + (".exe" if os.name == "nt" else "")


@dataclass(frozen=True)
class Config:
    llama_cpp_dir: Path
    models_dir: Path
    perfis_arquivo: Path
    logs_dir: Path
    ffmpeg_dir: Path | None = None
    bind: str = "127.0.0.1"
    porta: int = 8079
    portas_internas: tuple[int, int] = (8180, 8199)
    chaves: dict[str, str] = field(default_factory=dict)
    load_timeout_s: int = 600
    espera_s: int = 90
    subir_no_boot: list[str] = field(default_factory=list)

    @property
    def llama_server(self) -> Path:
        return self.llama_cpp_dir / _exe("llama-server")

    @property
    def llama_bench(self) -> Path:
        return self.llama_cpp_dir / _exe("llama-bench")

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.bind in ("0.0.0.0", "::") else self.bind
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.porta}"


def _caminho(base: Path, valor: str | None) -> Path | None:
    if not valor:
        return None
    p = Path(os.path.expandvars(os.path.expanduser(str(valor))))
    return (p if p.is_absolute() else base / p).resolve()


def carregar(caminho: Path) -> Config:
    caminho = Path(caminho).resolve()
    if not caminho.is_file():
        raise ErroConfig(f"config não encontrada: {caminho} (copie config.exemplo.json para config.json e edite)")
    try:
        bruto = json.loads(caminho.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ErroConfig(f"{caminho}: JSON inválido ({e})") from e
    for chave in ("llama_cpp_dir", "models_dir"):
        if not bruto.get(chave):
            raise ErroConfig(f"{caminho}: falta '{chave}'")
    base = caminho.parent
    faixa = bruto.get("portas_internas") or [8180, 8199]
    cfg = Config(
        llama_cpp_dir=_caminho(base, bruto["llama_cpp_dir"]),
        models_dir=_caminho(base, bruto["models_dir"]),
        perfis_arquivo=_caminho(base, bruto.get("perfis") or "perfis.json"),
        logs_dir=_caminho(base, bruto.get("logs_dir") or "logs"),
        ffmpeg_dir=_caminho(base, bruto.get("ffmpeg_dir")),
        bind=str(bruto.get("bind") or "127.0.0.1"),
        porta=int(bruto["porta"]) if bruto.get("porta") is not None else 8079,
        portas_internas=(int(faixa[0]), int(faixa[1])),
        chaves={k: str(v) for k, v in (bruto.get("chaves") or {}).items() if v},
        load_timeout_s=int(bruto.get("load_timeout_s") or 600),
        espera_s=int(bruto.get("espera_s") or 90),
        subir_no_boot=bruto.get("subir_no_boot", []),
    )
    validar(cfg)
    return cfg


def validar(cfg: Config) -> None:
    if not isinstance(cfg.subir_no_boot, list) or not all(isinstance(a, str) for a in cfg.subir_no_boot):
        raise ErroConfig("subir_no_boot deve ser uma lista de strings")
    if not cfg.llama_server.is_file():
        raise ErroConfig(f"llama-server não encontrado em {cfg.llama_cpp_dir} (confira llama_cpp_dir)")
    if not cfg.models_dir.is_dir():
        raise ErroConfig(f"models_dir não existe: {cfg.models_dir}")
    if not 1 <= cfg.porta <= 65535:
        raise ErroConfig(f"porta inválida: {cfg.porta}; use 1 a 65535")
    a, b = cfg.portas_internas
    if not (1024 <= a <= b <= 65535):
        raise ErroConfig(f"portas_internas inválida: {a}-{b}")
    if a <= cfg.porta <= b:
        raise ErroConfig(f"a porta do gateway ({cfg.porta}) está dentro de portas_internas ({a}-{b})")
    if cfg.bind not in LOCAIS and not (cfg.chaves.get("admin") and cfg.chaves.get("inference")):
        raise ErroConfig(f"bind '{cfg.bind}' expõe o serviço na rede: defina chaves.admin e chaves.inference")
    if cfg.ffmpeg_dir is not None and not cfg.ffmpeg_dir.is_dir():
        raise ErroConfig(f"ffmpeg_dir não existe: {cfg.ffmpeg_dir}")
