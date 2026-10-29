"""Perfis: a receita de cada modelo (arquivo, flags, capacidades), guardada em perfis.json."""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

CAPACIDADES = frozenset({"llm.chat", "llm.vision", "llm.fast", "embedding", "stt", "tts",
                         "text.generate", "audio.in", "video.in"})
ONDE = ("gpu", "gpu+cpu", "cpu")
# o serviço controla estas (porta, host, arquivos, alias, ffmpeg do vídeo)
FLAGS_DO_SERVICO = frozenset({"--port", "--host", "-m", "--model", "--mmproj", "-mm", "--alias", "-a",
                              "--video-ffmpeg-dir"})
# medido: std::bad_alloc perto do limite de RAM; mmap (o padrão) funciona
FLAGS_PROIBIDAS = frozenset({"--no-mmap", "--mlock"})
ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
# nome coringa do gateway: "o modelo que está no ar agora" (não pode ser alias de perfil)
ALIAS_CORINGA = "atual"
PARTE_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


class ErroPerfil(Exception):
    pass


@dataclass
class Voz:
    vozes: dict[str, dict] = field(default_factory=lambda: {"female": {}, "male": {}})
    idioma: str = "pt"
    pedaco_tts: int = 250
    pedaco_stt_s: float = 45.0
    numeros_por_extenso: bool = True   # TTS em pt: troca números por palavras antes de mandar ao modelo


@dataclass
class Medido:
    vram_gb: float
    ram_gb: float
    carga_s: float
    em: str


@dataclass
class Perfil:
    alias: str
    arquivo: str
    onde: str
    flags: list[str] = field(default_factory=list)
    capacidades: list[str] = field(default_factory=lambda: ["llm.chat"])
    mmproj: str | None = None
    anexos: bool = False
    video: bool = False
    voz: Voz | None = None
    notas: str = ""
    medido: Medido | None = None

    def para_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def de_dict(d: dict) -> "Perfil":
        d = dict(d)
        extras = set(d) - set(Perfil.__dataclass_fields__)
        if extras:
            raise ErroPerfil(f"campos desconhecidos no perfil: {', '.join(sorted(extras))}")
        for obrig in ("alias", "arquivo", "onde"):
            if not d.get(obrig):
                raise ErroPerfil(f"perfil sem '{obrig}'")
        try:
            if d.get("voz") is not None:
                d["voz"] = Voz(**d["voz"])
            if d.get("medido") is not None:
                d["medido"] = Medido(**d["medido"])
            d["flags"] = [str(x) for x in (d.get("flags") or [])]
            return Perfil(**d)
        except TypeError as e:
            raise ErroPerfil(f"perfil malformado: {e}") from e


def caminho_seguro(models_dir: Path, relativo: str) -> Path:
    """Resolve um caminho relativo a models_dir e recusa o que sair dele."""
    if not relativo or Path(relativo).is_absolute() or Path(relativo).drive or Path(relativo).root:
        raise ErroPerfil(f"caminho deve ser relativo a models_dir: {relativo!r}")
    base = Path(models_dir).resolve()
    alvo = (base / relativo).resolve()
    if alvo != base and base not in alvo.parents:
        raise ErroPerfil(f"caminho sai de models_dir: {relativo!r}")
    return alvo


def partes(caminho: Path) -> list[Path]:
    """GGUF dividido (-00001-of-00011): todas as partes; senão, só o próprio arquivo."""
    m = PARTE_RE.search(caminho.name)
    if not m:
        return [caminho]
    total = int(m.group(2))
    prefixo = caminho.name[:m.start()]
    return [caminho.with_name(f"{prefixo}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total + 1)]


def tamanho_gb(p: Perfil, models_dir: Path) -> float:
    arquivos = partes(caminho_seguro(models_dir, p.arquivo))
    if p.mmproj:
        arquivos.append(caminho_seguro(models_dir, p.mmproj))
    return sum(a.stat().st_size for a in arquivos if a.exists()) / 1024 ** 3


def validar(p: Perfil, models_dir: Path) -> None:
    if not ALIAS_RE.match(p.alias):
        raise ErroPerfil(f"alias inválido {p.alias!r}: use minúsculas, números, '.', '_' ou '-'")
    if p.alias == ALIAS_CORINGA:
        raise ErroPerfil(f"o alias '{ALIAS_CORINGA}' é reservado (aponta para o modelo no ar); escolha outro")
    if p.onde not in ONDE:
        raise ErroPerfil(f"onde deve ser um de {', '.join(ONDE)}; veio {p.onde!r}")
    desconhecidas = set(p.capacidades) - CAPACIDADES
    if desconhecidas:
        raise ErroPerfil(f"capacidades desconhecidas: {', '.join(sorted(desconhecidas))}; "
                         f"válidas: {', '.join(sorted(CAPACIDADES))}")
    for f in p.flags:
        nome = f.split("=", 1)[0]
        if nome in FLAGS_DO_SERVICO:
            raise ErroPerfil(f"a flag {nome} é controlada pelo serviço; tire do perfil")
        if nome in FLAGS_PROIBIDAS:
            raise ErroPerfil(f"a flag {nome} é recusada (std::bad_alloc perto do limite de RAM; use mmap, o padrão)")
    if p.voz is not None and not p.voz.vozes:
        raise ErroPerfil("o bloco 'voz' precisa de pelo menos uma voz em 'vozes'")
    for a in partes(caminho_seguro(models_dir, p.arquivo)):
        if not a.is_file():
            raise ErroPerfil(f"arquivo do modelo não existe: {a}")
    if p.mmproj and not caminho_seguro(models_dir, p.mmproj).is_file():
        raise ErroPerfil(f"mmproj não existe: {p.mmproj}")
    if ("stt" in p.capacidades or "tts" in p.capacidades) and p.voz is None:
        raise ErroPerfil("perfil com stt/tts precisa do bloco 'voz'")


def argv(p: Perfil, llama_server: Path, models_dir: Path, porta: int, ffmpeg_dir: Path | None) -> list[str]:
    cmd = [str(llama_server), "-m", str(caminho_seguro(models_dir, p.arquivo)), "--alias", p.alias, *p.flags]
    if p.mmproj:
        cmd += ["--mmproj", str(caminho_seguro(models_dir, p.mmproj))]
    if p.onde == "cpu":
        # medido: -ngl 0 sozinho ainda prende 0,5 GB de VRAM; --device none zera
        if "--device" not in p.flags and "-dev" not in p.flags:
            cmd += ["--device", "none"]
        if p.mmproj and "--no-mmproj-offload" not in p.flags:
            cmd += ["--no-mmproj-offload"]
    if p.video and ffmpeg_dir is not None:
        cmd += ["--video-ffmpeg-dir", str(ffmpeg_dir)]
    cmd += ["--host", "127.0.0.1", "--port", str(porta)]
    return cmd


class Catalogo:
    """CRUD de perfis em perfis.json, com escrita atômica (temporário + replace)."""

    def __init__(self, arquivo: Path, models_dir: Path):
        self.arquivo = Path(arquivo)
        self.models_dir = Path(models_dir)
        self._trava = threading.Lock()

    def _ler(self) -> list[Perfil]:
        if not self.arquivo.exists():
            return []
        dados = json.loads(self.arquivo.read_text(encoding="utf-8"))
        return [Perfil.de_dict(d) for d in dados.get("perfis", [])]

    def _gravar(self, perfis: list[Perfil]) -> None:
        self.arquivo.parent.mkdir(parents=True, exist_ok=True)
        texto = json.dumps({"perfis": [p.para_dict() for p in perfis]}, ensure_ascii=False, indent=2)
        fd, tmp = tempfile.mkstemp(dir=self.arquivo.parent, prefix=".perfis-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(texto + "\n")
            os.replace(tmp, self.arquivo)
        except OSError as e:
            Path(tmp).unlink(missing_ok=True)
            raise ErroPerfil(
                f"não consegui gravar {self.arquivo}: {e} (o arquivo está aberto em outro programa?)"
            ) from e

    def listar(self) -> list[Perfil]:
        with self._trava:
            return self._ler()

    def obter(self, alias: str) -> Perfil:
        perfis = self.listar()
        for p in perfis:
            if p.alias == alias:
                return p
        nomes = ", ".join(x.alias for x in perfis) or "(nenhum)"
        raise ErroPerfil(f"perfil {alias!r} não existe; perfis: {nomes}")

    def criar(self, p: Perfil) -> Perfil:
        validar(p, self.models_dir)
        with self._trava:
            perfis = self._ler()
            if any(x.alias == p.alias for x in perfis):
                raise ErroPerfil(f"já existe um perfil {p.alias!r}; use editar_perfil")
            perfis.append(p)
            self._gravar(perfis)
        return p

    def editar(self, alias: str, mudancas: dict) -> Perfil:
        if "alias" in mudancas and mudancas["alias"] != alias:
            raise ErroPerfil("não dá para trocar o alias; crie um perfil novo e apague o antigo")
        with self._trava:
            perfis = self._ler()
            for i, atual in enumerate(perfis):
                if atual.alias == alias:
                    base = atual.para_dict()
                    # blocos aninhados mesclam campo a campo: {"voz": {"idioma": "en"}} não apaga as vozes
                    for bloco in ("voz", "medido"):
                        if isinstance(mudancas.get(bloco), dict) and isinstance(base.get(bloco), dict):
                            mudancas = {**mudancas, bloco: {**base[bloco], **mudancas[bloco]}}
                    # a medição vale para o arquivo e as flags de quando foi feita
                    if "medido" not in mudancas and any(
                            k in mudancas and mudancas[k] != base.get(k) for k in ("arquivo", "mmproj", "onde", "flags")):
                        mudancas = {**mudancas, "medido": None}
                    novo = Perfil.de_dict({**base, **mudancas})
                    validar(novo, self.models_dir)
                    perfis[i] = novo
                    self._gravar(perfis)
                    return novo
        raise ErroPerfil(f"perfil {alias!r} não existe")

    def apagar(self, alias: str) -> None:
        with self._trava:
            perfis = self._ler()
            restantes = [p for p in perfis if p.alias != alias]
            if len(restantes) == len(perfis):
                raise ErroPerfil(f"perfil {alias!r} não existe")
            self._gravar(restantes)

    def gravar_medido(self, alias: str, medido: Medido) -> None:
        with self._trava:
            perfis = self._ler()
            for p in perfis:
                if p.alias == alias:
                    p.medido = medido
                    self._gravar(perfis)
                    return
