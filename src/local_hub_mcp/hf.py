"""Hugging Face: busca, detalhes do repositório (GGUF por quant, tamanho, se cabe) e download para models_dir."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .perfis import PARTE_RE, caminho_seguro
from .recursos import EstadoRecursos

QUANT_RE = re.compile(r"(IQ\d_[A-Z]+|Q\d_K_[SML]|Q\d_K|Q\d_\d|BF16|F16|F32)", re.I)
REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
GB = 1024 ** 3


class ErroHf(Exception):
    pass


def _repo(repo_id: str) -> str:
    if not REPO_RE.match(repo_id or "") or any(parte in {".", ".."} for parte in (repo_id or "").split("/")):
        raise ErroHf(f"repo_id inválido {repo_id!r}: use 'dono/nome'")
    return repo_id


def slug(repo_id: str) -> str:
    dono, nome = repo_id.rsplit("/", 1)
    nome = re.sub(r"[-_]?gguf$", "", nome, flags=re.I)
    dono = re.sub(r"[^a-z0-9.]+", "-", dono.lower()).strip("-")
    nome = re.sub(r"[^a-z0-9.]+", "-", nome.lower()).strip("-")
    return f"{dono}--{nome}"


def _grupo(nome: str) -> str:
    m = PARTE_RE.search(nome)
    return nome[:m.start()] + ".gguf" if m else nome


def agrupar(arquivos: list[tuple[str, int]]) -> list[dict]:
    """[(nome, bytes)] -> um item por modelo (partes juntas), com quant, tamanho e se é mmproj."""
    grupos: dict[str, dict] = {}
    for nome, tam in arquivos:
        if not nome.lower().endswith(".gguf"):
            continue
        chave = _grupo(nome)
        q = QUANT_RE.search(Path(chave).stem)
        g = grupos.setdefault(chave, {"nome": chave, "partes": [], "tamanho_gb": 0.0,
                                      "mmproj": "mmproj" in nome.lower(),
                                      "quant": q.group(1).upper() if q else None})
        g["partes"].append(nome)
        g["tamanho_gb"] += (tam or 0) / GB
    for g in grupos.values():
        g["partes"].sort()
        g["primeiro"] = g["partes"][0]
        g["tamanho_gb"] = round(g["tamanho_gb"], 2)
    return sorted(grupos.values(), key=lambda g: (g["mmproj"], g["tamanho_gb"]))


def cabe(tamanho_gb: float, estado: EstadoRecursos) -> str:
    """Estimativa pelo tamanho do arquivo: 'gpu', 'gpu+cpu', 'cpu' ou 'nao'."""
    livre = estado.gpu.livre_gb if estado.gpu else 0.0
    if estado.gpu and tamanho_gb + 0.8 <= livre:
        return "gpu"
    if tamanho_gb + 2.0 <= livre + estado.ram_livre_gb:
        return "gpu+cpu" if estado.gpu else "cpu"
    return "nao"


ORDENS = {"trending": "trending_score", "downloads": "downloads", "recentes": "last_modified", "likes": "likes"}
CAMPOS = ["tags", "downloads", "likes", "trendingScore", "lastModified"]


def _ordem(ordem: str) -> str:
    if ordem not in ORDENS:
        raise ErroHf(f"ordem deve ser uma de {', '.join(ORDENS)}; veio {ordem!r}")
    return ORDENS[ordem]


def _origem(tags: list[str] | None) -> str | None:
    """Repositório de onde a conversão veio, pela tag base_model do Hub."""
    for t in tags or []:
        if t.startswith("base_model:quantized:"):
            return t.split(":", 2)[2]
    for t in tags or []:
        if t.startswith("base_model:") and t.count(":") == 1:
            return t.split(":", 1)[1]
    return None


def _item(m) -> dict:
    data = getattr(m, "last_modified", None)
    return {"repo_id": m.id, "gguf": "gguf" in (m.tags or []), "origem": _origem(m.tags),
            "downloads": m.downloads, "likes": m.likes, "tendencia": getattr(m, "trending_score", None),
            "atualizado": data.date().isoformat() if data else None}


def procurar(texto: str, limite: int = 10, ordem: str = "trending", api: HfApi | None = None) -> dict:
    """Originais (o modelo publicado pelo autor, que mostra a versão atual) e conversões GGUF (o que dá para
    baixar), na ordem pedida. Derivados que não são GGUF (mlx, AWQ...) ficam de fora."""
    api = api or HfApi()
    sort = _ordem(ordem)
    texto = re.sub(r"^https?://(www\.)?huggingface\.co/", "", texto.strip()).strip("/")
    if REPO_RE.match(texto):   # link ou "dono/nome": o próprio repo e as conversões dele
        repo = texto.lower()
        achados = api.list_models(search=texto.split("/")[1], sort=sort, limit=100, expand=CAMPOS)
        itens = [_item(m) for m in achados if m.id.lower() == repo or (_origem(m.tags) or "").lower() == repo]
        itens = [i for i in itens if i["gguf"] or i["origem"] is None][:limite]
        return {"ordem": ordem, "originais": [i for i in itens if not i["gguf"]],
                "gguf": [i for i in itens if i["gguf"]],
                "dica": "para a lista completa de conversões GGUF deste repo, chame conversoes_gguf"}
    termos = [t for t in texto.split() if t]
    busca = texto if len(termos) <= 1 else max(termos, key=len)   # a busca do Hub é por trecho do nome
    achados = list(api.list_models(search=busca, sort=sort, limit=100, expand=CAMPOS))
    if len(termos) > 1:   # "baidu unlimited-ocr-2": todos os termos no repo_id
        achados = [m for m in achados if all(t.lower() in m.id.lower() for t in termos)]
    itens = [_item(m) for m in achados]
    itens = [i for i in itens if i["gguf"] or i["origem"] is None][:limite]
    return {"ordem": ordem, "originais": [i for i in itens if not i["gguf"]], "gguf": [i for i in itens if i["gguf"]],
            "dica": "os 'originais' não rodam direto no llama.cpp; para ver tudo o que dá para baixar de um deles, "
                    "chame conversoes_gguf com o repo_id dele"}


def conversoes_gguf(repo_origem: str, limite: int = 20, ordem: str = "downloads", api: HfApi | None = None) -> dict:
    """Todas as conversões GGUF que o Hub liga a um modelo original (tag base_model)."""
    api = api or HfApi()
    filtro = ["gguf", f"base_model:quantized:{_repo(repo_origem)}"]
    itens = [_item(m) for m in api.list_models(filter=filtro, sort=_ordem(ordem), limit=limite, expand=CAMPOS)]
    return {"origem": repo_origem, "ordem": ordem, "gguf": itens,
            "dica": "escolha um repo e chame detalhes_repo para ver quants, tamanhos e mmproj"}


def detalhes(repo_id: str, estado: EstadoRecursos, api: HfApi | None = None) -> dict:
    api = api or HfApi()
    info = api.model_info(_repo(repo_id), files_metadata=True)
    grupos = agrupar([(s.rfilename, s.size or 0) for s in info.siblings])
    for g in grupos:
        if not g["mmproj"]:
            g["cabe"] = cabe(g["tamanho_gb"], estado)
    return {"repo_id": repo_id, "pasta": slug(repo_id), "arquivos": grupos,
            "nota": "'cabe' é estimativa pelo tamanho do arquivo; o planejador decide de verdade ao subir"}


@dataclass
class Download:
    repo_id: str
    pasta: str
    destino: Path
    itens: list   # (nome no repo, bytes, caminho local, já existia)

    @property
    def total(self) -> int:
        return sum(b for _, b, _, _ in self.itens)


def preparar_download(repo_id: str, arquivos: list[str], models_dir: Path, api: HfApi | None = None) -> Download:
    """Valida o pedido e calcula o destino (rápido: os erros aparecem antes de começar a baixar)."""
    api = api or HfApi()
    info = api.model_info(_repo(repo_id), files_metadata=True)
    existentes = {s.rfilename: (s.size or 0) for s in info.siblings}
    if not arquivos:
        raise ErroHf("diga quais arquivos baixar (veja detalhes_repo)")
    pedidos: list[str] = []
    for nome in arquivos:
        if nome not in existentes:
            raise ErroHf(f"'{nome}' não existe em {repo_id}; veja detalhes_repo")
        if not nome.lower().endswith(".gguf"):
            raise ErroHf(f"só arquivos .gguf: '{nome}'")
        for n in sorted(existentes):
            if _grupo(n) == _grupo(nome) and n not in pedidos:
                pedidos.append(n)
    pasta = slug(repo_id)
    destino = caminho_seguro(models_dir, pasta)
    itens = []
    for n in pedidos:
        caminho = caminho_seguro(models_dir, f"{pasta}/{n}")
        ja_existia = caminho.exists()
        if ja_existia and (not caminho.is_file() or caminho.stat().st_size != existentes[n]):
            raise ErroHf(f"o arquivo {caminho} já existe com outro conteúdo; apague-o antes (apagar_arquivos) "
                         f"ou escolha outro")
        itens.append((n, existentes[n], caminho, ja_existia))
    return Download(repo_id, pasta, destino, itens)


def executar_download(d: Download, baixar_arquivo=hf_hub_download) -> dict:
    d.destino.mkdir(parents=True, exist_ok=True)
    for n, _, _, ja_existia in d.itens:
        if not ja_existia:
            baixar_arquivo(d.repo_id, n, local_dir=str(d.destino))
    return {"ok": True, "pasta": d.pasta,
            "arquivos": [{"arquivo": f"{d.pasta}/{n}", "ja_existia": je} for n, _, _, je in d.itens],
            "tamanho_gb": round(d.total / GB, 2),
            "proximo_passo": "criar_perfil com 'arquivo' = o campo 'arquivo' do primeiro item (e 'mmproj', se baixou um)"}


def _parcial(destino: Path, nome: str) -> int:
    """Bytes do temporário .incomplete DESTE arquivo (nome que o huggingface_hub calcula para ele)."""
    try:
        from huggingface_hub._local_folder import _short_hash, get_local_download_paths
        meta = get_local_download_paths(destino, nome).metadata_path
        return sum(f.stat().st_size for f in meta.parent.glob(f"{_short_hash(meta.name)}.*.incomplete"))
    except Exception:   # API interna do huggingface_hub mudou: sem parcial, só arquivos completos
        return 0


def progresso_download(d: Download) -> dict:
    """Bytes já no disco: arquivos completos + os temporários .incomplete que o huggingface_hub escreve."""
    completos = [c for _, b, c, _ in d.itens if c.is_file() and c.stat().st_size == b]
    prontos = sum(b for _, b, c, _ in d.itens if c in completos)
    parciais = sum(_parcial(d.destino, n) for n, _, c, _ in d.itens if c not in completos)
    feito = min(prontos + parciais, d.total)
    return {"baixado_gb": round(feito / GB, 2), "total_gb": round(d.total / GB, 2),
            "percentual": round(100 * feito / d.total, 1) if d.total else 100.0,
            "arquivos_prontos": len(completos), "arquivos_total": len(d.itens)}


def baixar(repo_id: str, arquivos: list[str], models_dir: Path, api: HfApi | None = None,
           baixar_arquivo=hf_hub_download) -> dict:
    """Baixa os arquivos pedidos (e todas as partes de um GGUF dividido) para models_dir/<pasta>/ (síncrono)."""
    return executar_download(preparar_download(repo_id, arquivos, models_dir, api), baixar_arquivo)
