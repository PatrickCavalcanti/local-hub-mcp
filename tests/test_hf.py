from types import SimpleNamespace

import pytest

from local_hub_mcp.hf import ErroHf, agrupar, baixar, cabe, detalhes, slug
from local_hub_mcp.recursos import EstadoRecursos, Gpu

GB = 1024 ** 3
ARQS = [("README.md", 10), ("M-Q4_K_M.gguf", int(1.1 * GB)), ("M-Q8_0.gguf", int(1.8 * GB)),
        ("mmproj-M-F16.gguf", int(0.8 * GB)),
        ("Big-IQ4_XS-00001-of-00002.gguf", 5 * GB), ("Big-IQ4_XS-00002-of-00002.gguf", 5 * GB)]


class ApiFalsa:
    def model_info(self, repo, files_metadata=False):
        return SimpleNamespace(siblings=[SimpleNamespace(rfilename=n, size=s) for n, s in ARQS])


def test_slug():
    assert slug("LiquidAI/LFM3-Omni-2B-GGUF") == "liquidai--lfm3-omni-2b"
    assert slug("org/Modelo_X") == "org--modelo-x"


def test_agrupar():
    g = {x["nome"]: x for x in agrupar(ARQS)}
    assert "README.md" not in g
    assert g["Big-IQ4_XS.gguf"]["partes"] == ["Big-IQ4_XS-00001-of-00002.gguf", "Big-IQ4_XS-00002-of-00002.gguf"]
    assert g["Big-IQ4_XS.gguf"]["tamanho_gb"] == 10.0
    assert g["M-Q4_K_M.gguf"]["quant"] == "Q4_K_M"
    assert g["mmproj-M-F16.gguf"]["mmproj"] is True


def test_cabe():
    e = EstadoRecursos(Gpu(6.0, 4.0), 32.0, 20.0)
    assert cabe(1.1, e) == "gpu"
    assert cabe(15.4, e) == "gpu+cpu"
    assert cabe(40.0, e) == "nao"
    assert cabe(2.0, EstadoRecursos(None, 32.0, 20.0)) == "cpu"


def test_detalhes():
    d = detalhes("org/M-GGUF", EstadoRecursos(Gpu(6.0, 4.0), 32.0, 20.0), api=ApiFalsa())
    assert d["pasta"] == "org--m"
    assert all("cabe" in a for a in d["arquivos"] if not a["mmproj"])


def test_baixar_expande_partes(tmp_path):
    chamadas = []
    r = baixar("org/Big-GGUF", ["Big-IQ4_XS-00001-of-00002.gguf"], tmp_path, api=ApiFalsa(),
               baixar_arquivo=lambda repo, nome, local_dir: chamadas.append((repo, nome, local_dir)))
    assert [c[1] for c in chamadas] == ["Big-IQ4_XS-00001-of-00002.gguf", "Big-IQ4_XS-00002-of-00002.gguf"]
    assert r["arquivos"][0] == {
        "arquivo": "org--big/Big-IQ4_XS-00001-of-00002.gguf", "ja_existia": False}
    assert (tmp_path / "org--big").is_dir()


def test_baixar_reutiliza_arquivo_existente_com_mesmo_tamanho(tmp_path):
    arq = "Tiny-Q4_K_M.gguf"
    api = SimpleNamespace(model_info=lambda *args, **kwargs: SimpleNamespace(
        siblings=[SimpleNamespace(rfilename=arq, size=10)]))
    destino = tmp_path / "org--tiny"
    destino.mkdir()
    (destino / arq).write_bytes(b"x" * 10)

    def nao_baixar(*args, **kwargs):
        raise AssertionError("arquivo do mesmo tamanho não deve ser baixado novamente")

    r = baixar("org/Tiny-GGUF", [arq], tmp_path, api=api, baixar_arquivo=nao_baixar)

    assert r["arquivos"] == [{"arquivo": f"org--tiny/{arq}", "ja_existia": True}]


def test_baixar_recusa_sobrescrever_arquivo_com_tamanho_diferente(tmp_path):
    arq = "Tiny-Q4_K_M.gguf"
    api = SimpleNamespace(model_info=lambda *args, **kwargs: SimpleNamespace(
        siblings=[SimpleNamespace(rfilename=arq, size=10)]))
    destino = tmp_path / "org--tiny"
    destino.mkdir()
    caminho = destino / arq
    caminho.write_bytes(b"x" * 9)

    with pytest.raises(ErroHf, match="já existe com outro conteúdo"):
        baixar("org/Tiny-GGUF", [arq], tmp_path, api=api,
               baixar_arquivo=lambda *args, **kwargs: pytest.fail("não deve sobrescrever"))
    assert caminho.read_bytes() == b"x" * 9


def test_baixar_recusa():
    with pytest.raises(ErroHf, match="não existe"):
        baixar("org/M", ["nada.gguf"], "x", api=ApiFalsa(), baixar_arquivo=None)
    with pytest.raises(ErroHf, match=".gguf"):
        baixar("org/M", ["README.md"], "x", api=ApiFalsa(), baixar_arquivo=None)
    with pytest.raises(ErroHf, match="repo_id"):
        baixar("../x", ["a.gguf"], "x", api=ApiFalsa(), baixar_arquivo=None)


def _modelo(repo, tags, downloads=10, trend=1):
    import datetime
    return SimpleNamespace(id=repo, tags=tags, downloads=downloads, likes=1, trending_score=trend,
                           last_modified=datetime.datetime(2026, 10, 28))


class ApiDeBusca:
    def __init__(self, modelos):
        self.modelos, self.chamadas = modelos, []

    def list_models(self, **kw):
        self.chamadas.append(kw)
        return list(self.modelos)


BUSCA = [
    _modelo("baidu/Unlimited-OCR-2", ["transformers"], 18000, 344),
    _modelo("baidu/Unlimited-OCR", ["transformers"], 3_500_000, 50),
    _modelo("AtomicChat/Unlimited-OCR-2-GGUF", ["gguf", "base_model:baidu/Unlimited-OCR-2",
                                                "base_model:quantized:baidu/Unlimited-OCR-2"], 2500, 8),
    _modelo("mlx-community/Unlimited-OCR-2-8bit", ["mlx", "base_model:quantized:baidu/Unlimited-OCR-2"], 355, 3),
]


def test_procurar_traz_originais_e_gguf_com_origem_em_trending():
    from local_hub_mcp.hf import procurar
    api = ApiDeBusca(BUSCA)
    r = procurar("Unlimited-OCR", api=api)
    assert api.chamadas[0]["sort"] == "trending_score" and "filter" not in api.chamadas[0]
    assert [o["repo_id"] for o in r["originais"]] == ["baidu/Unlimited-OCR-2", "baidu/Unlimited-OCR"]
    assert r["gguf"] == [{"repo_id": "AtomicChat/Unlimited-OCR-2-GGUF", "gguf": True, "origem": "baidu/Unlimited-OCR-2",
                          "downloads": 2500, "likes": 1, "tendencia": 8, "atualizado": "2026-10-28"}]
    assert all("mlx" not in i["repo_id"] for i in r["originais"] + r["gguf"])


def test_procurar_com_varios_termos_exige_todos_no_nome():
    from local_hub_mcp.hf import procurar
    api = ApiDeBusca(BUSCA)
    r = procurar("baidu unlimited-ocr-2", api=api)
    assert api.chamadas[0]["search"] == "unlimited-ocr-2"
    assert [o["repo_id"] for o in r["originais"]] == ["baidu/Unlimited-OCR-2"] and r["gguf"] == []


def test_procurar_ordem_invalida():
    from local_hub_mcp.hf import procurar
    with pytest.raises(ErroHf, match="ordem"):
        procurar("x", ordem="aleatoria", api=ApiDeBusca([]))


def test_conversoes_gguf_filtra_pela_origem():
    from local_hub_mcp.hf import conversoes_gguf
    api = ApiDeBusca(BUSCA[2:3])
    r = conversoes_gguf("baidu/Unlimited-OCR-2", api=api)
    assert api.chamadas[0]["filter"] == ["gguf", "base_model:quantized:baidu/Unlimited-OCR-2"]
    assert r["origem"] == "baidu/Unlimited-OCR-2" and r["gguf"][0]["repo_id"] == "AtomicChat/Unlimited-OCR-2-GGUF"
    with pytest.raises(ErroHf):
        conversoes_gguf("../x", api=api)


@pytest.mark.parametrize("texto", ["https://huggingface.co/baidu/Unlimited-OCR-2", "baidu/Unlimited-OCR-2"])
def test_procurar_por_link_ou_repo_traz_o_original_e_suas_conversoes(texto):
    from local_hub_mcp.hf import procurar
    api = ApiDeBusca(BUSCA)
    r = procurar(texto, api=api)
    assert api.chamadas[0]["search"] == "Unlimited-OCR-2"
    assert [o["repo_id"] for o in r["originais"]] == ["baidu/Unlimited-OCR-2"]
    assert [g["repo_id"] for g in r["gguf"]] == ["AtomicChat/Unlimited-OCR-2-GGUF"]


def test_progresso_do_download_conta_completos_e_parciais(tmp_path):
    from local_hub_mcp.hf import Download, progresso_download
    destino = tmp_path / "org--m"
    (destino / ".cache" / "huggingface" / "download").mkdir(parents=True)
    pronto = destino / "a.gguf"
    pronto.write_bytes(b"x" * 100)
    from huggingface_hub._local_folder import _short_hash, get_local_download_paths
    meta = get_local_download_paths(destino, "b.gguf").metadata_path
    (meta.parent / f"{_short_hash(meta.name)}.etag123.incomplete").write_bytes(b"x" * 50)   # temporário do b.gguf
    (meta.parent / "outro-download.etag9.incomplete").write_bytes(b"x" * 999)   # de outra tarefa: não conta
    d = Download("org/m", "org--m", destino, [("a.gguf", 100, pronto, False), ("b.gguf", 200, destino / "b.gguf", False)])
    p = progresso_download(d)
    assert p["percentual"] == 50.0 and p["arquivos_prontos"] == 1 and p["arquivos_total"] == 2
