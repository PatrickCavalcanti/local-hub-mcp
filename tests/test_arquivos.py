import pytest

from local_hub_mcp.arquivos import planejar, remover
from local_hub_mcp.perfis import ErroPerfil, Perfil


@pytest.fixture
def modelos(maquina):
    m = maquina["models"]
    (m / "org--novo").mkdir()
    (m / "org--novo" / "Novo-Q4-00001-of-00002.gguf").write_bytes(b"a" * 300)
    (m / "org--novo" / "Novo-Q4-00002-of-00002.gguf").write_bytes(b"b" * 200)
    (m / "org--novo" / ".cache").mkdir()
    (m / "org--novo" / ".cache" / "lixo.incomplete").write_bytes(b"c" * 10)
    return m


PERFIL = [Perfil(alias="tiny", arquivo="tiny/tiny-Q4_K_M.gguf", onde="gpu", mmproj="tiny/mmproj-tiny-F16.gguf")]


def test_planejar_nao_apaga_e_lista_partes(modelos):
    r = planejar(modelos, ["org--novo/Novo-Q4-00001-of-00002.gguf"], PERFIL)
    assert [a["caminho"] for a in r["arquivos"]] == ["org--novo/Novo-Q4-00001-of-00002.gguf",
                                                     "org--novo/Novo-Q4-00002-of-00002.gguf"]
    assert (modelos / "org--novo" / "Novo-Q4-00001-of-00002.gguf").exists() and r["confirmacao"]


def test_remover_pasta_com_codigo_certo(modelos):
    codigo = planejar(modelos, ["org--novo"], PERFIL)["confirmacao"]
    r = remover(modelos, ["org--novo"], PERFIL, codigo)
    assert len(r["apagados"]) == 3 and r["falhas"] == [] and not (modelos / "org--novo").exists()
    assert (modelos / "tiny" / "tiny-Q4_K_M.gguf").exists()


def test_codigo_errado_ou_lista_mudou_recusa(modelos):
    codigo = planejar(modelos, ["org--novo"], PERFIL)["confirmacao"]
    with pytest.raises(ErroPerfil, match="não bate"):
        remover(modelos, ["org--novo"], PERFIL, "errado")
    (modelos / "org--novo" / "extra.gguf").write_bytes(b"d")
    with pytest.raises(ErroPerfil, match="não bate"):
        remover(modelos, ["org--novo"], PERFIL, codigo)
    assert (modelos / "org--novo" / "Novo-Q4-00001-of-00002.gguf").exists()


def test_recusa_arquivo_de_perfil_e_fora_da_pasta(modelos):
    with pytest.raises(ErroPerfil, match="perfil.*tiny"):
        planejar(modelos, ["tiny"], PERFIL)
    with pytest.raises(ErroPerfil, match="perfil.*tiny"):
        planejar(modelos, ["tiny/mmproj-tiny-F16.gguf"], PERFIL)
    with pytest.raises(ErroPerfil):
        planejar(modelos, ["../fora"], PERFIL)
    with pytest.raises(ErroPerfil, match="inteira"):
        planejar(modelos, ["."], PERFIL)
    with pytest.raises(ErroPerfil, match="não existe"):
        planejar(modelos, ["nada.gguf"], PERFIL)



def test_arquivo_criado_depois_do_plano_nao_e_apagado(modelos, monkeypatch):
    """O código confere a lista enumerada UMA vez; um arquivo que surge depois muda o código e nada é apagado."""
    import local_hub_mcp.arquivos as mod
    codigo = planejar(modelos, ["org--novo"], PERFIL)["confirmacao"]
    original = mod._alvos

    def alvos_e_surge_um_arquivo(models_dir, caminhos):
        r = original(models_dir, caminhos)
        (modelos / "org--novo" / "surgiu.gguf").write_bytes(b"z")   # depois da enumeração
        return r
    monkeypatch.setattr(mod, "_alvos", alvos_e_surge_um_arquivo)
    r = remover(modelos, ["org--novo"], PERFIL, codigo)
    assert (modelos / "org--novo" / "surgiu.gguf").exists()      # não estava na lista: ficou
    assert "org--novo/surgiu.gguf" not in r["apagados"] and (modelos / "org--novo").is_dir()


def test_arquivo_de_processo_no_ar_e_protegido_mesmo_fora_do_catalogo(modelos):
    no_ar = {(modelos / "org--novo" / "Novo-Q4-00001-of-00002.gguf").resolve(): "rodando"}
    with pytest.raises(ErroPerfil, match="rodando"):
        planejar(modelos, ["org--novo"], [], no_ar)
