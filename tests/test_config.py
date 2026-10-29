import json

import pytest

from local_hub_mcp.config import ErroConfig, carregar


def _escrever(raiz, **extra):
    dados = {"llama_cpp_dir": "llama", "models_dir": "models", **extra}
    caminho = raiz / "config.json"
    caminho.write_text(json.dumps(dados), encoding="utf-8")
    return caminho


def test_carrega_com_padroes_e_caminhos_relativos(maquina):
    cfg = carregar(_escrever(maquina["raiz"]))
    assert cfg.porta == 8079
    assert cfg.portas_internas == (8180, 8199)
    assert cfg.models_dir == maquina["models"].resolve()
    assert cfg.perfis_arquivo == (maquina["raiz"] / "perfis.json").resolve()
    assert cfg.base_url == "http://127.0.0.1:8079"
    assert cfg.chaves == {}


def test_sem_llama_server_explica(maquina):
    with pytest.raises(ErroConfig, match="llama-server"):
        carregar(_escrever(maquina["raiz"], llama_cpp_dir="models"))


def test_porta_dentro_da_faixa_recusa(maquina):
    with pytest.raises(ErroConfig, match="dentro de portas_internas"):
        carregar(_escrever(maquina["raiz"], porta=8185))


def test_bind_de_rede_exige_chaves(maquina):
    with pytest.raises(ErroConfig, match="chaves.admin"):
        carregar(_escrever(maquina["raiz"], bind="0.0.0.0"))
    cfg = carregar(_escrever(maquina["raiz"], bind="0.0.0.0", chaves={"admin": "a", "inference": "b"}))
    assert cfg.base_url == "http://127.0.0.1:8079"


def test_config_ausente_explica(tmp_path):
    with pytest.raises(ErroConfig, match="config.exemplo.json"):
        carregar(tmp_path / "config.json")

def test_base_url_ipv6(maquina):
    assert carregar(_escrever(maquina["raiz"], bind="::1")).base_url == "http://[::1]:8079"


@pytest.mark.parametrize("porta", [-1, 0, 65536])
def test_porta_fora_do_intervalo(maquina, porta):
    with pytest.raises(ErroConfig, match="porta"):
        carregar(_escrever(maquina["raiz"], porta=porta))


def test_porta_null_usa_padrao(maquina):
    assert carregar(_escrever(maquina["raiz"], porta=None)).porta == 8079

@pytest.mark.parametrize("valor", [None, "tiny", [1], {}])
def test_subir_no_boot_rejeita_tipo(maquina, valor):
    with pytest.raises(ErroConfig, match="subir_no_boot"):
        carregar(_escrever(maquina["raiz"], subir_no_boot=valor))


def test_subir_no_boot_carrega(maquina):
    assert carregar(_escrever(maquina["raiz"], subir_no_boot=["tiny"])).subir_no_boot == ["tiny"]
    assert carregar(_escrever(maquina["raiz"])).subir_no_boot == []
