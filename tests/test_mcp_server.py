import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn

from local_hub_mcp.app import criar_app
from local_hub_mcp.config import Config
from local_hub_mcp.mcp_server import Contexto
from local_hub_mcp.perfis import Catalogo, Perfil
from local_hub_mcp.recursos import EstadoRecursos, Gpu
from local_hub_mcp.supervisor import Supervisor
from local_hub_mcp.tarefas import Tarefas

FAKE = Path(__file__).resolve().parent / "fake_llama_server.py"
CONTEXTOS = {}


class RecursosFalsos:
    def ler_estado(self):
        return EstadoRecursos(Gpu(6.0, 5.0), 32.0, 20.0)

    def memoria_gpu_processos(self):
        return {}

    def ram_processo_gb(self, pid):
        return 0.1


def porta_livre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def servico(maquina, tmp_path, request):
    cfg = Config(llama_cpp_dir=maquina["llama"], models_dir=maquina["models"], perfis_arquivo=tmp_path / "perfis.json",
                 logs_dir=tmp_path / "logs", porta=porta_livre(), portas_internas=(18190, 18199), espera_s=30,
                 subir_no_boot=getattr(request, "param", []))
    cat = Catalogo(cfg.perfis_arquivo, cfg.models_dir)
    cat.criar(Perfil(alias="tiny", arquivo="tiny/tiny-Q4_K_M.gguf", onde="gpu", capacidades=["llm.chat"]))
    sup = Supervisor(cfg, cat, RecursosFalsos(), prefixo=[sys.executable, str(FAKE)])
    ctx = Contexto(cfg, cat, sup, Tarefas(), None, "teste")
    servidor = uvicorn.Server(uvicorn.Config(criar_app(ctx), host="127.0.0.1", port=cfg.porta, log_level="warning"))
    t = threading.Thread(target=servidor.run, daemon=True)
    t.start()
    for _ in range(100):
        if servidor.started:
            break
        time.sleep(0.05)
    assert servidor.started, "uvicorn não iniciou"
    CONTEXTOS[cfg.porta] = ctx
    yield cfg
    servidor.should_exit = True
    t.join(10)
    CONTEXTOS.pop(cfg.porta, None)


async def chamar(url, tool, args=None):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    async with streamable_http_client(url) as (r, w, *_):
        async with ClientSession(r, w) as s:
            await s.initialize()
            if tool is None:
                return [t.name for t in (await s.list_tools()).tools]
            res = await s.call_tool(tool, args or {})
            return json.loads(res.content[0].text)


async def test_ferramentas_expostas(servico):
    nomes = await chamar(f"{servico.base_url}/mcp", None)
    assert {"status", "listar_modelos", "procurar_modelo", "conversoes_gguf", "detalhes_repo", "baixar_modelo", "criar_perfil",
            "editar_perfil", "apagar_perfil", "apagar_arquivos", "medir", "subir_modelo", "derrubar_modelo", "testar_modelo",
            "logs", "info_conexao", "tarefa"} <= set(nomes)


async def test_fluxo_subir_usar_derrubar(servico):
    import httpx
    url = f"{servico.base_url}/mcp"
    async with httpx.AsyncClient() as h:
        r = await h.post(f"{servico.base_url}/v1/chat/completions", json={"model": "tiny", "messages": []})
        assert r.status_code == 409
    sub = await chamar(url, "subir_modelo", {"alias": "tiny"})
    assert sub["estado"] == "ok" and sub["resultado"]["estado"] == "no_ar", sub
    teste = await chamar(url, "testar_modelo", {"alias": "tiny"})
    assert teste["ok"] is True, teste
    async with httpx.AsyncClient() as h:
        r = await h.post(f"{servico.base_url}/v1/chat/completions", json={"model": "tiny", "messages": []})
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "pronto"
    st = await chamar(url, "status")
    assert st["processos"][0]["alias"] == "tiny"
    der = await chamar(url, "derrubar_modelo", {"alias": "tiny"})
    assert der["ok"] is True


async def test_erro_esperado_vira_ok_false(servico):
    r = await chamar(f"{servico.base_url}/mcp", "subir_modelo", {"alias": "nao-existe"})
    assert r["estado"] == "erro" and "não existe" in r["erro"]
    r = await chamar(f"{servico.base_url}/mcp", "criar_perfil", {"alias": "x", "arquivo": "../fora.gguf", "onde": "gpu"})
    assert r["ok"] is False and "models_dir" in r["erro"]


async def test_listar_modelos_ignora_partes_gguf_apos_a_primeira(servico):
    (servico.models_dir / "x-00001-of-00120.gguf").write_bytes(b"x")
    (servico.models_dir / "x-00100-of-00120.gguf").write_bytes(b"x")
    r = await chamar(f"{servico.base_url}/mcp", "listar_modelos")
    assert r["ok"] is True
    assert "x-00001-of-00120.gguf" in r["sem_perfil"]
    assert "x-00100-of-00120.gguf" not in r["sem_perfil"]


async def test_apagar_perfil_com_processo_caido(servico):
    ctx = CONTEXTOS[servico.porta]
    ctx.supervisor.processos["tiny"] = SimpleNamespace(estado="caiu", alias="tiny", porta=1,
                                                       popen=SimpleNamespace(poll=lambda: 0), log=[])
    try:
        r = await chamar(f"{servico.base_url}/mcp", "apagar_perfil", {"alias": "tiny"})
        assert r["ok"] is True
        assert "tiny" not in ctx.supervisor.processos
        modelos = await chamar(f"{servico.base_url}/mcp", "listar_modelos")
        assert all(p["alias"] != "tiny" for p in modelos["perfis"])
    finally:
        ctx.supervisor.processos.pop("tiny", None)


async def test_editar_perfil_com_parametros_tipados(servico):
    url = f"{servico.base_url}/mcp"
    r = await chamar(url, "editar_perfil", {"alias": "tiny", "flags": ["-c", "8192"], "notas": "editado"})
    assert r["ok"] is True and r["perfil"]["flags"] == ["-c", "8192"] and r["perfil"]["notas"] == "editado"
    vazio = await chamar(url, "editar_perfil", {"alias": "tiny"})
    assert vazio["ok"] is False and "nenhum campo" in vazio["erro"]

async def test_editar_no_ar_avisa(servico):
    url = f"{servico.base_url}/mcp"
    await chamar(url, "subir_modelo", {"alias": "tiny"})
    r = await chamar(url, "editar_perfil", {"alias": "tiny", "flags": ["-c", "1024"]})
    assert r["aviso"] == "o perfil está no ar; a mudança vale depois de derrubar_modelo e subir_modelo"
    r = await chamar(url, "subir_modelo", {"alias": "tiny"})
    assert "mudou desde que subiu" in r["erro"]


@pytest.mark.parametrize("servico", [["tiny"]], indirect=True)
async def test_subir_no_boot(servico):
    url = f"{servico.base_url}/mcp"
    for _ in range(100):
        st = await chamar(url, "status")
        if any(p["alias"] == "tiny" and p["estado"] == "no_ar" for p in st["processos"]):
            break
        await asyncio.sleep(0.1)
    assert any(p["alias"] == "tiny" and p["estado"] == "no_ar" for p in st["processos"])


async def test_status_medicoes(servico):
    ctx = CONTEXTOS[servico.porta]
    ctx.supervisor.medicoes["bench"] = "gpu"
    try:
        st = await chamar(f"{servico.base_url}/mcp", "status")
        assert st["medicoes"] == [{"alias": "bench", "onde": "gpu"}]
    finally:
        ctx.supervisor.medicoes.clear()


async def test_apagar_arquivos_em_duas_etapas(servico):
    url = f"{servico.base_url}/mcp"
    pasta = servico.models_dir / "org--sobra"
    pasta.mkdir()
    (pasta / "x.gguf").write_bytes(b"x" * 64)
    plano = await chamar(url, "apagar_arquivos", {"caminhos": ["org--sobra"]})
    assert plano["ok"] and plano["apagado"] is False and (pasta / "x.gguf").exists()
    feito = await chamar(url, "apagar_arquivos", {"caminhos": ["org--sobra"], "confirmacao": plano["confirmacao"]})
    assert feito["ok"] and feito["apagado"] is True and not pasta.exists()
    preso = await chamar(url, "apagar_arquivos", {"caminhos": ["tiny"]})
    assert preso["ok"] is False and "perfil" in preso["erro"]
