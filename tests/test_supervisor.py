import asyncio
import contextlib
import io
from unittest.mock import Mock
import json
import sys
from pathlib import Path

import pytest

from local_hub_mcp.config import Config
from local_hub_mcp.perfis import Catalogo, Medido, Perfil
from local_hub_mcp.recursos import EstadoRecursos, Gpu, MemoriaProcesso
from local_hub_mcp.supervisor import ErroSupervisor, Supervisor, porta_responde

FAKE = Path(__file__).resolve().parent / "fake_llama_server.py"


class RecursosFalsos:
    def __init__(self, livre=5.0, ram=20.0, compartilhada=0.0):
        self.livre, self.ram, self.compartilhada = livre, ram, compartilhada

    def ler_estado(self):
        return EstadoRecursos(Gpu(6.0, self.livre), 32.0, self.ram)

    def memoria_gpu_processos(self):
        return self._mem

    def ram_processo_gb(self, pid):
        return 0.1

    _mem = {}


@pytest.fixture
def montar(maquina, tmp_path):
    def _montar(recursos=None, timeout=20, **perfil_kw):
        cfg = Config(llama_cpp_dir=maquina["llama"], models_dir=maquina["models"],
                     perfis_arquivo=tmp_path / "perfis.json", logs_dir=tmp_path / "logs",
                     portas_internas=(18180, 18189), load_timeout_s=timeout)
        cat = Catalogo(cfg.perfis_arquivo, cfg.models_dir)
        kw = dict(alias="tiny", arquivo="tiny/tiny-Q4_K_M.gguf", onde="gpu")
        kw.update(perfil_kw)
        cat.criar(Perfil(**kw))
        sup = Supervisor(cfg, cat, recursos or RecursosFalsos(), prefixo=[sys.executable, str(FAKE)])
        return sup, cat
    return _montar


async def test_subir_e_derrubar(montar):
    sup, cat = montar(onde="cpu")
    r = await sup.subir("tiny")
    try:
        assert r["ok"] and r["estado"] == "no_ar" and 18180 <= r["porta"] <= 18189
        assert sup.porta_de("tiny") == r["porta"]
        assert cat.obter("tiny").medido is not None
        assert any("fake llama-server" in l for l in sup.logs("tiny"))
        again = await sup.subir("tiny")
        assert again["avisos"] == ["já estava no ar"]
    finally:
        d = await sup.derrubar("tiny")
    assert d["ok"] and not porta_responde(r["porta"])
    assert sup.porta_de("tiny") is None


async def test_processo_que_sai_vira_erro_com_log(montar):
    sup, _ = montar(flags=["--falhar"])
    with pytest.raises(ErroSupervisor, match="código 1"):
        await sup.subir("tiny")
    assert sup.status()[0]["estado"] == "caiu"


async def test_espera_o_health_ficar_pronto(montar):
    sup, _ = montar(flags=["--atraso", "2"])
    r = await sup.subir("tiny")
    try:
        assert r["carga_s"] >= 2
    finally:
        await sup.derrubar("tiny")


async def test_timeout_derruba(montar):
    sup, _ = montar(flags=["--atraso", "60"], timeout=2)
    with pytest.raises(ErroSupervisor, match="não ficou pronto"):
        await sup.subir("tiny")
    proc = sup.processos["tiny"]
    assert proc.estado == "caiu"
    assert proc.popen.poll() is not None
    assert not porta_responde(proc.porta)


async def test_planejador_recusa(montar):
    sup, _ = montar(recursos=RecursosFalsos(livre=0.1))
    with pytest.raises(ErroSupervisor, match="não cabe"):
        await sup.subir("tiny")
    assert sup.processos == {}


async def test_transbordo_vira_aviso(montar):
    rec = RecursosFalsos()
    sup, _ = montar(recursos=rec)

    class Mem(dict):
        def get(self, pid, padrao=None):
            return MemoriaProcesso(dedicada_gb=1.0, compartilhada_gb=0.9)

    rec._mem = Mem()
    r = await sup.subir("tiny")
    try:
        assert any("transbordou" in a for a in r["avisos"])
    finally:
        await sup.derrubar("tiny")


async def test_derrubar_o_que_nao_subiu(montar):
    sup, _ = montar()
    with pytest.raises(ErroSupervisor, match="não está no ar"):
        await sup.derrubar("tiny")


async def test_desligar_mata_todos(montar):
    sup, _ = montar()
    r = await sup.subir("tiny")
    await sup.desligar()
    assert sup.processos == {} and not porta_responde(r["porta"])


async def test_derrubar_durante_a_subida_nao_devolve_no_ar(montar):
    import asyncio
    sup, _ = montar()
    subida = asyncio.create_task(sup.subir("tiny"))
    for _ in range(200):   # espera chegar à espera pós-health (estado no_ar, ainda sem resultado)
        p = sup.processos.get("tiny")
        if p and p.estado == "no_ar":
            break
        await asyncio.sleep(0.05)
    await sup.derrubar("tiny")
    with pytest.raises(ErroSupervisor, match="derrubado"):
        await subida


async def test_cancelar_subida(montar):
    sup, _ = montar(flags=["--atraso", "30"])
    task = asyncio.create_task(sup.subir("tiny"))
    try:
        await asyncio.sleep(1)
        proc = sup.processos["tiny"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert proc.estado == "caiu"
        assert proc.popen.poll() is not None
        assert not porta_responde(proc.porta)
    finally:
        await sup.desligar()


async def test_vigiar_subindo_morto(montar):
    sup, cat = montar()
    proc = sup._iniciar(cat.obter("tiny"), sup._escolher_porta())
    sup.processos["tiny"] = proc
    await sup._matar(proc)
    task = asyncio.create_task(sup.vigiar(0.01))
    try:
        await asyncio.sleep(0.05)
        assert proc.estado == "caiu"
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await sup.desligar()


async def test_aviso_subindo(montar):
    sup, cat = montar(flags=["--atraso", "30"])
    sup.processos["tiny"] = sup._iniciar(cat.obter("tiny"), sup._escolher_porta())
    try:
        r = await sup.subir("tiny")
        assert r["estado"] == "subindo"
        assert r["avisos"] == ["já está subindo em outra chamada; consulte status em alguns segundos"]
    finally:
        await sup.desligar()


async def test_cpu_subindo_nao_bloqueia_gpu(montar):
    sup, cat = montar(onde="cpu", flags=["--atraso", "30"])
    sup.processos["tiny"] = sup._iniciar(cat.obter("tiny"), sup._escolher_porta())
    cat.criar(Perfil(alias="gpu", arquivo="tiny/tiny-Q4_K_M.gguf", onde="gpu"))
    try:
        assert (await sup.subir("gpu"))["estado"] == "no_ar"
    finally:
        await sup.desligar()


async def test_derrubado_durante_health(montar):
    sup, _ = montar(flags=["--atraso", "30"])
    task = asyncio.create_task(sup.subir("tiny"))
    try:
        await asyncio.sleep(1)
        await sup.derrubar("tiny")
        with pytest.raises(ErroSupervisor, match="'tiny' foi derrubado enquanto subia"):
            await task
    finally:
        await sup.desligar()
        with contextlib.suppress(ErroSupervisor):
            await task


async def test_marcar_caiu_confere_processo_real(montar):
    sup, _ = montar()
    try:
        await sup.subir("tiny")
        proc = sup.processos["tiny"]
        sup.marcar_caiu("tiny", "stream quebrou")
        assert proc.estado == "no_ar"
        assert porta_responde(proc.porta)
        assert sup.logs("tiny")[-1] == "[local-hub-mcp] stream interrompido com o processo vivo: stream quebrou"
        proc.popen.kill()
        await asyncio.to_thread(proc.popen.wait, 5)
        sup.marcar_caiu("tiny", "saiu")
        assert proc.estado == "caiu"
    finally:
        await sup.desligar()


async def test_subir_encerra_anterior_caido_ainda_vivo(montar, monkeypatch):
    sup, _ = montar()
    anterior = None
    try:
        await sup.subir("tiny")
        anterior = sup.processos["tiny"]
        anterior.estado = "caiu"
        iniciar = sup._iniciar

        def conferir_antes_de_iniciar(perfil, porta):
            assert anterior.popen.poll() is not None
            assert not porta_responde(anterior.porta)
            return iniciar(perfil, porta)

        monkeypatch.setattr(sup, "_iniciar", conferir_antes_de_iniciar)
        novo = await sup.subir("tiny")
        assert novo["pid"] != anterior.pid
        assert novo["estado"] == "no_ar"
    finally:
        await sup.desligar()
        if anterior is not None:
            await sup._matar(anterior)


@pytest.mark.parametrize("falha", ["abrir", "rotacionar", "escrever"])
def test_log_falho_drena_pipe(montar, monkeypatch, falha):
    sup, _ = montar()
    sup.cfg.logs_dir.mkdir(parents=True)
    if falha == "rotacionar":
        (sup.cfg.logs_dir / "tiny.log").write_bytes(b"x" * (5 * 1024 * 1024 + 1))
        monkeypatch.setattr(Path, "replace", Mock(side_effect=OSError("rotacao")))
    elif falha == "abrir":
        monkeypatch.setattr("builtins.open", Mock(side_effect=OSError("abertura")))
    else:
        from unittest.mock import MagicMock
        arquivo = MagicMock()
        arquivo.__enter__.return_value.write.side_effect = OSError("escrita")
        monkeypatch.setattr("builtins.open", Mock(return_value=arquivo))
    proc = Mock(alias="tiny", porta=18180, log=[], popen=Mock(stdout=io.BytesIO(b"linha 1\nlinha 2\n")))
    sup._ler_log(proc)
    assert proc.popen.stdout.read() == b""
    assert "linha 2" in proc.log


async def test_subir_depois_de_desligar_recusa(montar):
    sup, _ = montar()
    await sup.desligar()
    with pytest.raises(ErroSupervisor, match="desligando"):
        await sup.subir("tiny")
    assert sup.processos == {}


async def test_fantasma_caido_nao_faz_o_proprio_subir_recusar(montar):
    """A VRAM do processo anterior (caído mas vivo) não pode contar contra o novo: ele morre antes de medir."""
    sup, _ = montar()
    anterior = None
    try:
        await sup.subir("tiny")
        anterior = sup.processos["tiny"]
        anterior.estado = "caiu"   # o processo segue vivo, segurando "VRAM"

        class RecursosQueSentemOFantasma(RecursosFalsos):
            def ler_estado(self):
                livre = 0.1 if anterior.popen.poll() is None else 5.0
                return EstadoRecursos(Gpu(6.0, livre), 32.0, 20.0)

        sup.recursos = RecursosQueSentemOFantasma()
        novo = await sup.subir("tiny")
        assert novo["estado"] == "no_ar" and novo["pid"] != anterior.pid
        assert anterior.popen.poll() is not None
    finally:
        await sup.desligar()
        if anterior is not None:
            await sup._matar(anterior)


async def test_no_ar_que_nao_responde_pede_derrubar(montar):
    sup, _ = montar()
    try:
        await sup.subir("tiny")
        proc = sup.processos["tiny"]
        proc.porta = 1   # simula o processo vivo que parou de responder
        with pytest.raises(ErroSupervisor, match="derrubar_modelo"):
            await sup.subir("tiny")
    finally:
        await sup.desligar()



async def test_no_ar_que_ja_morreu_e_recolhido_e_sobe_de_novo(montar):
    sup, _ = montar()
    try:
        await sup.subir("tiny")
        antigo = sup.processos["tiny"]
        antigo.popen.kill(); antigo.popen.wait(5)   # morreu e o vigiar ainda não viu
        novo = await sup.subir("tiny")
        assert novo["estado"] == "no_ar" and novo["pid"] != antigo.pid
    finally:
        await sup.desligar()

async def test_registro_medicao_bloqueia_gpu(montar):
    sup, _ = montar()
    await sup.iniciar_medicao("bench", "gpu")
    try:
        with pytest.raises(ErroSupervisor, match="medição de 'bench'"):
            await sup.subir("tiny")
        with pytest.raises(ErroSupervisor):
            await sup.iniciar_medicao("outra", "gpu")
        with pytest.raises(ErroSupervisor):
            await sup.iniciar_medicao("bench", "cpu")
        await sup.iniciar_medicao("cpu", "cpu")
        sup.encerrar_medicao("cpu")
    finally:
        sup.encerrar_medicao("bench")
    assert sup.medicoes == {}


async def test_medicao_recusa_gpu_no_ar_e_subindo(montar):
    sup, cat = montar(flags=["--atraso", "30"])
    proc = sup._iniciar(cat.obter("tiny"), sup._escolher_porta())
    sup.processos["tiny"] = proc
    try:
        for estado in ("subindo", "no_ar"):
            proc.estado = estado
            with pytest.raises(ErroSupervisor):
                await sup.iniciar_medicao("bench", "gpu")
    finally:
        await sup.desligar()


async def test_edicao_durante_subida_nao_grava_medicao(montar):
    sup, cat = montar(onde="cpu", flags=["--atraso", "2"])
    task = asyncio.create_task(sup.subir("tiny"))
    try:
        while "tiny" not in sup.processos:
            await asyncio.sleep(0.01)
        cat.editar("tiny", {"flags": ["-c", "1024"]})
        r = await task
        assert cat.obter("tiny").medido is None
        assert "o perfil mudou durante a subida; medição não gravada" in r["avisos"]
        with pytest.raises(ErroSupervisor, match="mudou desde que subiu"):
            await sup.subir("tiny")
    finally:
        await sup.desligar()
        await task


async def test_gpu_sem_leitura_nao_grava(montar):
    sup, cat = montar(recursos=RecursosFalsos())
    try:
        r = await sup.subir("tiny")
        assert cat.obter("tiny").medido is None
        assert "não consegui ler a VRAM do processo; medição não gravada" in r["avisos"]
    finally:
        await sup.desligar()


async def test_logs_depois_de_derrubar(montar):
    sup, _ = montar()
    try:
        await sup.subir("tiny")
        await sup.derrubar("tiny")
        assert any("fake llama-server" in l for l in sup.logs("tiny"))
        with pytest.raises(ErroSupervisor, match="logs/outro.log"):
            sup.logs("outro")
    finally:
        await sup.desligar()


async def test_falha_de_subida_fica_registrada_e_some_no_sucesso(montar):
    sup, _ = montar(recursos=RecursosFalsos(livre=0.1))
    with pytest.raises(ErroSupervisor, match="não cabe"):
        await sup.subir("tiny")
    assert "não cabe" in sup.ultimas_falhas["tiny"]["erro"]
    sup.recursos = RecursosFalsos()
    try:
        await sup.subir("tiny")
        assert "tiny" not in sup.ultimas_falhas
    finally:
        await sup.desligar()


def test_diagnosticar_reconhece_arquitetura_e_projetor():
    from local_hub_mcp.supervisor import diagnosticar
    assert "não reconhece a arquitetura 'qwen9'" in diagnosticar(["x", "unknown model architecture: 'qwen9'"])
    assert "projetor" in diagnosticar(["clip_init: unknown projector type: foo"])
    assert diagnosticar(["nada aqui"]) is None


async def test_modelo_novo_demais_vira_mensagem_clara(montar):
    sup, _ = montar(flags=["--arquitetura-desconhecida"])
    with pytest.raises(ErroSupervisor, match="não reconhece a arquitetura 'paddleocr-vl'.*atualize o llama.cpp"):
        await sup.subir("tiny")



async def test_arquivos_em_uso_segue_o_que_subiu_e_nao_o_catalogo(montar, maquina):
    sup, cat = montar()
    try:
        await sup.subir("tiny")
        cat.editar("tiny", {"mmproj": "tiny/mmproj-tiny-F16.gguf"})   # catálogo muda; o processo não
        usados = sup.arquivos_em_uso()
        assert (maquina["models"] / "tiny" / "tiny-Q4_K_M.gguf").resolve() in usados
        assert (maquina["models"] / "tiny" / "mmproj-tiny-F16.gguf").resolve() not in usados
    finally:
        await sup.desligar()
    assert sup.arquivos_em_uso() == {}


async def test_exclusivo_espera_a_trava_do_subir(montar):
    import asyncio
    sup, _ = montar()
    vistos = []
    await sup._trava.acquire()   # como um subir no meio do caminho
    tarefa = asyncio.create_task(sup.exclusivo(lambda em_uso: vistos.append(em_uso) or "feito"))
    await asyncio.sleep(0.2)
    assert not tarefa.done() and vistos == []   # não captura nem apaga enquanto a subida segura a trava
    sup._trava.release()
    assert await tarefa == "feito" and vistos == [{}]


async def test_exclusivo_cancelado_segura_a_trava_ate_a_thread_terminar(montar):
    import asyncio, threading
    sup, _ = montar()
    liberar, comecou = threading.Event(), threading.Event()

    def apagando(em_uso):
        comecou.set()
        liberar.wait(5)
        return "apagou"
    tarefa = asyncio.create_task(sup.exclusivo(apagando))
    while not comecou.is_set():
        await asyncio.sleep(0.01)
    tarefa.cancel()
    await asyncio.sleep(0.2)
    assert sup._trava.locked()          # a thread ainda roda: nenhum subir pode entrar
    liberar.set()
    with pytest.raises(asyncio.CancelledError):
        await tarefa
    assert not sup._trava.locked()
