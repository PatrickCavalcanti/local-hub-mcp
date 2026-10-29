import asyncio

import pytest

from local_hub_mcp import tarefas as modulo_tarefas
from local_hub_mcp.tarefas import Tarefas


async def test_rapida_devolve_resultado():
    async def rapida():
        return {"ok": True}
    r = await Tarefas().rodar("x", rapida(), espera_s=1)
    assert r["estado"] == "ok" and r["resultado"] == {"ok": True}


async def test_lenta_vira_tarefa_e_termina():
    t = Tarefas()

    async def lenta():
        await asyncio.sleep(0.3)
        return 42
    r = await t.rodar("lenta", lenta(), espera_s=0.05)
    assert r["estado"] == "em_andamento"
    assert t.consultar(r["tarefa"])["estado"] == "em_andamento"
    await asyncio.sleep(0.4)
    fim = t.consultar(r["tarefa"])
    assert fim["estado"] == "ok" and fim["resultado"] == 42


async def test_erro_vira_estado_erro_e_vai_para_log(caplog):
    async def falha():
        raise ValueError("deu ruim")
    r = await Tarefas().rodar("f", falha(), espera_s=1)
    assert r["estado"] == "erro" and "deu ruim" in r["erro"]
    registro = next(x for x in caplog.records if x.getMessage() == "tarefa t1 (f) falhou")
    assert registro.exc_info[1].args == ("deu ruim",)


async def test_tarefa_cancelada_consultar_e_listar():
    t = Tarefas()

    async def longa():
        await asyncio.Event().wait()

    r = await t.rodar("longa", longa(), espera_s=0)
    tarefa = t._t[r["tarefa"]].task
    tarefa.cancel()
    await asyncio.gather(tarefa, return_exceptions=True)

    resultado = t.consultar(r["tarefa"])
    assert resultado["ok"] is False
    assert resultado["estado"] == "cancelada"
    listada = next(item for item in t.listar() if item["tarefa"] == r["tarefa"])
    assert listada["ok"] is False
    assert listada["estado"] == "cancelada"


def test_tarefa_inexistente():
    assert Tarefas().consultar("t99")["estado"] == "erro"


async def test_listar_remove_tarefa_concluida_ha_mais_de_uma_hora(monkeypatch):
    agora = [1000.0]
    monkeypatch.setattr(modulo_tarefas.time, "time", lambda: agora[0])
    t = Tarefas()

    async def rapida():
        return 1

    await t.rodar("antiga", rapida(), espera_s=1)
    await asyncio.sleep(0)
    agora[0] += 3601

    assert t.listar() == []


async def test_rodar_remove_tarefa_concluida_ha_mais_de_uma_hora(monkeypatch):
    agora = [1000.0]
    monkeypatch.setattr(modulo_tarefas.time, "time", lambda: agora[0])
    t = Tarefas()

    async def rapida():
        return 1

    await t.rodar("antiga", rapida(), espera_s=1)
    await asyncio.sleep(0)
    antiga = next(iter(t._t))
    agora[0] += 3601
    await t.rodar("nova", rapida(), espera_s=1)

    assert antiga not in t._t


async def test_tarefa_em_andamento_mostra_progresso():
    import asyncio
    from local_hub_mcp.tarefas import Tarefas
    t = Tarefas()
    feito = asyncio.Event()

    async def demorada():
        await feito.wait()
        return "fim"
    r = await t.rodar("baixar x", demorada(), 0.01, progresso=lambda: {"percentual": 42.0})
    assert r["estado"] == "em_andamento" and r["progresso"] == {"percentual": 42.0}
    assert t.consultar(r["tarefa"])["progresso"] == {"percentual": 42.0}
    assert t.listar()[0]["progresso"] == {"percentual": 42.0}
    feito.set()
    await asyncio.sleep(0.01)
    assert "progresso" not in t.consultar(r["tarefa"])
