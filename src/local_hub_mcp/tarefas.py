"""Operações longas (baixar, medir, subir): respondem em até espera_s; senão seguem em segundo plano.
O cliente MCP do Igris corta a chamada em 120 s (call_timeout_ms), por isso a espera padrão é 90 s."""
from __future__ import annotations

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass


@dataclass
class Tarefa:
    id: str
    nome: str
    inicio: float
    task: asyncio.Task
    concluida: float | None = None
    progresso: object = None   # função sem argumentos que devolve um dict (ex.: bytes baixados)


class Tarefas:
    def __init__(self) -> None:
        self._t: dict[str, Tarefa] = {}
        self._seq = itertools.count(1)

    def _podar(self) -> None:
        agora = time.time()
        antigas = [tid for tid, t in self._t.items()
                   if t.concluida is not None and agora - t.concluida > 3600]
        for tid in antigas:
            del self._t[tid]

    def _ler_progresso(self, t: Tarefa) -> dict | None:
        if t.progresso is None or t.task.done():
            return None
        try:
            return t.progresso()
        except Exception as e:   # progresso é informativo: nunca derruba a consulta
            return {"erro": f"não consegui medir o progresso: {e}"}

    async def rodar(self, nome: str, coro, espera_s: float, progresso=None) -> dict:
        self._podar()
        tid = f"t{next(self._seq)}"
        t = Tarefa(tid, nome, time.time(), asyncio.create_task(coro), progresso=progresso)
        self._t[tid] = t

        def ao_terminar(task: asyncio.Task) -> None:
            t.concluida = time.time()
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                logging.getLogger(__name__).error("tarefa %s (%s) falhou", t.id, t.nome, exc_info=exc)

        t.task.add_done_callback(ao_terminar)
        await asyncio.wait({t.task}, timeout=espera_s)   # timeout não cancela a tarefa
        if t.task.done():
            return self._resultado(t)
        r = {"ok": True, "estado": "em_andamento", "tarefa": tid, "nome": nome,
             "mensagem": f"segue em segundo plano; consulte com tarefa('{tid}')"}
        prog = self._ler_progresso(t)
        if prog is not None:
            r["progresso"] = prog
        return r

    def consultar(self, tid: str) -> dict:
        t = self._t.get(tid)
        if not t:
            return {"ok": False, "estado": "erro", "erro": f"tarefa {tid!r} não existe"}
        if not t.task.done():
            r = {"ok": True, "estado": "em_andamento", "tarefa": tid, "nome": t.nome,
                 "decorrido_s": round(time.time() - t.inicio)}
            prog = self._ler_progresso(t)
            if prog is not None:
                r["progresso"] = prog
            return r
        return self._resultado(t)

    def _resultado(self, t: Tarefa) -> dict:
        if t.task.cancelled():
            return {"ok": False, "estado": "cancelada", "tarefa": t.id, "nome": t.nome}
        exc = t.task.exception()
        if exc is not None:
            return {"ok": False, "estado": "erro", "tarefa": t.id, "nome": t.nome, "erro": str(exc)}
        return {"ok": True, "estado": "ok", "tarefa": t.id, "nome": t.nome, "resultado": t.task.result()}

    def listar(self) -> list[dict]:
        self._podar()
        out = []
        for t in self._t.values():
            if not t.task.done():
                estado = "em_andamento"
            elif t.task.cancelled():
                estado = "cancelada"
            else:
                estado = "erro" if t.task.exception() else "ok"
            item = {"tarefa": t.id, "nome": t.nome, "estado": estado}
            prog = self._ler_progresso(t)
            if prog is not None:
                item["progresso"] = prog
            if estado == "cancelada":
                item["ok"] = False
            out.append(item)
        return out
