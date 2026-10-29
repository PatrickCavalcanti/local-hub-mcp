"""Monta o app ASGI: /v1 (gateway), /health e /mcp na mesma porta, com chaves por endpoint."""
from __future__ import annotations

import asyncio
import contextlib
import logging

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from .gateway import Gateway
from .mcp_server import Contexto, criar_mcp
from .seguranca import MiddlewareChaves


def criar_app(ctx: Contexto):
    mcp = criar_mcp(ctx)
    mcp_app = mcp.streamable_http_app(streamable_http_path="/mcp")
    gateway = Gateway(ctx.catalogo, ctx.supervisor)

    async def saude(req):
        return JSONResponse({"status": "ok"})

    async def subir_boot():
        for alias in ctx.cfg.subir_no_boot:
            try:
                resultado = await ctx.supervisor.subir(alias)
                logging.getLogger(__name__).info("subir_no_boot %s: %s", alias, resultado)
            except Exception:
                logging.getLogger(__name__).exception("subir_no_boot %s falhou", alias)

    @contextlib.asynccontextmanager
    async def ciclo(app):
        vigia = asyncio.create_task(ctx.supervisor.vigiar())
        async with mcp.session_manager.run():
            boot = asyncio.create_task(subir_boot())
            try:
                yield
            finally:
                boot.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await boot
                vigia.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await vigia
                await ctx.supervisor.desligar()
                await gateway.cliente.aclose()

    app = Starlette(routes=[*gateway.rotas(), Route("/health", saude), Mount("/", app=mcp_app)], lifespan=ciclo)
    return MiddlewareChaves(app, ctx.cfg.chaves)
