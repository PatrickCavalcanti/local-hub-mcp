"""Chaves por endpoint: /mcp exige 'admin'; /v1 aceita 'inference' ou 'admin'. Sem chaves: liberado
(a config só permite isso com bind local)."""
from __future__ import annotations

import hmac

from starlette.responses import JSONResponse


def _confere(token: str, chave: str | None) -> bool:
    return bool(chave) and hmac.compare_digest(token.encode(), chave.encode())


def autorizado(chaves: dict[str, str], caminho: str, cabecalho: str | None) -> bool:
    if not chaves:
        return True
    token = cabecalho[7:].strip() if cabecalho and cabecalho.lower().startswith("bearer ") else ""
    if caminho.startswith("/mcp"):
        return _confere(token, chaves.get("admin"))
    if caminho.startswith("/v1"):
        return _confere(token, chaves.get("admin")) or _confere(token, chaves.get("inference"))
    return True


class MiddlewareChaves:
    def __init__(self, app, chaves: dict[str, str]):
        self.app, self.chaves = app, chaves

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            cab = dict(scope.get("headers") or []).get(b"authorization", b"").decode("latin-1") or None
            if not autorizado(self.chaves, scope.get("path", ""), cab):
                resposta = JSONResponse({"error": {"message": "chave inválida ou ausente (Authorization: Bearer <chave>)",
                                                   "type": "invalid_api_key", "code": "invalid_api_key"}},
                                        status_code=401)
                await resposta(scope, receive, send)
                return
        await self.app(scope, receive, send)
