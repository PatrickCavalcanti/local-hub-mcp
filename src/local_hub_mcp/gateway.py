"""Gateway OpenAI /v1: roteia pelo campo 'model' (alias do perfil) ao llama-server do perfil."""
from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .adaptador import ErroApi, anexos, stt, tts
from .perfis import ALIAS_CORINGA, Catalogo, ErroPerfil, Perfil

REPASSAR_CAB = {"content-type", "cache-control"}
LIMITE_CORPO = 512 * 1024 * 1024
CAPS_CHAT = ("llm.chat", "text.generate", "llm.fast")


def _caps_do_caminho(caminho: str) -> tuple[str, ...]:
    """Capacidade que o coringa procura em cada endpoint repassado."""
    return ("embedding",) if caminho.startswith("embeddings") else CAPS_CHAT


def erro(status: int, msg: str, tipo: str = "invalid_request_error", code: str | None = None) -> JSONResponse:
    headers = {"x-should-retry": "false"} if status < 500 else None
    return JSONResponse({"error": {"message": msg, "type": tipo, "code": code}}, status_code=status,
                        headers=headers)


def _corpo_acima_do_limite(req: Request) -> bool:
    try:
        return int(req.headers.get("content-length", "0")) > LIMITE_CORPO
    except (TypeError, ValueError):
        return False


class Gateway:
    def __init__(self, catalogo: Catalogo, supervisor, cliente: httpx.AsyncClient | None = None):
        self.catalogo, self.supervisor = catalogo, supervisor
        self.cliente = cliente or httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0))

    def rotas(self) -> list[Route]:
        return [
            Route("/v1/models", self.modelos, methods=["GET"]),
            Route("/v1/models/{id}", self.modelo, methods=["GET"]),
            Route("/v1/audio/transcriptions", self.transcricao, methods=["POST"]),
            Route("/v1/audio/speech", self.fala, methods=["POST"]),
            Route("/v1", self.desconhecido, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
            Route("/v1/{caminho:path}", self.desconhecido, methods=["GET", "PUT", "DELETE", "PATCH"]),
            Route("/v1/{caminho:path}", self.repassar, methods=["POST"]),
        ]

    def _listar_catalogo(self) -> list[Perfil] | JSONResponse:
        try:
            return self.catalogo.listar()
        except (ErroPerfil, json.JSONDecodeError, OSError) as e:
            return erro(500, f"o catálogo de perfis está inválido: {e}", "server_error", code="catalog_invalid")

    def _coringa(self, perfis: list[Perfil], caps: tuple[str, ...]) -> Perfil | None:
        """O perfil no ar com uma das capacidades; havendo vários, o último que subiu."""
        no_ar = [p for p in perfis if set(caps) & set(p.capacidades) and self.supervisor.porta_de(p.alias)]
        return max(no_ar, key=lambda p: self.supervisor.pronto_em(p.alias) or 0.0, default=None)

    def _resolver(self, modelo, capacidade: str | None = None,
                  caps_coringa: tuple[str, ...] = CAPS_CHAT) -> tuple[Perfil, int] | JSONResponse:
        if not modelo or not isinstance(modelo, str):
            return erro(400, "faltou o campo 'model' (use o alias de um perfil; veja GET /v1/models)")
        perfis = self._listar_catalogo()
        if isinstance(perfis, JSONResponse):
            return perfis
        if modelo == ALIAS_CORINGA:
            escolhido = self._coringa(perfis, caps_coringa)
            if escolhido is None:
                return erro(409, f"nenhum modelo com {' / '.join(caps_coringa)} está no ar para '{ALIAS_CORINGA}'; "
                                 f"chame a ferramenta subir_modelo no MCP do local-hub-mcp", code="model_not_loaded")
            modelo = escolhido.alias
        perfil = next((p for p in perfis if p.alias == modelo), None)
        if perfil is None:
            return erro(404, f"o modelo '{modelo}' não existe; veja GET /v1/models", code="model_not_found")
        if capacidade and capacidade not in perfil.capacidades:
            return erro(400, f"o perfil '{modelo}' não tem a capacidade '{capacidade}'", code="capability_missing")
        porta = self.supervisor.porta_de(modelo)
        if porta is None:
            return erro(409, f"o perfil \"{modelo}\" não está no ar; chame a ferramenta subir_modelo(\"{modelo}\") "
                             f"no MCP do local-hub-mcp", code="model_not_loaded")
        return perfil, porta

    async def modelos(self, req: Request) -> JSONResponse:
        agora = int(time.time())
        perfis = self._listar_catalogo()
        if isinstance(perfis, JSONResponse):
            return perfis
        dados = [self._objeto_coringa(perfis, agora)]
        dados += [{"id": p.alias, "object": "model", "created": agora, "owned_by": "local-hub-mcp",
                   "status": "no_ar" if self.supervisor.porta_de(p.alias) else "fora_do_ar",
                   "capacidades": p.capacidades} for p in perfis]
        return JSONResponse({"object": "list", "data": dados})

    def _objeto_coringa(self, perfis: list[Perfil], agora: int) -> dict:
        alvo = self._coringa(perfis, CAPS_CHAT)
        return {"id": ALIAS_CORINGA, "object": "model", "created": agora, "owned_by": "local-hub-mcp",
                "status": "no_ar" if alvo else "fora_do_ar", "aponta_para": alvo.alias if alvo else None,
                "capacidades": alvo.capacidades if alvo else []}

    async def modelo(self, req: Request) -> JSONResponse:
        alias = req.path_params["id"]
        perfis = self._listar_catalogo()
        if isinstance(perfis, JSONResponse):
            return perfis
        if alias == ALIAS_CORINGA:
            return JSONResponse(self._objeto_coringa(perfis, int(time.time())))
        perfil = next((p for p in perfis if p.alias == alias), None)
        if perfil is None:
            return erro(404, f"o modelo '{alias}' não existe; veja GET /v1/models", code="model_not_found")
        return JSONResponse({"id": perfil.alias, "object": "model", "created": int(time.time()),
                             "owned_by": "local-hub-mcp",
                             "status": "no_ar" if self.supervisor.porta_de(perfil.alias) else "fora_do_ar",
                             "capacidades": perfil.capacidades})

    async def desconhecido(self, req: Request) -> JSONResponse:
        caminho = req.path_params.get("caminho", "")
        return erro(404, f"endpoint /v1{'/' + caminho if caminho else ''} não existe neste serviço", code="unknown_endpoint")

    async def _fluxo(self, up: httpx.Response, alias: str):
        try:
            async for bloco in up.aiter_raw():
                yield bloco
        except httpx.HTTPError as e:
            self.supervisor.marcar_caiu(alias, f"stream interrompido: {e}")
            msg = {"error": {"message": f"o llama-server de '{alias}' caiu no meio da resposta: {e}",
                             "type": "server_error"}}
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n".encode()
        finally:
            # cliente que desconecta não roda o BackgroundTask: fechar aqui faz o llama-server parar de gerar
            await up.aclose()

    async def repassar(self, req: Request) -> Response:
        try:
            if _corpo_acima_do_limite(req):
                return erro(413, "pedido maior que 512 MB; envie um arquivo menor", code="payload_too_large")
            caminho = req.path_params.get("caminho", "")
            if not caminho:
                return await self.desconhecido(req)
            corpo = await req.body()
            try:
                dados = json.loads(corpo or b"{}")
            except json.JSONDecodeError:
                return erro(400, "corpo JSON inválido")
            r = self._resolver(dados.get("model") if isinstance(dados, dict) else None,
                              caps_coringa=_caps_do_caminho(caminho))
            if isinstance(r, JSONResponse):
                return r
            perfil, porta = r
            if perfil.anexos and caminho in ("responses", "chat/completions"):
                try:
                    novo = await asyncio.to_thread(anexos.reescrever, corpo,
                                                   "responses" if caminho == "responses" else "chat")
                except ErroApi as e:
                    return erro(e.status, str(e), e.tipo)
                corpo = novo or corpo
            url = f"http://127.0.0.1:{porta}/v1/{caminho}"
            if req.url.query:
                url += f"?{req.url.query}"
            pedido = self.cliente.build_request("POST", url, content=corpo,
                                                headers={"Content-Type": "application/json"})
            try:
                up = await self.cliente.send(pedido, stream=True)
            except httpx.HTTPError as e:
                return erro(502, f"o llama-server de '{perfil.alias}' não respondeu: {e}", "server_error")
            cab = {k: v for k, v in up.headers.items() if k.lower() in REPASSAR_CAB}
            return StreamingResponse(self._fluxo(up, perfil.alias), status_code=up.status_code, headers=cab,
                                     background=BackgroundTask(up.aclose))
        except Exception as e:
            logging.getLogger(__name__).exception("erro interno ao preparar o pedido /v1")
            return erro(500, f"erro interno ao preparar o pedido: {type(e).__name__}: {e}", "server_error")

    async def transcricao(self, req: Request) -> Response:
        try:
            if _corpo_acima_do_limite(req):
                return erro(413, "pedido maior que 512 MB; envie um arquivo menor", code="payload_too_large")
            corpo = await req.body()
            try:
                campos, arquivo = stt.campos_multipart(req.headers.get("content-type", ""), corpo)
            except ErroApi as e:
                return erro(e.status, str(e), e.tipo)
            r = self._resolver(campos.get("model"), "stt", caps_coringa=("stt",))
            if isinstance(r, JSONResponse):
                return r
            perfil, porta = r
            try:
                status, mime, dados = await asyncio.to_thread(stt.transcrever, campos, arquivo,
                                                              f"http://127.0.0.1:{porta}", perfil.voz)
            except ErroApi as e:
                return erro(e.status, str(e), e.tipo)
            return Response(dados, status_code=status, media_type=mime)
        except Exception as e:
            logging.getLogger(__name__).exception("erro interno ao preparar o pedido /v1/audio/transcriptions")
            return erro(500, f"erro interno ao preparar o pedido: {type(e).__name__}: {e}", "server_error")

    async def fala(self, req: Request) -> Response:
        try:
            if _corpo_acima_do_limite(req):
                return erro(413, "pedido maior que 512 MB; envie um arquivo menor", code="payload_too_large")
            try:
                dados = json.loads(await req.body() or b"{}")
            except json.JSONDecodeError:
                return erro(400, "corpo JSON inválido")
            r = self._resolver(dados.get("model"), "tts", caps_coringa=("tts",))
            if isinstance(r, JSONResponse):
                return r
            perfil, porta = r
            try:
                status, mime, audio = await asyncio.to_thread(tts.falar, dados, f"http://127.0.0.1:{porta}", perfil.voz)
            except ErroApi as e:
                return erro(e.status, str(e), e.tipo)
            return Response(audio, status_code=status, media_type=mime)
        except Exception as e:
            logging.getLogger(__name__).exception("erro interno ao preparar o pedido /v1/audio/speech")
            return erro(500, f"erro interno ao preparar o pedido: {type(e).__name__}: {e}", "server_error")
