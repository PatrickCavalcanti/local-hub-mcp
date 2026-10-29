from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from local_hub_mcp.seguranca import MiddlewareChaves, autorizado

CHAVES = {"admin": "adm", "inference": "inf"}


def test_matriz():
    assert autorizado({}, "/mcp", None)
    assert autorizado(CHAVES, "/mcp", "Bearer adm")
    assert not autorizado(CHAVES, "/mcp", "Bearer inf")
    assert not autorizado(CHAVES, "/mcp", None)
    assert autorizado(CHAVES, "/v1/models", "Bearer inf")
    assert autorizado(CHAVES, "/v1/models", "bearer adm")
    assert not autorizado(CHAVES, "/v1/models", "Bearer errada")
    assert autorizado(CHAVES, "/health", None)


def test_middleware_401():
    ok = lambda req: PlainTextResponse("ok")
    app = MiddlewareChaves(Starlette(routes=[Route("/v1/models", ok), Route("/mcp", ok)]), CHAVES)
    c = TestClient(app)
    assert c.get("/v1/models").status_code == 401
    assert c.get("/v1/models", headers={"Authorization": "Bearer inf"}).text == "ok"
    assert c.get("/mcp", headers={"Authorization": "Bearer inf"}).status_code == 401
