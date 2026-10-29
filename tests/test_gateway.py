import base64
import io
import json
import wave

import httpx
import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from local_hub_mcp.gateway import Gateway
from local_hub_mcp.perfis import Catalogo, Perfil, Voz


class SupFalso:
    def __init__(self, portas):
        self.portas = portas
        self.perfis_que_cairam = []

    def porta_de(self, alias):
        return self.portas.get(alias)

    def pronto_em(self, alias):
        return getattr(self, "prontos", {}).get(alias)

    def marcar_caiu(self, alias, motivo):
        self.perfis_que_cairam.append((alias, motivo))


def wav():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\x00\x00" * 800)
    return buf.getvalue()


@pytest.fixture
def cliente(maquina, tmp_path):
    cat = Catalogo(tmp_path / "perfis.json", maquina["models"])
    cat.criar(Perfil(alias="texto", arquivo="tiny/tiny-Q4_K_M.gguf", onde="gpu"))
    cat.criar(Perfil(alias="omni", arquivo="tiny/tiny-Q4_K_M.gguf", onde="gpu", anexos=True,
                     capacidades=["llm.chat", "stt", "tts"], voz=Voz()))
    cat.criar(Perfil(alias="parado", arquivo="tiny/tiny-Q4_K_M.gguf", onde="gpu"))
    recebidos = []

    def upstream(req: httpx.Request):
        corpo = json.loads(req.content or b"{}")
        recebidos.append((str(req.url), corpo))
        if req.url.port == 9999:
            raise httpx.ConnectError("recusado")
        if corpo.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  stream=httpx.ByteStream(b"data: {\"x\": 1}\n\ndata: [DONE]\n\n"))
        return httpx.Response(200, headers={"content-type": "application/json"},
                              stream=httpx.ByteStream(json.dumps({"ok": True, "url": str(req.url)}).encode()))

    gw = Gateway(cat, SupFalso({"texto": 18180, "omni": 18181, "quebrado": 9999}),
                 httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
    with TestClient(Starlette(routes=gw.rotas())) as c:
        c.recebidos = recebidos
        c.catalogo = cat
        yield c


def test_models_lista_com_status(cliente):
    dados = cliente.get("/v1/models").json()["data"]
    estados = {d["id"]: d["status"] for d in dados}
    assert estados == {"atual": "no_ar", "texto": "no_ar", "omni": "no_ar", "parado": "fora_do_ar"}


def test_modelo_por_id(cliente):
    r = cliente.get("/v1/models/omni")
    assert r.status_code == 200
    assert r.json()["id"] == "omni" and r.json()["object"] == "model"


def test_modelo_por_id_desconhecido(cliente):
    r = cliente.get("/v1/models/nada")
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
    assert r.headers["x-should-retry"] == "false"


@pytest.mark.parametrize("metodo", ["GET", "PUT", "DELETE", "PATCH"])
def test_metodo_ou_rota_v1_desconhecidos_tem_erro_openai(cliente, metodo):
    r = cliente.request(metodo, "/v1/nao-existe")
    assert r.status_code == 404
    assert r.json()["error"] == {
        "message": "endpoint /v1/nao-existe não existe neste serviço",
        "type": "invalid_request_error",
        "code": "unknown_endpoint",
    }


def test_post_v1_sem_caminho_tem_erro_openai(cliente):
    r = cliente.post("/v1")
    assert r.json()["error"]["message"] == "endpoint /v1 não existe neste serviço"
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "unknown_endpoint"


def test_modelo_desconhecido_404(cliente):
    r = cliente.post("/v1/chat/completions", json={"model": "nada", "messages": []})
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"


def test_fora_do_ar_409_manda_subir(cliente):
    r = cliente.post("/v1/responses", json={"model": "parado", "input": "oi"})
    assert r.status_code == 409
    assert r.headers["x-should-retry"] == "false"
    assert r.json()["error"]["code"] == "model_not_loaded"
    assert 'subir_modelo("parado")' in r.json()["error"]["message"]


def test_repasse_vai_a_porta_do_perfil(cliente):
    r = cliente.post("/v1/chat/completions", json={"model": "texto", "messages": [{"role": "user", "content": "oi"}]})
    assert r.status_code == 200
    url, corpo = cliente.recebidos[-1]
    assert url == "http://127.0.0.1:18180/v1/chat/completions"
    assert corpo["messages"][0]["content"] == "oi"


def test_repasse_preserva_query_string(cliente):
    r = cliente.post("/v1/chat/completions?foo=bar&limit=3",
                     json={"model": "texto", "messages": []})
    assert r.status_code == 200
    url, _ = cliente.recebidos[-1]
    assert url == "http://127.0.0.1:18180/v1/chat/completions?foo=bar&limit=3"


@pytest.mark.parametrize("path,content_type", [
    ("/v1/chat/completions", "application/json"),
    ("/v1/audio/transcriptions", "multipart/form-data; boundary=x"),
    ("/v1/audio/speech", "application/json"),
])
def test_corpo_maior_que_limite_rejeitado_antes_da_leitura(cliente, path, content_type, monkeypatch):
    async def nao_ler(_self):
        raise AssertionError("o corpo não deve ser lido antes do limite")

    monkeypatch.setattr("starlette.requests.Request.body", nao_ler)
    r = cliente.post(path, content=b"{}", headers={
        "content-length": str(512 * 1024 * 1024 + 1),
        "content-type": content_type,
    })
    assert r.status_code == 413
    assert r.json()["error"] == {
        "message": "pedido maior que 512 MB; envie um arquivo menor",
        "type": "invalid_request_error",
        "code": "payload_too_large",
    }


def test_anexo_de_audio_reescrito(cliente):
    b64 = base64.b64encode(wav()).decode()
    cliente.post("/v1/responses", json={"model": "omni", "input": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}]})
    _, corpo = cliente.recebidos[-1]
    assert corpo["input"][0]["content"][0]["type"] == "input_image"
    assert corpo["instructions"].endswith("Respond with text.")


def test_audio_com_base64_invalido_responde_400_openai(cliente):
    r = cliente.post("/v1/responses", json={"model": "omni", "input": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "!!!nao-e-base64!!!", "format": "wav"}}]}]})
    assert r.status_code == 400
    assert r.json()["error"]["message"] == "o anexo não é base64 válido; reenvie o arquivo"


def test_excecao_inesperada_ao_reescrever_anexo_responde_500_openai(cliente, monkeypatch):
    def falhar(*args, **kwargs):
        raise RuntimeError("falha simulada")

    monkeypatch.setattr("local_hub_mcp.gateway.anexos.reescrever", falhar)
    r = cliente.post("/v1/responses", json={"model": "omni", "input": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav()).decode(), "format": "wav"}}]}]})
    assert r.status_code == 500
    assert r.json()["error"]["message"] == "erro interno ao preparar o pedido: RuntimeError: falha simulada"
    assert r.json()["error"]["type"] == "server_error"


def test_streaming_sse_repassado(cliente):
    r = cliente.post("/v1/chat/completions", json={"model": "texto", "stream": True, "messages": []})
    assert r.headers["content-type"].startswith("text/event-stream")
    assert b"data: [DONE]" in r.content


def test_llama_server_fora_502(maquina, cliente):
    cliente.catalogo.criar(Perfil(alias="quebrado", arquivo="tiny/tiny-Q4_K_M.gguf", onde="gpu"))
    r = cliente.post("/v1/chat/completions", json={"model": "quebrado", "messages": []})
    assert r.status_code == 502
    assert "x-should-retry" not in r.headers


@pytest.mark.parametrize("endpoint", ["/v1/models", "/v1/chat/completions"])
def test_catalogo_invalido_responde_500(endpoint, maquina, tmp_path):
    caminho = tmp_path / "perfis.json"
    catalogo = Catalogo(caminho, maquina["models"])
    catalogo.criar(Perfil(alias="texto", arquivo="tiny/tiny-Q4_K_M.gguf", onde="gpu"))
    dados = json.loads(caminho.read_text(encoding="utf-8"))

    def adicionar_campo_desconhecido(valor):
        if isinstance(valor, dict):
            if "arquivo" in valor and "onde" in valor:
                valor["campo_desconhecido"] = "inválido"
                return True
            return any(adicionar_campo_desconhecido(v) for v in valor.values())
        if isinstance(valor, list):
            return any(adicionar_campo_desconhecido(v) for v in valor)
        return False

    assert adicionar_campo_desconhecido(dados)
    caminho.write_text(json.dumps(dados), encoding="utf-8")
    gw = Gateway(catalogo, SupFalso({"texto": 18180}))
    with TestClient(Starlette(routes=gw.rotas())) as c:
        if endpoint.endswith("completions"):
            r = c.post(endpoint, json={"model": "texto", "messages": []})
        else:
            r = c.get(endpoint)
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "catalog_invalid"
    assert r.json()["error"]["message"].startswith("o catálogo de perfis está inválido:")


def test_tts_sem_capacidade_400(cliente):
    r = cliente.post("/v1/audio/speech", json={"model": "texto", "input": "oi"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "capability_missing"


def test_sem_model_400(cliente):
    assert cliente.post("/v1/chat/completions", json={"messages": []}).status_code == 400


async def test_cliente_que_desconecta_fecha_o_upstream(maquina, tmp_path):
    class UpFalso:
        fechado = False

        async def aiter_raw(self):
            for i in range(100):
                yield f"data: {i}\n\n".encode()

        async def aclose(self):
            self.fechado = True

    gw = Gateway(Catalogo(tmp_path / "p.json", maquina["models"]), SupFalso({}))
    up = UpFalso()
    fluxo = gw._fluxo(up, "x")
    assert await fluxo.__anext__() == b"data: 0\n\n"
    await fluxo.aclose()   # o que acontece quando o cliente some no meio do stream
    assert up.fechado


async def test_stream_interrompido_marca_perfil_como_caido(maquina, tmp_path):
    class StreamQueFalha(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: parcial\n\n"
            raise httpx.ReadError("falha no meio")

        async def aclose(self):
            pass

    supervisor = SupFalso({})
    gw = Gateway(Catalogo(tmp_path / "p.json", maquina["models"]), supervisor)
    resposta = httpx.Response(200, stream=StreamQueFalha())
    blocos = [bloco async for bloco in gw._fluxo(resposta, "x")]

    assert blocos[0] == b"data: parcial\n\n"
    assert b"caiu no meio da resposta" in blocos[1]
    assert supervisor.perfis_que_cairam == [("x", "stream interrompido: falha no meio")]


@pytest.mark.parametrize("audio", [{"format": "wav"}, {"format": "wav", "data": None}, {"format": "wav", "data": 123}])
def test_audio_sem_dados_validos_e_400(cliente, audio):
    r = cliente.post("/v1/responses", json={"model": "omni", "input": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": audio}]}]})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"



def _gateway_coringa(maquina, tmp_path, portas, prontos):
    cat = Catalogo(tmp_path / "coringa.json", maquina["models"])
    cat.criar(Perfil(alias="chat-a", arquivo="tiny/tiny-Q4_K_M.gguf", onde="gpu"))
    cat.criar(Perfil(alias="chat-b", arquivo="tiny/tiny-Q4_K_M.gguf", onde="cpu"))
    cat.criar(Perfil(alias="falante", arquivo="tiny/tiny-Q4_K_M.gguf", onde="cpu", capacidades=["tts", "stt"],
                     voz=Voz()))
    cat.criar(Perfil(alias="vetores", arquivo="tiny/tiny-Q4_K_M.gguf", onde="cpu", capacidades=["embedding"]))
    urls = []

    def upstream(req):
        urls.append(str(req.url))
        return httpx.Response(200, stream=httpx.ByteStream(b'{"ok": true}'), headers={"content-type": "application/json"})

    sup = SupFalso(portas)
    sup.prontos = prontos
    gw = Gateway(cat, sup, httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
    return TestClient(Starlette(routes=gw.rotas())), urls


def test_coringa_atual_vai_ao_ultimo_chat_que_subiu(maquina, tmp_path):
    c, urls = _gateway_coringa(maquina, tmp_path, {"chat-a": 18180, "chat-b": 18181, "falante": 18182},
                               {"chat-a": 100.0, "chat-b": 200.0, "falante": 300.0})
    r = c.post("/v1/responses", json={"model": "atual", "input": "oi"})
    assert r.status_code == 200 and urls[-1] == "http://127.0.0.1:18181/v1/responses"
    c.post("/v1/chat/completions", json={"model": "atual", "messages": []})
    assert urls[-1] == "http://127.0.0.1:18181/v1/chat/completions"


def test_coringa_atual_por_capacidade_do_endpoint(maquina, tmp_path):
    c, urls = _gateway_coringa(maquina, tmp_path, {"chat-a": 18180, "vetores": 18183}, {"chat-a": 1.0, "vetores": 2.0})
    c.post("/v1/embeddings", json={"model": "atual", "input": "x"})
    assert urls[-1] == "http://127.0.0.1:18183/v1/embeddings"
    c.post("/v1/responses", json={"model": "atual", "input": "oi"})
    assert urls[-1] == "http://127.0.0.1:18180/v1/responses"


def test_coringa_atual_sem_modelo_no_ar_409(maquina, tmp_path):
    c, _ = _gateway_coringa(maquina, tmp_path, {"falante": 18182}, {"falante": 1.0})
    r = c.post("/v1/responses", json={"model": "atual", "input": "oi"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "model_not_loaded"
    assert "subir_modelo" in r.json()["error"]["message"]


def test_coringa_atual_na_fala_procura_tts(maquina, tmp_path, monkeypatch):
    from local_hub_mcp.adaptador import tts
    chamadas = []
    monkeypatch.setattr(tts, "falar", lambda dados, url, voz: chamadas.append(url) or (200, "audio/wav", b"RIFF"))
    c, _ = _gateway_coringa(maquina, tmp_path, {"chat-a": 18180, "falante": 18182}, {"chat-a": 9.0, "falante": 1.0})
    r = c.post("/v1/audio/speech", json={"model": "atual", "input": "oi"})
    assert r.status_code == 200 and chamadas == ["http://127.0.0.1:18182"]


def test_models_lista_o_coringa_com_o_alvo(maquina, tmp_path):
    c, _ = _gateway_coringa(maquina, tmp_path, {"chat-a": 18180, "chat-b": 18181}, {"chat-a": 5.0, "chat-b": 1.0})
    dados = c.get("/v1/models").json()["data"]
    assert dados[0]["id"] == "atual" and dados[0]["aponta_para"] == "chat-a" and dados[0]["status"] == "no_ar"
    assert c.get("/v1/models/atual").json()["aponta_para"] == "chat-a"
    vazio, _ = _gateway_coringa(maquina, tmp_path / "v", {}, {})
    assert vazio.get("/v1/models").json()["data"][0] == {**vazio.get("/v1/models").json()["data"][0],
                                                          "aponta_para": None, "status": "fora_do_ar"}
