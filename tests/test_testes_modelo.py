import json
import zlib

import httpx
import pytest

from local_hub_mcp import testes_modelo as modulo_testes_modelo
from local_hub_mcp.perfis import Perfil, Voz
from local_hub_mcp.testes_modelo import png_vermelho, testar as executar_testes


def test_png_valido():
    png = png_vermelho(8)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and b"IHDR" in png and png.endswith(b"IEND\xaeB`\x82")


async def test_texto_e_visao_com_llama_falso():
    def responder(req):
        corpo = json.loads(req.content)
        conteudo = corpo["messages"][0]["content"]
        resposta = "Vermelho" if isinstance(conteudo, list) else "pronto"
        return httpx.Response(200, json={"choices": [{"message": {"content": resposta}}],
                                         "timings": {"predicted_per_second": 30.0}})

    perfil = Perfil(alias="m", arquivo="m.gguf", onde="gpu", capacidades=["llm.chat", "llm.vision"])
    cli = httpx.AsyncClient(transport=httpx.MockTransport(responder))
    r = await executar_testes(perfil, 18180, cliente=cli)
    assert r["ok"] is True
    assert [t["capacidade"] for t in r["testes"]] == ["texto", "visao"]
    assert "30.0 t/s" in r["testes"][0]["detalhe"]


async def test_visao_errada_falha():
    cli = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"choices": [{"message": {"content": "azul"}}]})))
    perfil = Perfil(alias="m", arquivo="m.gguf", onde="gpu", capacidades=["llm.vision"])
    r = await executar_testes(perfil, 18180, cliente=cli)
    assert r["ok"] is False


@pytest.mark.parametrize(("conteudo", "raciocinio", "esperado"), [
    ("verde", "vermelho, verde, azul ou amarelo", False),
    ("Vermelho.", "green", True),
])
async def test_visao_julga_somente_content_e_pede_512_tokens(conteudo, raciocinio, esperado):
    max_tokens = []

    def responder(req):
        max_tokens.append(json.loads(req.content)["max_tokens"])
        return httpx.Response(200, json={"choices": [{"message": {
            "content": conteudo, "reasoning_content": raciocinio}}]})

    perfil = Perfil(alias="m", arquivo="m.gguf", onde="gpu", capacidades=["llm.vision"])
    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as cli:
        r = await executar_testes(perfil, 18180, cliente=cli)
    assert r["testes"][0]["ok"] is esperado
    assert max_tokens == [512]


async def test_erro_http_em_texto_vira_resultado_falho():
    perfil = Perfil(alias="m", arquivo="m.gguf", onde="gpu", capacidades=["llm.chat"])
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(500, text="erro do servidor"))) as cli:
        r = await executar_testes(perfil, 18180, cliente=cli)
    item = r["testes"][0]
    assert item["capacidade"] == "texto"
    assert item["ok"] is False
    assert item["segundos"] >= 0
    assert item["detalhe"].startswith("falha HTTP:")


async def test_erro_http_em_visao_vira_resultado_falho():
    perfil = Perfil(alias="m", arquivo="m.gguf", onde="gpu", capacidades=["llm.vision"])
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(500, text="erro do servidor"))) as cli:
        r = await executar_testes(perfil, 18180, cliente=cli)
    item = r["testes"][0]
    assert item["capacidade"] == "visao"
    assert item["ok"] is False
    assert item["detalhe"].startswith("falha HTTP:")


async def test_tts_wav_invalido_vira_resultado_falho(monkeypatch):
    monkeypatch.setattr(modulo_testes_modelo.tts, "falar", lambda *args: (200, "audio/wav", b"lixo"))
    perfil = Perfil(alias="m", arquivo="m.gguf", onde="cpu", capacidades=["tts"], voz=Voz())
    r = await executar_testes(perfil, 18180)
    item = r["testes"][0]
    assert item["capacidade"] == "tts"
    assert item["ok"] is False
    assert item["segundos"] >= 0
    assert item["detalhe"].startswith("falha WAV:")


async def test_tts_erro_http_vira_resultado_falho(monkeypatch):
    def falhar(*args):
        raise httpx.ConnectError("servidor de voz indisponível")

    monkeypatch.setattr(modulo_testes_modelo.tts, "falar", falhar)
    perfil = Perfil(alias="m", arquivo="m.gguf", onde="cpu", capacidades=["tts"], voz=Voz())
    r = await executar_testes(perfil, 18180)
    item = r["testes"][0]
    assert item["capacidade"] == "tts" and item["ok"] is False
    assert item["detalhe"].startswith("falha HTTP:")
    assert "servidor de voz indisponível" in item["detalhe"]


async def test_stt_erro_http_vira_resultado_falho(monkeypatch):
    def falhar(*args):
        raise httpx.ConnectError("servidor de voz indisponível")

    monkeypatch.setattr(modulo_testes_modelo.stt, "transcrever", falhar)
    perfil = Perfil(alias="m", arquivo="m.gguf", onde="cpu", voz=Voz())
    item = await modulo_testes_modelo._stt("http://localhost", perfil, b"wav")
    assert item["capacidade"] == "stt" and item["ok"] is False
    assert item["detalhe"]


@pytest.mark.parametrize("resposta", [b"{", b"{}"])
async def test_stt_json_invalido_ou_sem_texto_vira_resultado_falho(monkeypatch, resposta):
    monkeypatch.setattr(modulo_testes_modelo.stt, "transcrever",
                        lambda *args: (200, "application/json", resposta))
    perfil = Perfil(alias="m", arquivo="m.gguf", onde="cpu", voz=Voz())
    item = await modulo_testes_modelo._stt("http://localhost", perfil, b"wav")
    assert item["capacidade"] == "stt" and item["ok"] is False
    assert item["detalhe"]


async def test_stt_sem_tts_e_pulado():
    perfil = Perfil(alias="m", arquivo="m.gguf", onde="cpu", capacidades=["stt"])
    r = await executar_testes(perfil, 18180, cliente=httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(500))))
    assert r["testes"][0]["ok"] is None and r["ok"] is True


def test_duracao_wav_com_tamanho_desconhecido_no_cabecalho():
    import io, struct, wave
    from local_hub_mcp.testes_modelo import duracao_wav
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b"\x00\x00" * 24000)
    dados = bytearray(buf.getvalue())
    dados[4:8] = struct.pack("<I", 0xFFFFFFFF)     # como o ffmpeg grava num pipe
    dados[40:44] = struct.pack("<I", 0xFFFFFFFF)
    assert abs(duracao_wav(bytes(dados)) - 1.0) < 0.01
