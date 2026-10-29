import base64
import io
import json
import re
import struct
import wave
from types import SimpleNamespace

import httpx
import pytest

from local_hub_mcp.adaptador import ErroApi, anexos, ffmpeg, stt, tts
from local_hub_mcp.adaptador.numeros_pt import por_extenso

VOZ = SimpleNamespace(vozes={"female": {}, "male": {"seed": 42, "tom": -1.5, "velocidade": 1.1}},
                      idioma="pt", pedaco_tts=250, pedaco_stt_s=45.0, numeros_por_extenso=True)


def wav_silencio(seg=0.5, sr=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(b"\x00\x00" * int(seg * sr))
    return buf.getvalue()


@pytest.fixture
def com_ffmpeg():
    caminho = ffmpeg.achar(None)
    if not caminho:
        pytest.skip("ffmpeg não instalado")
    ffmpeg.configurar(caminho)
    yield
    ffmpeg.configurar(None)


# ---- anexos
def test_sem_audio_nao_mexe():
    corpo = json.dumps({"input": [{"role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]}]}).encode()
    assert anexos.reescrever(corpo, "responses") is None


def test_audio_wav_vira_input_image_e_ganha_frase():
    b64 = base64.b64encode(wav_silencio()).decode()
    corpo = json.dumps({"input": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}]}).encode()
    novo = json.loads(anexos.reescrever(corpo, "responses"))
    parte = novo["input"][0]["content"][0]
    assert parte["type"] == "input_image" and parte["image_url"].startswith("data:audio/wav;base64,")
    assert novo["instructions"].endswith(anexos.FRASE_TEXTO)


def test_chat_com_system_recebe_sufixo_no_fim():
    b64 = base64.b64encode(wav_silencio()).decode()
    corpo = json.dumps({"messages": [
        {"role": "system", "content": "Você é útil."},
        {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}]}).encode()
    novo = json.loads(anexos.reescrever(corpo, "chat"))
    assert novo["messages"][0]["content"].endswith(anexos.FRASE_TEXTO)
    assert novo["messages"][1]["content"][0]["type"] == "image_url"


def test_video_mp4_ganha_frase_de_video():
    corpo = json.dumps({"input": [{"role": "user", "content": [
        {"type": "input_file", "filename": "v.mp4", "file_data": "data:video/mp4;base64,AAAA"}]}]}).encode()
    novo = json.loads(anexos.reescrever(corpo, "responses"))
    assert anexos.FRASE_VIDEO in novo["instructions"]


def test_sem_ffmpeg_audio_webm_da_503():
    ffmpeg.configurar(None)
    b64 = base64.b64encode(b"\x1aE\xdf\xa3 nao-e-wav").decode()
    corpo = json.dumps({"input": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": b64, "format": "webm"}}]}]}).encode()
    with pytest.raises(ErroApi) as e:
        anexos.reescrever(corpo, "responses")
    assert e.value.status == 503


# ---- stt
def test_cortes_nas_pausas():
    assert stt.cortes(100, [20, 40, 60, 80], 45, 15) == [0.0, 40, 80, 100]
    assert stt.cortes(100, [], 45, 15) == [0.0, 45.0, 90.0, 100]
    assert stt.cortes(30, [10], 45, 15) == [0.0, 30]


def test_tempo():
    assert stt.tempo(3661.5, ",") == "01:01:01,500"


def test_campos_multipart():
    fronteira = "XyZ"
    corpo = (f"--{fronteira}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nvoz\r\n"
             f"--{fronteira}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
             f"Content-Type: audio/wav\r\n\r\n").encode() + b"RIFF1234" + f"\r\n--{fronteira}--\r\n".encode()
    campos, (nome, dados) = stt.campos_multipart(f"multipart/form-data; boundary={fronteira}", corpo)
    assert campos == {"model": "voz"} and nome == "a.wav" and dados == b"RIFF1234"
    with pytest.raises(ErroApi):
        stt.campos_multipart("application/json", b"{}")


@pytest.mark.ffmpeg
def test_transcrever_com_llama_falso(com_ffmpeg):
    pedidos = []

    def responder(req):
        pedidos.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "olá mundo"}}]})

    cli = httpx.Client(transport=httpx.MockTransport(responder))
    status, mime, dados = stt.transcrever({"response_format": "json"}, ("a.wav", wav_silencio(1.0)),
                                          "http://x", VOZ, cliente=cli)
    assert status == 200 and json.loads(dados)["text"] == "olá mundo"
    assert pedidos[0]["messages"][0]["content"] == "Perform ASR in Portuguese."


# ---- tts
def test_pedacos_respeitam_limite():
    texto = "Frase curta. " * 60
    ps = tts.pedacos(texto, 250)
    assert len(ps) > 1 and all(len(p) <= 260 for p in ps)
    assert tts.pedacos("Uma frase só.") == ["Uma frase só."]


def test_filtros():
    assert tts.filtros(0, 1.0) == []
    f = tts.filtros(-1.5, 1.1)
    assert f[0] == "-af" and "asetrate" in f[1] and "atempo" in f[1]


def test_numeros_por_extenso():
    assert not re.search(r"\d", por_extenso("Tenho 3 gatos"))


@pytest.mark.ffmpeg
def test_falar_voz_desconhecida_cai_na_female(com_ffmpeg):
    pedidos = []

    def responder(req):
        pedidos.append(json.loads(req.content))
        return httpx.Response(200, content=struct.pack("<f", 0.0) * 2400)

    cli = httpx.Client(transport=httpx.MockTransport(responder))
    status, mime, dados = tts.falar({"input": "Olá.", "voice": "alloy", "response_format": "wav"},
                                    "http://x", VOZ, cliente=cli)
    assert status == 200 and mime == "audio/wav" and dados[:4] == b"RIFF"
    assert pedidos[0]["voice"] == "female" and pedidos[0]["response_format"] == "pcm"


@pytest.mark.ffmpeg
@pytest.mark.parametrize("ligado, esperado", [(True, "três"), (False, "3")])
def test_falar_numeros_por_extenso_respeita_o_perfil(com_ffmpeg, ligado, esperado):
    pedidos = []

    def responder(req):
        pedidos.append(json.loads(req.content))
        return httpx.Response(200, content=struct.pack("<f", 0.0) * 2400)

    voz = SimpleNamespace(**{**vars(VOZ), "numeros_por_extenso": ligado})
    cli = httpx.Client(transport=httpx.MockTransport(responder))
    tts.falar({"input": "Tenho 3 gatos.", "response_format": "wav"}, "http://x", voz, cliente=cli)
    assert esperado in pedidos[0]["input"]


def test_falar_formato_invalido():
    with pytest.raises(ErroApi, match="response_format"):
        tts.falar({"input": "oi", "response_format": "xyz"}, "http://x", VOZ)


def test_audio_base64_com_quebras_de_linha():
    dados = wav_silencio()
    b64 = base64.encodebytes(dados).decode()
    assert "\n" in b64
    parte, tipo = anexos._parte({"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}, "responses")
    assert tipo == "audio"
    assert base64.b64decode(parte["image_url"].split(",", 1)[1]) == dados


def test_input_audio_como_texto_vira_400():
    corpo = json.dumps({"input": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": "UklGRg=="}]}]}).encode()
    with pytest.raises(ErroApi) as e:
        anexos.reescrever(corpo, "responses")
    assert e.value.status == 400
