"""Reescreve anexos de áudio/vídeo para o único formato que o /v1/responses do llama-server aceita.

Medido (LFM3, build 11968):
- o /v1/responses só aceita input_text e input_image; input_image com data URL de áudio/vídeo funciona
  (o llama-server detecta pelo conteúdo);
- o llama-server só decodifica wav, mp3 e flac: webm/ogg/m4a viram wav;
- vídeo webm (vp9 + opus) é visto mas NÃO ouvido: remuxar para mp4 com AAC;
- com áudio na entrada e um system prompt qualquer, o modelo tenta responder em áudio e o texto corta:
  acrescentar "Respond with text." ao FIM do system prompt; vídeo precisa da frase que faz ouvir a trilha.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import mimetypes
import os
import re
import tempfile
from pathlib import Path

from . import ErroApi, ffmpeg

log = logging.getLogger(__name__)
FRASE_VIDEO = "Answer questions about the video using what you see and hear."
FRASE_TEXTO = "Respond with text."


def _le_direto(dados: bytes) -> bool:
    """O llama-server (miniaudio) só decodifica wav, mp3 e flac."""
    return (dados[:4] == b"RIFF" and dados[8:12] == b"WAVE") or dados[:3] == b"ID3" or dados[:4] == b"fLaC" \
        or (len(dados) > 1 and dados[0] == 0xFF and dados[1] & 0xE0 == 0xE0)


def _para_wav(dados: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(dados)
        caminho = f.name
    try:
        r = ffmpeg.rodar(["-loglevel", "error", "-i", caminho, "-vn", "-ac", "1", "-ar", "16000",
                          "-c:a", "pcm_s16le", "-f", "wav", "-"])
    finally:
        os.unlink(caminho)
    if r.returncode or not r.stdout:
        raise ErroApi(400, "não consegui ler o áudio anexado: " + r.stderr.decode(errors="replace")[-200:])
    return r.stdout


def _video_mp4(dados: bytes) -> bytes:
    """Vídeo fora de mp4 (webm...) -> mp4 com áudio AAC."""
    r = None
    with tempfile.TemporaryDirectory() as d:
        ent, sai = os.path.join(d, "in.bin"), os.path.join(d, "out.mp4")
        Path(ent).write_bytes(dados)
        for video in (["-c:v", "copy"], ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28"]):
            r = ffmpeg.rodar(["-y", "-loglevel", "error", "-i", ent, *video,
                              "-c:a", "aac", "-movflags", "+faststart", sai])
            if r.returncode == 0:
                return Path(sai).read_bytes()
    detalhe = r.stderr.decode(errors="replace")[-200:] if r is not None else ""
    raise ErroApi(400, "não consegui ler o vídeo anexado: " + detalhe)


def _midia(mime: str, b64dados: str, api: str, nome: str = "") -> dict:
    """Parte de conteúdo no formato que o llama-server aceita: imagem com data URL."""
    try:
        b64dados = re.sub(r"\s+", "", b64dados)
        if not b64dados:
            raise ValueError("anexo vazio")
        if mime.startswith("video/") and mime != "video/mp4":
            dados = _video_mp4(base64.b64decode(b64dados, validate=True))
            log.info("[anexo] vídeo %s %s convertido para mp4 (áudio AAC)", mime, nome)
            mime, b64dados = "video/mp4", base64.b64encode(dados).decode()
        if mime.startswith("audio/"):
            dados = base64.b64decode(b64dados, validate=True)
            if not _le_direto(dados):
                antes = len(dados)
                dados = _para_wav(dados)
                log.info("[anexo] áudio %s %s (%d KB) convertido para wav", mime, nome, antes // 1024)
                mime = "audio/wav"
            b64dados = base64.b64encode(dados).decode()
    except (binascii.Error, ValueError, TypeError) as e:
        raise ErroApi(400, "o anexo não é base64 válido; reenvie o arquivo") from e
    url = f"data:{mime};base64,{b64dados}"
    return {"type": "input_image", "image_url": url} if api == "responses" else \
        {"type": "image_url", "image_url": {"url": url}}


def _data_url(s: str | None) -> tuple[str | None, str | None]:
    m = re.match(r"data:([^;,]+)(?:;[^,]*)?,(.*)", s or "", re.S)
    return (m.group(1).lower(), m.group(2)) if m else (None, None)


def _parte(p: dict, api: str) -> tuple[dict | None, str | None]:
    """(nova parte, tipo de mídia) se a parte for áudio/vídeo; senão (None, None)."""
    t = p.get("type")
    if t == "input_audio":
        a = p.get("input_audio") if isinstance(p.get("input_audio"), dict) else {}
        if not a.get("data"):
            raise ErroApi(400, "o input_audio precisa de 'data' com o áudio em base64")
        return _midia(f"audio/{(a.get('format') or 'wav').lower()}", a["data"], api), "audio"
    if t == "input_video":
        v = p.get("input_video") if isinstance(p.get("input_video"), dict) else {}
        if v.get("data"):
            return _midia(f"video/{(v.get('format') or 'mp4').lower()}", v["data"], api), "video"
        mime, dados = _data_url(v.get("url"))
        if mime:
            return _midia(mime, dados, api), "video"
        return None, None
    if t in ("input_file", "file"):
        f = p.get("file") if t == "file" else p
        f = f if isinstance(f, dict) else {}
        mime, dados = _data_url(f.get("file_data"))
        nome = f.get("filename") or ""
        if not mime and f.get("file_data"):
            mime, dados = mimetypes.guess_type(nome)[0], f["file_data"]
        if mime and mime.split("/")[0] in ("audio", "video"):
            return _midia(mime, dados, api, nome), mime.split("/")[0]
        return None, None
    if t in ("input_image", "image_url"):
        url = p.get("image_url")
        url = url.get("url") if isinstance(url, dict) else url
        mime, dados = _data_url(url)
        if mime and mime.split("/")[0] in ("audio", "video"):
            return _midia(mime, dados, api), mime.split("/")[0]
    return None, None


def reescrever(corpo: bytes, api: str) -> bytes | None:
    """Reescreve anexos de áudio/vídeo de /v1/responses (api='responses') ou /v1/chat/completions ('chat').
    Devolve o corpo novo, ou None se não havia nada para mexer."""
    try:
        req = json.loads(corpo)
    except Exception:
        return None
    itens = req.get("input") if api == "responses" else req.get("messages")
    if not isinstance(itens, list):
        return None
    tipos = []
    for item in itens:
        conteudo = item.get("content") if isinstance(item, dict) else None
        if not isinstance(conteudo, list):
            continue
        for i, p in enumerate(conteudo):
            if isinstance(p, dict):
                nova, tipo = _parte(p, api)
                if nova:
                    conteudo[i] = nova
                    tipos.append(tipo)
    if not tipos:
        return None
    sufixo = ([FRASE_VIDEO] if "video" in tipos else []) + [FRASE_TEXTO]
    sistema = next((x for x in itens if isinstance(x, dict) and x.get("role") in ("system", "developer")
                    and isinstance(x.get("content"), str)), None)
    if sistema:
        falta = [f for f in sufixo if f not in sistema["content"]]
        if falta:
            sistema["content"] = sistema["content"].rstrip() + "\n\n" + " ".join(falta)
    elif api == "responses":
        req["instructions"] = ((req.get("instructions") or "") + "\n" + " ".join(sufixo)).strip()
    else:
        itens.insert(0, {"role": "system", "content": " ".join(sufixo)})
    log.info("[anexo] /%s: %s reescrito(s) para o llama-server", api, ", ".join(tipos))
    return json.dumps(req).encode()
