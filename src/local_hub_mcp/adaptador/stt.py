"""STT (/v1/audio/transcriptions) pelo chat do llama-server, cortando o áudio nas pausas.

Medido: áudio longo inteiro perde metade do texto (>~1 min); cortado nas pausas em pedaços <= 45 s,
10 min transcritos em 38 s com WER 5,5%.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import tempfile
import time
import wave
from email import policy
from email.parser import BytesParser
from pathlib import Path

import httpx

from . import ErroApi, ffmpeg

log = logging.getLogger(__name__)
SR = 16000
NOMES = {"pt": "Portuguese", "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
         "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ar": "Arabic", "nl": "Dutch", "pl": "Polish",
         "sv": "Swedish", "tr": "Turkish", "da": "Danish", "fi": "Finnish", "el": "Greek", "he": "Hebrew",
         "ru": "Russian", "hi": "Hindi", "vi": "Vietnamese", "th": "Thai", "id": "Indonesian"}


def campos_multipart(cab_tipo: str, corpo: bytes) -> tuple[dict, tuple[str, bytes]]:
    msg = BytesParser(policy=policy.HTTP).parsebytes(b"Content-Type: " + cab_tipo.encode() + b"\r\n\r\n" + corpo)
    if not msg.is_multipart():
        raise ErroApi(400, "use multipart/form-data com o campo 'file'")
    campos, arquivo = {}, None
    for parte in msg.iter_parts():
        nome = parte.get_param("name", header="content-disposition")
        dados = parte.get_payload(decode=True) or b""
        if nome == "file":
            arquivo = (parte.get_filename() or "audio", dados)
        elif nome:
            campos.setdefault(nome, dados.decode("utf-8", "replace"))
    if not arquivo or not arquivo[1]:
        raise ErroApi(400, "faltou o campo 'file'")
    return campos, arquivo


def decodificar(nome: str, dados: bytes) -> bytes:
    """Qualquer áudio/vídeo -> PCM16 mono 16 kHz (arquivo temporário: mp4/m4a precisam de seek)."""
    sufixo = Path(nome).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=sufixo, delete=False) as f:
        f.write(dados)
        caminho = f.name
    try:
        r = ffmpeg.rodar(["-loglevel", "error", "-i", caminho, "-vn", "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"])
    finally:
        os.unlink(caminho)
    if r.returncode or not r.stdout:
        raise ErroApi(400, "não consegui ler o áudio: " + r.stderr.decode(errors="replace")[-300:])
    return r.stdout


def tempo(s: float, sep: str) -> str:
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int(round((s % 1) * 1000)) % 1000:03d}"


def pausas(dados: bytes) -> list[float]:
    """Meio de cada silêncio >= 0,3 s (em segundos), via silencedetect do ffmpeg sobre o PCM."""
    r = ffmpeg.rodar(["-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
                      "-af", "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-"], dados)
    err = r.stderr.decode(errors="replace")
    ini = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", err)]
    fim = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    return [(a + b) / 2 for a, b in zip(ini, fim)]


def cortes(total: float, meios: list[float], max_seg: float = 45.0, min_seg: float = 15.0) -> list[float]:
    """Pontos de corte nas pausas, com pedaços entre min_seg e max_seg."""
    pontos, ultimo = [], 0.0
    while total - ultimo > max_seg:
        candidatos = [m for m in meios if ultimo + min_seg <= m <= ultimo + max_seg]
        corte = candidatos[-1] if candidatos else ultimo + max_seg   # sem pausa: corta seco
        pontos.append(corte)
        ultimo = corte
    return [0.0] + pontos + [total]


def wav_b64(amostras: bytes) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(amostras)
    return base64.b64encode(buf.getvalue()).decode()


def segmentos(dados: bytes, idioma: str | None, url: str, max_seg: float = 45.0,
              cliente: httpx.Client | None = None) -> list[tuple[float, float, str]]:
    """PCM16 16 kHz -> [(início_s, fim_s, texto)], um item por pedaço. idioma None = o modelo descobre."""
    total = len(dados) / 2 / SR
    pts = cortes(total, pausas(dados), max_seg, min(15.0, max_seg / 3))
    sistema = f"Perform ASR in {idioma}." if idioma else "Perform ASR."
    out = []
    cli = cliente or httpx.Client(timeout=600)
    try:
        for a, b in zip(pts, pts[1:]):
            trecho = dados[int(a * SR) * 2:int(b * SR) * 2]
            r = cli.post(f"{url}/v1/chat/completions", json={
                "temperature": 0, "max_tokens": 2048,
                "messages": [{"role": "system", "content": sistema},
                             {"role": "user", "content": [{"type": "input_audio",
                                                           "input_audio": {"data": wav_b64(trecho), "format": "wav"}}]}]})
            if r.status_code != 200:
                raise ErroApi(502, f"o llama-server respondeu {r.status_code} na transcrição: {r.text[:200]}", "server_error")
            out.append((a, b, r.json()["choices"][0]["message"]["content"].strip()))
    finally:
        if cliente is None:
            cli.close()
    return out


def transcrever(campos: dict, arquivo: tuple[str, bytes], url: str, voz,
                cliente: httpx.Client | None = None) -> tuple[int, str, bytes]:
    nome, dados = arquivo
    lang = (campos.get("language") or voz.idioma).strip().lower()
    idioma = None if lang == "auto" else NOMES.get(lang.split("-")[0], lang.capitalize())
    t0 = time.time()
    pcm = decodificar(nome, dados)
    dur = len(pcm) / 2 / SR
    segs = segmentos(pcm, idioma, url, voz.pedaco_stt_s, cliente)
    texto = " ".join(t for _, _, t in segs).strip()
    fmt = campos.get("response_format", "json")
    log.info("[stt] %.1f s de áudio (%s) em %.1f s: %r", dur, idioma or "auto", time.time() - t0, texto[:70])
    if fmt == "text":
        return 200, "text/plain; charset=utf-8", texto.encode()
    if fmt in ("srt", "vtt"):
        sep = "," if fmt == "srt" else "."
        linhas = ["WEBVTT", ""] if fmt == "vtt" else []
        for i, (a, b, t) in enumerate(segs, 1):
            if fmt == "srt":
                linhas.append(str(i))
            linhas += [f"{tempo(a, sep)} --> {tempo(b, sep)}", t, ""]
        return 200, "text/plain; charset=utf-8", "\n".join(linhas).encode()
    if fmt == "verbose_json":
        corpo = {"task": "transcribe", "language": (idioma or "auto").lower(), "duration": round(dur, 2),
                 "text": texto,
                 "segments": [{"id": i, "seek": 0, "start": round(a, 2), "end": round(b, 2), "text": t,
                               "tokens": [], "temperature": 0.0, "avg_logprob": 0.0,
                               "compression_ratio": 0.0, "no_speech_prob": 0.0}
                              for i, (a, b, t) in enumerate(segs)]}
    else:
        corpo = {"text": texto, "usage": {"type": "duration", "seconds": round(dur)}}
    return 200, "application/json", json.dumps(corpo, ensure_ascii=False).encode()
