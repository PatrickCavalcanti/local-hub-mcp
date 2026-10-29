"""TTS (/v1/audio/speech) pelo /tts do llama-server.

Medido: o /tts para em 512 quadros (~41 s) -> texto em pedaços de ~250 caracteres; o modelo só tem as
vozes do perfil (LFM3: female/male); o LFM3 lê número em dígito errado -> por extenso antes (pt; desliga com voz.numeros_por_extenso=false).
Ajuste por voz (seed, tom em semitons, velocidade) aplicado depois com o ffmpeg.
"""
from __future__ import annotations

import logging
import re
import time

import httpx

from . import ErroApi, ffmpeg
from .numeros_pt import por_extenso

log = logging.getLogger(__name__)
TTS_RATE = 24000
FORMATOS = {  # response_format -> (args do ffmpeg, Content-Type)
    "mp3": (["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"], "audio/mpeg"),
    "opus": (["-c:a", "libopus", "-b:a", "64k", "-f", "ogg"], "audio/ogg"),
    "aac": (["-c:a", "aac", "-b:a", "128k", "-f", "adts"], "audio/aac"),
    "flac": (["-c:a", "flac", "-f", "flac"], "audio/flac"),
    "wav": (["-c:a", "pcm_s16le", "-f", "wav"], "audio/wav"),
    "pcm": (["-c:a", "pcm_s16le", "-f", "s16le"], "audio/pcm"),
}


def pedacos(texto: str, limite: int = 250) -> list[str]:
    """Frases agrupadas até ~limite caracteres; frase gigante quebra nas vírgulas/espaços."""
    frases = re.split(r"(?<=[.!?…;:])\s+", texto.strip())
    out, atual = [], ""
    for f in frases:
        while len(f) > limite:
            corte = max(f.rfind(",", 0, limite), f.rfind(" ", 0, limite))
            corte = corte if corte > limite // 3 else limite
            out.append((atual + " " + f[:corte + 1]).strip())
            atual, f = "", f[corte + 1:].strip()
        if len(atual) + len(f) + 1 > limite and atual:
            out.append(atual)
            atual = f
        else:
            atual = (atual + " " + f).strip()
    if atual:
        out.append(atual)
    return [p for p in out if re.search(r"\w", p)]


def atempo(v: float) -> str:
    v = min(max(v, 0.25), 4.0)
    filtros = []
    while v < 0.5:
        filtros.append("atempo=0.5")
        v /= 0.5
    filtros.append(f"atempo={v:.4f}")
    return ",".join(filtros)


def filtros(tom: float, velocidade: float) -> list[str]:
    """Tom (semitons, sem mudar a duração) + velocidade, como cadeia do ffmpeg."""
    f = []
    if tom:
        k = 2 ** (tom / 12)
        f.append(f"asetrate={TTS_RATE * k:.2f},aresample={TTS_RATE},{atempo(1 / k)}")
    if abs(velocidade - 1.0) > 1e-3:
        f.append(atempo(velocidade))
    return ["-af", ",".join(f)] if f else []


def falar(req: dict, url: str, voz, cliente: httpx.Client | None = None) -> tuple[int, str, bytes]:
    texto = (req.get("input") or "").strip()
    if not texto:
        raise ErroApi(400, "faltou 'input'")
    vozes = voz.vozes
    nome = str(req.get("voice") or "").lower()
    if nome not in vozes:
        # o Igris manda 'alloy' quando a config do modelo não define voz: cai na padrão, sem erro
        padrao = "female" if "female" in vozes else next(iter(vozes))
        log.info("[tts] voice %r não existe (vozes: %s): usando %s", req.get("voice"), ", ".join(vozes), padrao)
        nome = padrao
    lang = str(req.get("language") or voz.idioma).lower()
    if lang.startswith("pt") and voz.numeros_por_extenso:
        texto = por_extenso(texto)
    fmt = req.get("response_format", "mp3")
    if fmt not in FORMATOS:
        raise ErroApi(400, f"response_format '{fmt}' não suportado: {', '.join(FORMATOS)}")
    ajuste = vozes.get(nome) or {}
    velocidade = float(req.get("speed") or 1.0) * ajuste.get("velocidade", 1.0)
    t0 = time.time()
    silencio = b"\x00" * 4 * int(0.12 * TTS_RATE)   # 0,12 s entre pedaços (f32)
    bruto = bytearray()
    cli = cliente or httpx.Client(timeout=900)
    try:
        for i, p in enumerate(pedacos(texto, voz.pedaco_tts)):
            r = cli.post(f"{url}/tts", json={"input": p, "lang": lang, "voice": nome,
                                             "seed": ajuste.get("seed", -1), "response_format": "pcm"})
            if r.status_code != 200:
                raise ErroApi(502, f"llama-server /tts respondeu {r.status_code}: {r.content[:200]!r}", "server_error")
            bruto += (silencio if i else b"") + r.content
    finally:
        if cliente is None:
            cli.close()
    args, mime = FORMATOS[fmt]
    r = ffmpeg.rodar(["-loglevel", "error", "-f", "f32le", "-ar", str(TTS_RATE), "-ac", "1", "-i", "-",
                      *filtros(ajuste.get("tom", 0), velocidade), "-ar", str(TTS_RATE), *args, "-"], bytes(bruto))
    if r.returncode:
        raise ErroApi(500, "ffmpeg falhou: " + r.stderr.decode(errors="replace")[-300:], "server_error")
    log.info("[tts] %d car., voz %s, %s, %s: %.1f s de fala em %.1f s", len(texto), nome, lang, fmt,
             len(bruto) / 4 / TTS_RATE / velocidade, time.time() - t0)
    return 200, mime, r.stdout
