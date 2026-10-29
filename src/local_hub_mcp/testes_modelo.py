"""Testes curtos por capacidade contra um perfil no ar (direto no llama-server dele)."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import struct
import time
import wave
import zlib

import httpx

from .adaptador import ErroApi, stt, tts
from .perfis import Perfil


def png_vermelho(lado: int = 64) -> bytes:
    """PNG RGB de uma cor só, gerado aqui (nada de arquivo de teste com dado pessoal)."""
    def bloco(tipo: bytes, dados: bytes) -> bytes:
        return struct.pack(">I", len(dados)) + tipo + dados + struct.pack(">I", zlib.crc32(tipo + dados) & 0xFFFFFFFF)
    cru = (b"\x00" + b"\xff\x00\x00" * lado) * lado
    return (b"\x89PNG\r\n\x1a\n" + bloco(b"IHDR", struct.pack(">IIBBBBB", lado, lado, 8, 2, 0, 0, 0))
            + bloco(b"IDAT", zlib.compress(cru)) + bloco(b"IEND", b""))


def duracao_wav(dados: bytes) -> float:
    # WAV do ffmpeg por pipe traz o tamanho "desconhecido" (0xFFFFFFFF) no cabeçalho: conta os quadros lidos
    with wave.open(io.BytesIO(dados)) as w:
        quadros = len(w.readframes(w.getnframes())) // (w.getsampwidth() * w.getnchannels())
        return quadros / w.getframerate()


def _conteudo(resp: dict) -> str:
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()


async def _chat(cli: httpx.AsyncClient, url: str, alias: str, conteudo,
                max_tokens: int = 256) -> tuple[dict, float]:
    t0 = time.time()
    r = await cli.post(f"{url}/v1/chat/completions", json={
        "model": alias, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": conteudo}]})
    r.raise_for_status()
    return r.json(), time.time() - t0


async def _texto(cli, url, perfil) -> dict:
    t0 = time.time()
    try:
        resp, dt = await _chat(cli, url, perfil.alias, "Responda apenas com a palavra: pronto")
    except httpx.HTTPError as e:
        return {"capacidade": "texto", "ok": False, "segundos": round(time.time() - t0, 1),
                "detalhe": f"falha HTTP: {e}"}
    texto = _conteudo(resp)
    tps = (resp.get("timings") or {}).get("predicted_per_second")
    return {"capacidade": "texto", "ok": bool(texto), "segundos": round(dt, 1),
            "detalhe": f"{texto[:80]!r}" + (f", {tps:.1f} t/s" if tps else "")}


async def _visao(cli, url, perfil) -> dict:
    img = "data:image/png;base64," + base64.b64encode(png_vermelho()).decode()
    t0 = time.time()
    try:
        resp, dt = await _chat(cli, url, perfil.alias, [
            {"type": "text", "text": "Qual é a cor desta imagem: vermelho, verde, azul ou amarelo? Responda com uma palavra."},
            {"type": "image_url", "image_url": {"url": img}}], max_tokens=512)
    except httpx.HTTPError as e:
        return {"capacidade": "visao", "ok": False, "segundos": round(time.time() - t0, 1),
                "detalhe": f"falha HTTP: {e}"}
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    texto = (msg.get("content") or "").strip().lower()
    vermelho = ("verm", "red", "rojo", "rouge", "carmes", "crimson", "escarlat", "scarlet", "rubi", "ruby")
    outra_cor = re.search(r"\b(?:verde|azul|amarelo|green|blue|yellow)\b", texto)
    return {"capacidade": "visao", "ok": any(c in texto for c in vermelho) and not outra_cor,
            "segundos": round(dt, 1), "detalhe": f"imagem vermelha -> {texto[:60]!r}"}


async def _tts(url, perfil) -> tuple[dict, bytes | None]:
    t0 = time.time()
    try:
        _, _, wav = await asyncio.to_thread(tts.falar, {
            "input": "Teste de voz, um dois três.", "voice": next(iter(perfil.voz.vozes)),
            "response_format": "wav", "language": perfil.voz.idioma}, url, perfil.voz)
        dur = duracao_wav(wav)
    except ErroApi as e:
        return {"capacidade": "tts", "ok": False, "segundos": round(time.time() - t0, 1), "detalhe": str(e)}, None
    except httpx.HTTPError as e:
        return {"capacidade": "tts", "ok": False, "segundos": round(time.time() - t0, 1),
                "detalhe": f"falha HTTP: {e}"}, None
    except (wave.Error, EOFError) as e:
        return {"capacidade": "tts", "ok": False, "segundos": round(time.time() - t0, 1),
                "detalhe": f"falha WAV: {e}"}, None
    return {"capacidade": "tts", "ok": dur > 0.5, "segundos": round(time.time() - t0, 1),
            "detalhe": f"{dur:.1f} s de fala"}, wav


async def _stt(url, perfil, wav: bytes) -> dict:
    t0 = time.time()
    try:
        _, _, dados = await asyncio.to_thread(stt.transcrever, {"language": perfil.voz.idioma, "response_format": "json"},
                                              ("teste.wav", wav), url, perfil.voz)
        texto = json.loads(dados)["text"]
    except ErroApi as e:
        return {"capacidade": "stt", "ok": False, "segundos": round(time.time() - t0, 1), "detalhe": str(e)}
    except httpx.HTTPError as e:
        return {"capacidade": "stt", "ok": False, "segundos": round(time.time() - t0, 1),
                "detalhe": f"falha HTTP: {e}"}
    except (json.JSONDecodeError, KeyError) as e:
        return {"capacidade": "stt", "ok": False, "segundos": round(time.time() - t0, 1),
                "detalhe": f"resposta STT inválida: {e}"}
    return {"capacidade": "stt", "ok": any(p in texto.lower() for p in ("teste", "voz")),
            "segundos": round(time.time() - t0, 1), "detalhe": f"ida e volta: {texto[:80]!r}"}


async def testar(perfil: Perfil, porta: int, capacidades: list[str] | None = None,
                 cliente: httpx.AsyncClient | None = None) -> dict:
    url = f"http://127.0.0.1:{porta}"
    caps = capacidades or perfil.capacidades
    resultados: list[dict] = []
    cli = cliente or httpx.AsyncClient(timeout=300)
    try:
        if any(c in caps for c in ("llm.chat", "text.generate", "llm.fast")):
            resultados.append(await _texto(cli, url, perfil))
        if "llm.vision" in caps:
            resultados.append(await _visao(cli, url, perfil))
    finally:
        if cliente is None:
            await cli.aclose()
    if "tts" in caps and perfil.voz is not None:
        r, wav = await _tts(url, perfil)
        resultados.append(r)
        if "stt" in caps and wav:
            resultados.append(await _stt(url, perfil, wav))
    elif "stt" in caps:
        resultados.append({"capacidade": "stt", "ok": None, "segundos": 0,
                           "detalhe": "pulado: sem áudio de teste (o perfil não tem tts)"})
    return {"ok": all(r["ok"] is not False for r in resultados), "alias": perfil.alias, "testes": resultados}
