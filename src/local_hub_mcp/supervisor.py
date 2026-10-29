"""Supervisor: sobe e derruba os llama-server (um por perfil, cada um numa porta interna).

Medido no Windows: dois llama-server conseguem ouvir a mesma porta (a porta é conferida antes);
modelo que não cabe sai com código 1 e sem mensagem; transbordo de VRAM não aparece no nvidia-smi
(lido nos contadores por PID).
"""
from __future__ import annotations

import asyncio
import collections
import datetime
import os
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from . import jobobject
from . import recursos as recursos_padrao
from .config import Config
from .perfis import Catalogo, ErroPerfil, Medido, Perfil, argv, caminho_seguro, partes, tamanho_gb, validar
from .planejador import NoAr, planejar

LIMITE_COMPARTILHADA_GB = 0.3
LINHAS_LOG = 2000
SEM_JANELA = 0x08000000 if os.name == "nt" else 0


ARQUITETURA_RE = re.compile(r"unknown model architecture: '([^']+)'")
PROJETOR_RE = re.compile(r"unknown projector type[^\n]*", re.I)


def diagnosticar(linhas: list[str]) -> str | None:
    """Motivo conhecido da saída do llama-server, a partir do log (None se não reconhecer)."""
    texto = "\n".join(linhas)
    m = ARQUITETURA_RE.search(texto)
    if m:
        return (f"o llama.cpp instalado não reconhece a arquitetura '{m.group(1)}' deste modelo (modelo novo demais "
                f"para este build); atualize o llama.cpp ou escolha outro modelo")
    if PROJETOR_RE.search(texto):
        return ("o llama.cpp instalado não reconhece o tipo de projetor do mmproj deste modelo; atualize o "
                "llama.cpp, use outro mmproj ou suba o perfil sem mmproj (só texto)")
    return None


class ErroSupervisor(Exception):
    pass


@dataclass
class Processo:
    alias: str
    onde: str
    porta: int
    popen: subprocess.Popen
    inicio: float
    estado: str = "subindo"   # subindo | no_ar | caiu
    log: collections.deque = field(default_factory=lambda: collections.deque(maxlen=LINHAS_LOG))
    pronto_em: float | None = None
    assinatura: tuple = ()

    @property
    def pid(self) -> int:
        return self.popen.pid


def porta_responde(porta: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", porta), timeout=timeout):
            return True
    except OSError:
        return False


class Supervisor:
    def __init__(self, cfg: Config, catalogo: Catalogo, recursos=recursos_padrao,
                 ffmpeg_dir: Path | None = None, prefixo: list[str] | None = None):
        self.cfg, self.catalogo, self.recursos = cfg, catalogo, recursos
        self.ffmpeg_dir = ffmpeg_dir
        self.prefixo = prefixo   # testes: troca o executável do llama-server por [python, fake.py]
        self.processos: dict[str, Processo] = {}
        self.medicoes = {}
        self.ultimos_logs = {}
        self._trava = asyncio.Lock()
        self.ultimas_falhas: dict[str, dict] = {}
        self._desligando = False

    async def exclusivo(self, func):
        """Roda func(arquivos_em_uso) sob a mesma trava do subir/derrubar: nenhuma subida se intercala entre
        a captura dos arquivos em uso e o que func faz com eles (ex.: apagar arquivos)."""
        async with self._trava:
            futuro = asyncio.ensure_future(asyncio.to_thread(func, self.arquivos_em_uso()))
            try:
                return await asyncio.shield(futuro)
            except asyncio.CancelledError:
                # a thread não para com o cancelamento: segura a trava até ela terminar, senão um subir
                # registraria um processo cujos arquivos a thread ainda vai apagar
                while not futuro.done():
                    try:
                        await asyncio.shield(futuro)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                raise

    def arquivos_em_uso(self) -> dict:
        """Arquivos (resolvidos) dos processos subindo/no ar, pelo que foi de fato carregado (a assinatura)."""
        usados = {}
        for proc in self.processos.values():
            if proc.estado not in ("subindo", "no_ar") or not proc.assinatura:
                continue
            arquivo, mmproj = proc.assinatura[0], proc.assinatura[1]
            for rel in [r for r in (arquivo, mmproj) if r]:
                try:
                    for a in partes(caminho_seguro(self.cfg.models_dir, rel)):
                        usados[a.resolve()] = proc.alias
                except ErroPerfil:
                    continue
        return usados

    @staticmethod
    def _assinatura(perfil):
        return (perfil.arquivo, perfil.mmproj, perfil.onde, tuple(perfil.flags))

    async def iniciar_medicao(self, alias: str, onde: str):
        async with self._trava:
            if self._desligando:
                raise ErroSupervisor("o serviço está desligando")
            if alias in self.medicoes:
                raise ErroSupervisor(f"'{alias}' já está sendo medido")
            if onde != "cpu":
                if any(n.onde != "cpu" for n in self.no_ar()):
                    raise ErroSupervisor("derrube antes o que está na GPU para medir")
                if any(o != "cpu" for o in self.medicoes.values()):
                    raise ErroSupervisor("espere a medição de GPU terminar")
            self.medicoes[alias] = onde

    def encerrar_medicao(self, alias: str):
        self.medicoes.pop(alias, None)

    # ---------------------------------------------------------------- consulta
    def pronto_em(self, alias: str) -> float | None:
        """Quando o perfil ficou no ar (para o coringa escolher o último que subiu)."""
        p = self.processos.get(alias)
        return p.pronto_em if p and p.estado == "no_ar" else None

    def porta_de(self, alias: str) -> int | None:
        p = self.processos.get(alias)
        return p.porta if p and p.estado == "no_ar" else None

    def no_ar(self) -> list[NoAr]:
        return [NoAr(p.alias, p.onde) for p in self.processos.values() if p.estado in ("subindo", "no_ar")]

    def logs(self, alias: str, n: int = 100) -> list[str]:
        p = self.processos.get(alias)
        buffer = p.log if p else self.ultimos_logs.get(alias)
        if buffer is None:
            raise ErroSupervisor(f"sem processo deste serviço para '{alias}'; veja logs/{alias}.log")
        return list(buffer)[-n:]

    def marcar_caiu(self, alias: str, motivo: str) -> None:
        proc = self.processos.get(alias)
        if proc and proc.estado == "no_ar":
            if proc.popen.poll() is None:
                proc.log.append("[local-hub-mcp] stream interrompido com o processo vivo: " + motivo)
                return
            proc.estado = "caiu"
            proc.log.append("[local-hub-mcp] " + motivo)

    def status(self) -> list[dict]:
        agora = time.time()
        return [{"alias": p.alias, "estado": p.estado, "porta": p.porta, "pid": p.pid, "onde": p.onde,
                 "no_ar_ha_s": round(agora - p.pronto_em) if p.pronto_em else None,
                 "codigo_saida": p.popen.poll()} for p in self.processos.values()]

    # ---------------------------------------------------------------- subir
    def _escolher_porta(self) -> int:
        usadas = {p.porta for p in self.processos.values() if p.estado != "caiu"}
        a, b = self.cfg.portas_internas
        for porta in range(a, b + 1):
            if porta not in usadas and not porta_responde(porta):
                return porta
        raise ErroSupervisor(f"nenhuma porta livre em {a}-{b}")

    def _iniciar(self, perfil: Perfil, porta: int) -> Processo:
        cmd = argv(perfil, self.cfg.llama_server, self.cfg.models_dir, porta, self.ffmpeg_dir)
        if self.prefixo:
            cmd = [*self.prefixo, *cmd[1:]]
        self.cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 creationflags=SEM_JANELA)
        jobobject.apos_criar(popen.pid)
        proc = Processo(perfil.alias, perfil.onde, porta, popen, time.time(), assinatura=self._assinatura(perfil))
        proc.log.append("$ " + subprocess.list2cmdline(cmd))
        threading.Thread(target=self._ler_log, args=(proc,), daemon=True).start()
        return proc

    def _ler_log(self, proc: Processo) -> None:
        try:
            arquivo = self.cfg.logs_dir / f"{proc.alias}.log"
            if arquivo.exists() and arquivo.stat().st_size > 5 * 1024 * 1024:
                arquivo.replace(arquivo.with_suffix(".log.1"))
            with open(arquivo, "a", encoding="utf-8") as f:
                f.write(f"\n==== {datetime.datetime.now().isoformat(timespec='seconds')} porta {proc.porta}\n")
                for bruto in proc.popen.stdout:
                    linha = bruto.decode("utf-8", "replace").rstrip()
                    proc.log.append(linha)
                    f.write(linha + "\n")
                    f.flush()
        except OSError:
            for bruto in proc.popen.stdout:
                proc.log.append(bruto.decode("utf-8", "replace").rstrip())

    async def subir(self, alias: str) -> dict:
        """Sobe o perfil; a última falha de cada alias fica em `ultimas_falhas` (mostrada no status)."""
        try:
            r = await self._subir(alias)
        except Exception as e:
            self.ultimas_falhas[alias] = {"erro": str(e), "em": datetime.datetime.now().isoformat(timespec="seconds")}
            raise
        self.ultimas_falhas.pop(alias, None)
        return r

    async def _subir(self, alias: str) -> dict:
        async with self._trava:
            if self._desligando:
                raise ErroSupervisor("o serviço está desligando; não sobe mais modelos")
            perfil = self.catalogo.obter(alias)
            validar(perfil, self.cfg.models_dir)
            atual = self.processos.get(alias)
            if atual and atual.estado == "no_ar" and atual.popen.poll() is not None:
                atual.estado = "caiu"   # morreu antes do vigiar notar: segue como caído (é recolhido abaixo)
            if atual and atual.estado == "no_ar" and atual.assinatura != self._assinatura(perfil):
                raise ErroSupervisor(f"o perfil '{alias}' mudou desde que subiu; chame derrubar_modelo e subir_modelo para aplicar")
            if atual and atual.estado == "no_ar" and not porta_responde(atual.porta)                     and not (await asyncio.sleep(0.5) or porta_responde(atual.porta, timeout=1.0)):
                raise ErroSupervisor(f"o processo de '{alias}' está vivo mas não responde na porta; "
                                     f"chame derrubar_modelo e suba de novo")
            if atual and atual.estado in ("subindo", "no_ar"):
                return {"ok": True, "alias": alias, "estado": atual.estado, "porta": atual.porta,
                        "pid": atual.pid, "avisos": ["já está subindo em outra chamada; consulte status em alguns segundos"
                                                    if atual.estado == "subindo" else "já estava no ar"]}
            if perfil.onde != "cpu":
                for medindo, onde in self.medicoes.items():
                    if onde != "cpu":
                        raise ErroSupervisor(f"espere a medição de '{medindo}' terminar")
            subindo = [p.alias for p in self.processos.values() if p.estado == "subindo" and p.onde != "cpu"]
            if subindo and perfil.onde != "cpu":
                raise ErroSupervisor(f"espere '{', '.join(subindo)}' terminar de subir antes de subir outro na GPU")
            if atual:   # anterior em 'caiu': mata ANTES de medir, senão a VRAM dele conta contra o novo
                await self._descartar(atual)
            for tentativa in range(3):   # a VRAM só volta depois que o processo sai: relê antes de recusar
                estado = await asyncio.to_thread(self.recursos.ler_estado)
                plano = planejar(perfil, tamanho_gb(perfil, self.cfg.models_dir), estado, self.no_ar())
                if plano.cabe or not atual:
                    break
                await asyncio.sleep(1)
            if not plano.cabe:
                raise ErroSupervisor("não cabe agora: " + "; ".join(plano.motivos))
            proc = self._iniciar(perfil, self._escolher_porta())
            self.processos[alias] = proc
        return await self._esperar_pronto(perfil, proc, plano.avisos)

    async def _esperar_pronto(self, perfil: Perfil, proc: Processo, avisos: list[str]) -> dict:
        try:
            url = f"http://127.0.0.1:{proc.porta}/health"
            limite = time.time() + self.cfg.load_timeout_s
            async with httpx.AsyncClient(timeout=3) as cli:
                while True:
                    if proc.popen.poll() is not None:
                        if self.processos.get(perfil.alias) is not proc:
                            raise ErroSupervisor(f"'{perfil.alias}' foi derrubado enquanto subia")
                        proc.estado = "caiu"
                        await asyncio.sleep(0.2)   # deixa o leitor de log pegar as últimas linhas
                        fim = "\n".join(list(proc.log)[-30:])
                        motivo = diagnosticar(list(proc.log)) or "código 1 sem mensagem costuma ser falta de VRAM"
                        raise ErroSupervisor(f"o llama-server de '{perfil.alias}' saiu com código {proc.popen.returncode} "
                                             f"({motivo}). Últimas linhas:\n{fim}")
                    try:
                        if (await cli.get(url)).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.time() > limite:
                        raise ErroSupervisor(f"'{perfil.alias}' não ficou pronto em {self.cfg.load_timeout_s} s; derrubado")
                    await asyncio.sleep(1)
            proc.estado = "no_ar"
            proc.pronto_em = time.time()
            carga = proc.pronto_em - proc.inicio
            await asyncio.sleep(2)
            # derrubar_modelo pode ter rodado durante a carga ou a espera acima (ele não espera o subir)
            if self.processos.get(perfil.alias) is not proc or proc.popen.poll() is not None:
                raise ErroSupervisor(f"'{perfil.alias}' foi derrubado ou caiu enquanto subia")
            mem = (await asyncio.to_thread(self.recursos.memoria_gpu_processos)).get(proc.pid)
            ram = await asyncio.to_thread(self.recursos.ram_processo_gb, proc.pid)
            vram = mem.dedicada_gb if mem else 0.0
            compartilhada = mem.compartilhada_gb if mem else 0.0
            avisos = list(avisos)
            if compartilhada > LIMITE_COMPARTILHADA_GB:
                avisos.append(f"transbordou {compartilhada:.2f} GB da VRAM para a RAM (lento, causa travadas): "
                              f"reduza -c ou -ub, ou aumente -ncmoe")
            mudou = self._assinatura(self.catalogo.obter(perfil.alias)) != proc.assinatura
            if mudou:
                avisos.append("o perfil mudou durante a subida; medição não gravada")
            if perfil.onde != "cpu" and mem is None:
                avisos.append("não consegui ler a VRAM do processo; medição não gravada")
            if not mudou and (perfil.onde == "cpu" or mem is not None):
                self.catalogo.gravar_medido(perfil.alias, Medido(round(vram, 2), round(ram, 2), round(carga, 1),
                                                                 datetime.date.today().isoformat()))
            return {"ok": True, "alias": perfil.alias, "estado": "no_ar", "porta": proc.porta, "pid": proc.pid,
                    "carga_s": round(carga, 1), "vram_gb": round(vram, 2), "compartilhada_gb": round(compartilhada, 2),
                    "ram_gb": round(ram, 2), "avisos": avisos}
        except BaseException:
            try:
                await self._matar(proc)
                for _ in range(20):
                    if not porta_responde(proc.porta):
                        break
                    await asyncio.sleep(0.5)
            except BaseException:
                pass  # preserva a excecao original, inclusive cancelamento
            proc.estado = "caiu"
            raise

    # ---------------------------------------------------------------- derrubar
    async def _descartar(self, proc: Processo) -> None:
        """Mata um processo que não está servindo e espera a porta dele fechar; tira do dicionário."""
        await self._matar(proc)
        for _ in range(20):
            if not porta_responde(proc.porta):
                break
            await asyncio.sleep(0.5)
        if self.processos.get(proc.alias) is proc:
            self.ultimos_logs[proc.alias] = proc.log
            del self.processos[proc.alias]

    async def descartar_caido(self, alias: str) -> None:
        """Para quem vai apagar o perfil: recolhe (mata) um processo em 'caiu', vivo ou não."""
        async with self._trava:
            proc = self.processos.get(alias)
            if proc and proc.estado in ("subindo", "no_ar"):
                raise ErroSupervisor(f"'{alias}' está no ar; derrube antes")
            if proc:
                await self._descartar(proc)

    async def _matar(self, proc: Processo) -> None:
        if proc.popen.poll() is None:
            proc.popen.terminate()   # TerminateProcess no Windows
            try:
                await asyncio.to_thread(proc.popen.wait, 15)
            except subprocess.TimeoutExpired:
                proc.popen.kill()
                await asyncio.to_thread(proc.popen.wait, 5)

    async def derrubar(self, alias: str) -> dict:
        async with self._trava:
            proc = self.processos.get(alias)
            if not proc:
                raise ErroSupervisor(f"'{alias}' não está no ar (este serviço só derruba o que ele mesmo subiu)")
            antes = await asyncio.to_thread(self.recursos.ler_estado)
            await self._matar(proc)
            self.ultimos_logs[alias] = proc.log
            del self.processos[alias]
            for _ in range(20):
                if not porta_responde(proc.porta):
                    break
                await asyncio.sleep(0.5)
            avisos = []
            if porta_responde(proc.porta):
                avisos.append(f"a porta {proc.porta} ainda responde: outro processo pode estar nela")
            await asyncio.sleep(1)
            if proc.pid in await asyncio.to_thread(self.recursos.memoria_gpu_processos):
                avisos.append(f"o PID {proc.pid} ainda aparece nos contadores de GPU")
            depois = await asyncio.to_thread(self.recursos.ler_estado)
            livre = lambda e: round(e.gpu.livre_gb, 2) if e.gpu else None
            return {"ok": True, "alias": alias, "vram_livre_antes_gb": livre(antes),
                    "vram_livre_depois_gb": livre(depois), "avisos": avisos}

    # ---------------------------------------------------------------- ciclo de vida
    async def vigiar(self, intervalo: float = 10) -> None:
        """Marca 'caiu' o processo que morreu. Sem reinício automático na v1 (esconderia o problema)."""
        while True:
            await asyncio.sleep(intervalo)
            for proc in list(self.processos.values()):
                if proc.estado in ("subindo", "no_ar") and proc.popen.poll() is not None:
                    proc.estado = "caiu"
                    proc.log.append(f"[local-hub-mcp] o processo saiu com código {proc.popen.returncode}")

    async def desligar(self) -> None:
        async with self._trava:   # nenhum subir registra processo depois daqui
            self._desligando = True
        for proc in list(self.processos.values()):
            await self._matar(proc)
            for _ in range(20):
                if not porta_responde(proc.porta):
                    break
                await asyncio.sleep(0.5)
        self.processos.clear()
