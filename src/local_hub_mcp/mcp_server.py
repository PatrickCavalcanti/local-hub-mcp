"""Ferramentas MCP: fachada fina sobre os módulos. Toda tool devolve um dict com 'ok'."""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from . import arquivos, hf, jobobject, medir as medir_mod, recursos, testes_modelo
from .config import Config
from .perfis import PARTE_RE, Catalogo, ErroPerfil, Perfil, partes, tamanho_gb
from .supervisor import ErroSupervisor, Supervisor
from .tarefas import Tarefas

log = logging.getLogger(__name__)
ESPERADOS = (ErroPerfil, ErroSupervisor, hf.ErroHf, medir_mod.ErroMedir, ValueError)
INSTRUCOES = ("local-hub-mcp gerencia modelos llama.cpp locais. Fluxo: status -> listar_modelos -> "
              "subir_modelo(alias) -> use o modelo pelo /v1 (OpenAI) com model=alias -> derrubar_modelo(alias). "
              "Só um modelo de GPU por vez numa GPU pequena; perfis 'cpu' convivem. "
              "Operações longas devolvem 'tarefa': consulte com tarefa(id).")


@dataclass
class Contexto:
    cfg: Config
    catalogo: Catalogo
    supervisor: Supervisor
    tarefas: Tarefas
    ffmpeg: Path | None
    versao_llama: str


def criar_mcp(ctx: Contexto) -> MCPServer:
    mcp = MCPServer("local-hub-mcp", instructions=INSTRUCOES)

    def auditar(tool: str, args: dict) -> None:
        ctx.cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        linha = json.dumps({"em": datetime.datetime.now().isoformat(timespec="seconds"), "tool": tool, "args": args},
                           ensure_ascii=False)
        with open(ctx.cfg.logs_dir / "auditoria.log", "a", encoding="utf-8") as f:
            f.write(linha + "\n")

    async def seguro(coro) -> dict:
        try:
            return await coro
        except ESPERADOS as e:
            return {"ok": False, "erro": str(e)}
        except Exception as e:   # nunca stack trace para o cliente
            log.exception("erro interno numa tool")
            return {"ok": False, "erro": f"erro interno: {type(e).__name__}: {e} (detalhe no log do serviço)"}

    # ------------------------------------------------------------ leitura
    @mcp.tool()
    async def status() -> dict:
        """Perfis no ar (porta, PID, VRAM dedicada e compartilhada), caídos, VRAM e RAM livres,
        versão do llama.cpp, ffmpeg e tarefas em andamento."""
        async def _():
            estado = await asyncio.to_thread(recursos.ler_estado)
            mem = await asyncio.to_thread(recursos.memoria_gpu_processos)
            procs = ctx.supervisor.status()
            for p in procs:
                m = mem.get(p["pid"])
                p["vram_gb"] = round(m.dedicada_gb, 2) if m else 0.0
                p["compartilhada_gb"] = round(m.compartilhada_gb, 2) if m else 0.0
            return {"ok": True, "processos": procs,
                    "medicoes": [{"alias": a, "onde": o} for a, o in ctx.supervisor.medicoes.items()],
                    "gpu": {k: round(v, 2) for k, v in asdict(estado.gpu).items()} if estado.gpu else None,
                    "ram_livre_gb": round(estado.ram_livre_gb, 1), "ram_total_gb": round(estado.ram_total_gb, 1),
                    "llama_cpp": ctx.versao_llama, "ffmpeg": str(ctx.ffmpeg) if ctx.ffmpeg else None,
                    "job_object": jobobject.modo(), "tarefas": ctx.tarefas.listar(),
                    "falhas_recentes_de_subida": ctx.supervisor.ultimas_falhas}
        return await seguro(_())

    @mcp.tool()
    async def listar_modelos() -> dict:
        """Perfis (no ar ou não, tamanho, medição) e os GGUF de models_dir que ainda não têm perfil."""
        async def _():
            perfis, usados = [], set()
            for p in ctx.catalogo.listar():
                for a in partes(ctx.cfg.models_dir / p.arquivo):
                    usados.add(a.resolve())
                if p.mmproj:
                    usados.add((ctx.cfg.models_dir / p.mmproj).resolve())
                perfis.append({"alias": p.alias, "onde": p.onde, "capacidades": p.capacidades,
                               "no_ar": ctx.supervisor.porta_de(p.alias) is not None, "arquivo": p.arquivo,
                               "tamanho_gb": round(tamanho_gb(p, ctx.cfg.models_dir), 2),
                               "medido": asdict(p.medido) if p.medido else None, "notas": p.notas})
            sem_perfil = []
            for g in sorted(ctx.cfg.models_dir.rglob("*.gguf")):
                if g.resolve() in usados or "mmproj" in g.name.lower():
                    continue
                m = PARTE_RE.search(g.name)
                if m and int(m.group(1)) > 1:
                    continue   # só a primeira parte de um GGUF dividido
                sem_perfil.append(g.relative_to(ctx.cfg.models_dir).as_posix())
            return {"ok": True, "perfis": perfis, "sem_perfil": sem_perfil}
        return await seguro(_())

    @mcp.tool()
    async def procurar_modelo(texto: str, limite: int = 10, ordem: str = "trending") -> dict:
        """Busca modelos no Hugging Face pelo nome (ex.: "Unlimited-OCR" ou "baidu unlimited-ocr-2").
        Devolve 'originais' (o modelo do autor, que mostra a versão mais nova; não roda direto no llama.cpp) e
        'gguf' (conversões que dá para baixar; 'origem' diz de qual original vieram).
        ordem: 'trending' (padrão, como o site), 'downloads', 'recentes' ou 'likes'.
        Para listar todas as conversões GGUF de um original, use conversoes_gguf."""
        async def _():
            return {"ok": True, **await asyncio.to_thread(hf.procurar, texto, limite, ordem)}
        return await seguro(_())

    @mcp.tool()
    async def conversoes_gguf(repo_id: str, ordem: str = "downloads", limite: int = 20) -> dict:
        """Lista as conversões GGUF de um modelo original do Hugging Face (ex.: repo_id="baidu/Unlimited-OCR-2"),
        pela ligação base_model do Hub. ordem: 'downloads' (padrão), 'trending', 'recentes' ou 'likes'."""
        async def _():
            return {"ok": True, **await asyncio.to_thread(hf.conversoes_gguf, repo_id, limite, ordem)}
        return await seguro(_())

    @mcp.tool()
    async def detalhes_repo(repo_id: str) -> dict:
        """Arquivos GGUF de um repositório: quant, tamanho, partes, mmproj e estimativa de onde cabe."""
        async def _():
            estado = await asyncio.to_thread(recursos.ler_estado)
            return {"ok": True, **await asyncio.to_thread(hf.detalhes, repo_id, estado)}
        return await seguro(_())

    @mcp.tool()
    async def logs(alias: str, linhas: int = 100) -> dict:
        """Últimas linhas do llama-server de um perfil (subido por este serviço)."""
        async def _():
            return {"ok": True, "alias": alias, "linhas": ctx.supervisor.logs(alias, max(1, min(linhas, 2000)))}
        return await seguro(_())

    @mcp.tool()
    async def info_conexao() -> dict:
        """Como configurar um cliente: base URL do /v1, URL do MCP, aliases e capacidades por perfil."""
        async def _():
            return {"ok": True, "base_url": ctx.cfg.base_url + "/v1", "mcp_url": ctx.cfg.base_url + "/mcp",
                    "chave_necessaria": bool(ctx.cfg.chaves),
                    "modelos": [{"model": p.alias, "capacidades": p.capacidades,
                                 "vozes": list(p.voz.vozes) if p.voz else None} for p in ctx.catalogo.listar()],
                    "igris": {"protocol": "openai", "base_url": ctx.cfg.base_url + "/v1", "api_style": "responses",
                              "model_id": "<alias do perfil>"}}
        return await seguro(_())

    @mcp.tool()
    async def tarefa(id: str) -> dict:
        """Estado e resultado de uma operação longa (em_andamento / ok / erro)."""
        return ctx.tarefas.consultar(id)

    # ------------------------------------------------------------ mudam a máquina
    @mcp.tool()
    async def baixar_modelo(repo_id: str, arquivos: list[str]) -> dict:
        """Baixa arquivos GGUF (e todas as partes de um dividido) do Hugging Face para models_dir/<pasta>/.
        Volta NA HORA com o id de uma 'tarefa' e segue baixando em segundo plano: acompanhe com tarefa(id),
        que mostra baixado_gb, total_gb e percentual. Não espere o download terminar para responder ao usuário."""
        auditar("baixar_modelo", {"repo_id": repo_id, "arquivos": arquivos})

        async def _():
            d = await asyncio.to_thread(hf.preparar_download, repo_id, arquivos, ctx.cfg.models_dir)
            r = await ctx.tarefas.rodar(f"baixar {repo_id}", asyncio.to_thread(hf.executar_download, d), 1.0,
                                        progresso=lambda: hf.progresso_download(d))
            if r.get("estado") == "em_andamento":
                r["mensagem"] = (f"baixando {round(d.total / hf.GB, 2)} GB em segundo plano para {d.pasta}/; "
                                 f"acompanhe com tarefa('{r['tarefa']}')")
            return r
        return await seguro(_())

    @mcp.tool()
    async def criar_perfil(alias: str, arquivo: str, onde: str, flags: list[str] | None = None,
                           capacidades: list[str] | None = None, mmproj: str | None = None, anexos: bool = False,
                           video: bool = False, voz: dict | None = None, notas: str = "") -> dict:
        """Cria um perfil. arquivo/mmproj relativos a models_dir; onde = gpu | gpu+cpu | cpu;
        flags = flags livres do llama-server (sem --port/--host/-m/--mmproj/--alias)."""
        auditar("criar_perfil", {"alias": alias, "arquivo": arquivo, "onde": onde, "flags": flags})
        async def _():
            p = Perfil.de_dict({"alias": alias, "arquivo": arquivo, "onde": onde, "flags": flags or [],
                                "capacidades": capacidades or ["llm.chat"], "mmproj": mmproj, "anexos": anexos,
                                "video": video, "voz": voz, "notas": notas})
            return {"ok": True, "perfil": ctx.catalogo.criar(p).para_dict()}
        return await seguro(_())

    @mcp.tool()
    async def editar_perfil(alias: str, arquivo: str | None = None, onde: str | None = None,
                            flags: list[str] | None = None, capacidades: list[str] | None = None,
                            mmproj: str | None = None, anexos: bool | None = None, video: bool | None = None,
                            voz: dict | None = None, notas: str | None = None) -> dict:
        """Altera campos de um perfil existente; só os parâmetros informados mudam. Vale na próxima vez que ele subir.
        onde: 'gpu' | 'gpu+cpu' | 'cpu'. flags: lista com cada token separado, ex.: ["-c", "8192", "-ub", "512"]
        (substitui a lista inteira). mmproj: caminho relativo a models_dir; "" remove. voz: só as chaves que mudam,
        ex.: {"idioma": "en"}. Em pt, {"numeros_por_extenso": true} liga o transformador
        que escreve os números por extenso antes da voz (desligado por padrão).
        O alias não muda (crie outro perfil). Mudar arquivo/onde/flags descarta a medição."""
        mudancas = {k: v for k, v in {"arquivo": arquivo, "onde": onde, "flags": flags, "capacidades": capacidades,
                                      "mmproj": mmproj, "anexos": anexos, "video": video, "voz": voz,
                                      "notas": notas}.items() if v is not None}
        if mudancas.get("mmproj") == "":
            mudancas["mmproj"] = None
        auditar("editar_perfil", {"alias": alias, "mudancas": mudancas})
        async def _():
            if not mudancas:
                raise ValueError("nenhum campo para mudar; informe ao menos um (ex.: flags)")
            resposta = {"ok": True, "perfil": ctx.catalogo.editar(alias, mudancas).para_dict()}
            if ctx.supervisor.porta_de(alias) is not None:
                resposta["aviso"] = "o perfil está no ar; a mudança vale depois de derrubar_modelo e subir_modelo"
            return resposta
        return await seguro(_())

    @mcp.tool()
    async def apagar_perfil(alias: str) -> dict:
        """Apaga um perfil (não apaga o arquivo GGUF). Recusa se o perfil estiver no ar."""
        auditar("apagar_perfil", {"alias": alias})
        async def _():
            await ctx.supervisor.descartar_caido(alias)   # recusa se no ar; mata o caído antes de esquecer
            ctx.catalogo.apagar(alias)
            return {"ok": True}
        return await seguro(_())

    @mcp.tool()
    async def apagar_arquivos(caminhos: list[str], confirmacao: str | None = None) -> dict:
        """APAGA arquivos de modelo do disco (irreversível). caminhos: relativos a models_dir — um .gguf (leva todas
        as partes de um dividido) ou uma pasta inteira (ex.: "baidu--unlimited-ocr-2"). Veja os caminhos em
        listar_modelos (campo sem_perfil).
        Duas etapas: 1) chame SEM confirmacao: nada é apagado; volta a lista de arquivos, o tamanho e um código.
        2) MOSTRE a lista ao usuário e, só depois do OK explícito dele, chame de novo com confirmacao=<código>.
        Recusa arquivo usado por um perfil (apague o perfil antes) e qualquer coisa fora de models_dir."""
        auditar("apagar_arquivos", {"caminhos": caminhos, "confirmacao": confirmacao})

        async def _():
            if not confirmacao:
                plano = await ctx.supervisor.exclusivo(
                    lambda em_uso: arquivos.planejar(ctx.cfg.models_dir, caminhos, ctx.catalogo.listar(), em_uso))
                return {"ok": True, "apagado": False, **plano,
                        "proximo_passo": "mostre a lista ao usuário; com o OK dele, chame apagar_arquivos de novo "
                                         "com os mesmos caminhos e confirmacao=<código>"}
            r = await ctx.supervisor.exclusivo(   # captura, conferência e exclusão sem subir no meio
                lambda em_uso: arquivos.remover(ctx.cfg.models_dir, caminhos, ctx.catalogo.listar(), confirmacao,
                                                em_uso))
            return {"ok": not r["falhas"], "apagado": True, **r}
        return await seguro(_())

    @mcp.tool()
    async def subir_modelo(alias: str) -> dict:
        """Sobe o llama-server do perfil, se couber (VRAM e RAM). Devolve porta, tempo de carga,
        VRAM usada e aviso de transbordo. Carga longa vira 'tarefa'."""
        auditar("subir_modelo", {"alias": alias})
        return await seguro(ctx.tarefas.rodar(f"subir {alias}", ctx.supervisor.subir(alias), ctx.cfg.espera_s))

    @mcp.tool()
    async def derrubar_modelo(alias: str) -> dict:
        """Derruba o llama-server do perfil e confere que a porta e a VRAM foram liberadas."""
        auditar("derrubar_modelo", {"alias": alias})
        return await seguro(ctx.supervisor.derrubar(alias))

    @mcp.tool()
    async def medir(alias: str, ncmoe: list[int] | None = None, ub: list[int] | None = None,
                    gravar: bool = False) -> dict:
        """Roda o llama-bench variando -ncmoe e -ub. gravar=true grava o melhor nas flags do perfil.
        Exige a GPU livre (derrube os perfis de GPU antes)."""
        auditar("medir", {"alias": alias, "ncmoe": ncmoe, "ub": ub, "gravar": gravar})

        async def _():
            perfil = ctx.catalogo.obter(alias)
            await ctx.supervisor.iniciar_medicao(alias, perfil.onde)
            try:
                cmd = medir_mod.montar_argv(ctx.cfg.llama_bench, ctx.cfg.models_dir, perfil, ncmoe, ub)
                linhas = medir_mod.interpretar(await asyncio.to_thread(medir_mod.rodar, cmd))
                escolhido = medir_mod.melhor(linhas)
                resultado = {"ok": True, "alias": alias, "tabela": linhas, "melhor": escolhido, "comando": cmd}
                if gravar:
                    flags = medir_mod.aplicar(perfil.flags, escolhido, bool(ncmoe), bool(ub))
                    resultado["flags_gravadas"] = ctx.catalogo.editar(alias, {"flags": flags}).flags
                return resultado
            finally:
                ctx.supervisor.encerrar_medicao(alias)
        return await seguro(ctx.tarefas.rodar(f"medir {alias}", _(), ctx.cfg.espera_s))

    @mcp.tool()
    async def testar_modelo(alias: str, capacidades: list[str] | None = None) -> dict:
        """Testes curtos contra o perfil no ar: texto, visão (imagem gerada), TTS e STT (ida e volta)."""
        async def _():
            perfil = ctx.catalogo.obter(alias)
            porta = ctx.supervisor.porta_de(alias)
            if porta is None:
                raise ErroSupervisor(f"'{alias}' não está no ar; chame subir_modelo antes")
            return await testes_modelo.testar(perfil, porta, capacidades)
        return await seguro(_())

    return mcp
