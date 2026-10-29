"""Decide se um perfil cabe agora, e explica por quê. Função pura: recebe o estado medido."""
from __future__ import annotations

from dataclasses import dataclass, field

from .perfis import Perfil
from .recursos import EstadoRecursos

MARGEM_VRAM_GB = 0.3
MARGEM_RAM_GB = 1.0
MIN_VRAM_GPU_CPU_GB = 1.0
RAM_BASE_GPU_GB = 0.5   # processo com tudo na GPU ainda usa RAM (mmap, buffers)


@dataclass
class NoAr:
    alias: str
    onde: str


@dataclass
class Plano:
    cabe: bool
    motivos: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)


def _gb(x: float) -> str:
    return f"{x:.1f} GB"


def planejar(p: Perfil, tamanho_gb: float, estado: EstadoRecursos, no_ar: list[NoAr]) -> Plano:
    plano = Plano(cabe=True)
    na_gpu = [n.alias for n in no_ar if n.onde != "cpu" and n.alias != p.alias]
    if p.onde != "cpu":
        if estado.gpu is None:
            return Plano(False, ["nenhuma GPU NVIDIA encontrada (nvidia-smi); use onde='cpu'"])
        livre = estado.gpu.livre_gb
        if p.medido is not None:
            precisa = p.medido.vram_gb + MARGEM_VRAM_GB
            if livre < precisa:
                quem = f"; no ar na GPU: {', '.join(na_gpu)}" if na_gpu else ""
                plano.cabe = False
                plano.motivos.append(f"precisa de {_gb(precisa)} de VRAM livre (medido + margem) e há {_gb(livre)}{quem}")
        elif na_gpu:
            plano.cabe = False
            plano.motivos.append(f"sem medição de VRAM para '{p.alias}': derrube antes o que está na GPU "
                                 f"({', '.join(na_gpu)})")
        elif p.onde == "gpu" and livre < tamanho_gb + MARGEM_VRAM_GB:
            plano.cabe = False
            plano.motivos.append(f"o modelo tem {_gb(tamanho_gb)} e há {_gb(livre)} de VRAM livre; "
                                 "use onde='gpu+cpu' com -ncmoe, ou -ngl menor")
        elif p.onde == "gpu+cpu" and livre < MIN_VRAM_GPU_CPU_GB:
            plano.cabe = False
            plano.motivos.append(f"há só {_gb(livre)} de VRAM livre; "
                                 "derrube o que ocupa a GPU ou use onde='cpu'")
        elif p.onde == "gpu+cpu":
            plano.avisos.append("sem medição: quanto vai para a GPU depende das flags; "
                                "confira o aviso de transbordo depois de subir")
    if p.medido is not None:
        precisa_ram = p.medido.ram_gb
    elif p.onde == "cpu":
        precisa_ram = tamanho_gb
    elif p.onde == "gpu+cpu":
        livre_gpu = estado.gpu.livre_gb if estado.gpu else 0.0
        precisa_ram = max(tamanho_gb - livre_gpu, 0.0) + RAM_BASE_GPU_GB
    else:
        precisa_ram = RAM_BASE_GPU_GB
    if precisa_ram + MARGEM_RAM_GB > estado.ram_total_gb:
        plano.cabe = False
        plano.motivos.append(f"o modelo precisa de ~{_gb(precisa_ram + MARGEM_RAM_GB)} de RAM "
                             f"e a máquina tem {_gb(estado.ram_total_gb)}")
    elif estado.ram_livre_gb < precisa_ram + MARGEM_RAM_GB:
        plano.avisos.append(f"pouca RAM livre ({_gb(estado.ram_livre_gb)} de ~{_gb(precisa_ram + MARGEM_RAM_GB)}): "
                            "os pesos usam mmap e o modelo pode ficar lento; feche programas se travar")
    return plano
