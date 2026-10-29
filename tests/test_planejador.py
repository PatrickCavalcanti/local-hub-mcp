import pytest

from local_hub_mcp.perfis import Medido, Perfil
from local_hub_mcp.planejador import NoAr, planejar
from local_hub_mcp.recursos import EstadoRecursos, Gpu


def P(onde="gpu", medido=None, alias="m"):
    return Perfil(alias=alias, arquivo="m/m.gguf", onde=onde, medido=medido)


def E(livre_gpu=4.0, ram=20.0, gpu=True):
    return EstadoRecursos(gpu=Gpu(6.0, livre_gpu) if gpu else None, ram_total_gb=32.0, ram_livre_gb=ram)


@pytest.mark.parametrize("perfil, tamanho, estado, no_ar, cabe, trecho", [
    (P("gpu"), 2.0, E(4.0), [], True, None),
    (P("gpu"), 3.9, E(4.0), [], False, "gpu+cpu"),
    (P("gpu"), 1.0, E(gpu=False), [], False, "nenhuma GPU"),
    (P("gpu"), 1.0, E(4.0), [NoAr("outro", "gpu")], False, "derrube antes"),
    (P("gpu"), 1.0, E(4.0), [NoAr("voz", "cpu")], True, None),
    (P("gpu", Medido(3.3, 0.6, 5, "x")), 1.1, E(3.5), [], False, "VRAM livre"),
    (P("gpu", Medido(3.3, 0.6, 5, "x")), 1.1, E(3.7), [NoAr("voz", "cpu")], True, None),
    (P("gpu+cpu"), 15.4, E(4.0, ram=20.0), [], True, None),
    (P("gpu+cpu"), 15.4, E(4.0, ram=10.0), [], True, None),
    (P("cpu"), 2.0, E(gpu=False, ram=10.0), [NoAr("qwen", "gpu+cpu")], True, None),
    (P("cpu"), 9.5, E(ram=10.0), [], True, None),
    (P("cpu"), 40.0, E(ram=10.0), [], False, "a máquina tem"),
    (P("gpu+cpu"), 15.4, E(0.5), [], False, "VRAM livre"),
])
def test_tabela(perfil, tamanho, estado, no_ar, cabe, trecho):
    plano = planejar(perfil, tamanho, estado, no_ar)
    assert plano.cabe is cabe, plano.motivos
    if trecho:
        assert any(trecho in m for m in plano.motivos), plano.motivos


def test_gpu_cpu_sem_medicao_avisa():
    plano = planejar(P("gpu+cpu"), 15.4, E(4.0, ram=20.0), [])
    assert plano.avisos and "transbordo" in plano.avisos[0]


def test_o_proprio_alias_no_ar_nao_bloqueia():
    assert planejar(P("gpu", alias="m"), 1.0, E(4.0), [NoAr("m", "gpu")]).cabe

@pytest.mark.parametrize("perfil,tamanho", [(P("gpu+cpu"), 15.4), (P("cpu"), 9.5)])
def test_pouca_ram_livre_avisa(perfil, tamanho):
    plano = planejar(perfil, tamanho, E(4.0, ram=10.0), [])
    assert plano.cabe
    assert any("pouca RAM livre" in a and "mmap" in a for a in plano.avisos)
