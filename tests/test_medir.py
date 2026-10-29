import json
from types import SimpleNamespace

import pytest

from local_hub_mcp import medir
from local_hub_mcp.medir import ErroMedir, aplicar, flags_bench, interpretar, melhor, montar_argv, rodar
from local_hub_mcp.perfis import Perfil


def reg(ncmoe, ub, p, n, ts):
    return {"n_cpu_moe": ncmoe, "n_ubatch": ub, "n_prompt": p, "n_gen": n, "avg_ts": ts, "model_type": "x"}


SAIDA = "log antes\n" + json.dumps([
    reg(30, 512, 512, 0, 100.0), reg(30, 512, 0, 64, 20.0),
    reg(32, 512, 512, 0, 110.0), reg(32, 512, 0, 64, 31.0),
    reg(32, 2048, 512, 0, 190.0), reg(32, 2048, 0, 64, 31.0),
])


def test_flags_bench_pula_o_que_o_bench_nao_entende():
    f = flags_bench(["-ngl", "999", "-c", "131072", "--jinja", "-ncmoe", "32", "-fa", "on", "--device", "none"])
    assert f == {"-ngl": "999", "-ncmoe": "32", "-fa": "on", "-dev": "none"}


def test_montar_argv(maquina):
    p = Perfil(alias="t", arquivo="tiny/tiny-Q4_K_M.gguf", onde="gpu+cpu", flags=["-ngl", "999", "-ncmoe", "32"])
    cmd = montar_argv(maquina["llama"] / "llama-bench", maquina["models"], p, ncmoe=[30, 32], ub=[512, 2048])
    assert cmd[cmd.index("-ncmoe") + 1] == "30,32" and cmd[cmd.index("-ub") + 1] == "512,2048"
    assert cmd[cmd.index("-o") + 1] == "json"
    cpu = montar_argv(maquina["llama"] / "llama-bench", maquina["models"],
                      Perfil(alias="c", arquivo="tiny/tiny-Q4_K_M.gguf", onde="cpu"))
    assert cpu[cpu.index("-dev") + 1] == "none"


def test_interpretar_e_melhor():
    linhas = interpretar(SAIDA)
    assert linhas[0] == {"ncmoe": 30, "ub": 512, "pp_ts": 100.0, "tg_ts": 20.0}
    assert melhor(linhas) == {"ncmoe": 32, "ub": 2048, "pp_ts": 190.0, "tg_ts": 31.0}


def test_interpretar_sem_json():
    with pytest.raises(ErroMedir):
        interpretar("erro: modelo não encontrado")


def test_aplicar_troca_ou_acrescenta():
    escolhido = {"ncmoe": 32, "ub": 2048}
    assert aplicar(["-ncmoe", "28", "-c", "4096"], escolhido, True, True) == ["-ncmoe", "32", "-c", "4096", "-ub", "2048"]
    assert aplicar(["-ub", "512"], escolhido, False, True) == ["-ub", "2048"]


def test_rodar_falha_quando_returncode_nao_zero_mesmo_com_json_no_stdout(monkeypatch):
    stderr = ("x" * 410 + "causa final").encode()
    monkeypatch.setattr(medir.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stdout=b"[", stderr=stderr))

    with pytest.raises(ErroMedir) as e:
        rodar(["llama-bench"])

    assert str(e.value) == "o llama-bench falhou: " + stderr.decode()[-400:]


def test_rodar_erro_sem_stderr_usa_fim_stdout(monkeypatch):
    stdout = ("x" * 410 + "causa no stdout").encode()
    monkeypatch.setattr(medir.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stdout=stdout, stderr=b""))
    with pytest.raises(ErroMedir) as e:
        rodar(["llama-bench"])
    assert str(e.value) == "o llama-bench falhou: " + stdout.decode()[-400:]
