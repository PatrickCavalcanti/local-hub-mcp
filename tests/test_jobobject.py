import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import psutil
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Job Object só existe no Windows")


def test_matar_o_servico_mata_o_filho(tmp_path):
    script = tmp_path / "servico.py"
    script.write_text(textwrap.dedent(f"""
        import subprocess, sys, time
        sys.path.insert(0, {str(SRC)!r})
        from local_hub_mcp import jobobject
        modo = jobobject.entrar()
        filho = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        jobobject.apos_criar(filho.pid)
        print(modo, filho.pid, flush=True)
        time.sleep(120)
    """), encoding="utf-8")
    servico = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    modo, pid_filho = servico.stdout.readline().split()
    assert modo in ("proprio", "por_filho")
    assert psutil.pid_exists(int(pid_filho))
    servico.kill()   # TerminateProcess, como o Gerenciador de Tarefas
    servico.wait(10)
    for _ in range(50):
        if not psutil.pid_exists(int(pid_filho)):
            break
        time.sleep(0.1)
    assert not psutil.pid_exists(int(pid_filho)), "o filho ficou órfão"
