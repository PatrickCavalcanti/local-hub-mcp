import os
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

EXE = ".exe" if os.name == "nt" else ""


@pytest.fixture(autouse=True, scope="session")
def job_object_da_sessao():
    if os.name == "nt":
        from local_hub_mcp import jobobject

        jobobject.entrar()


@pytest.fixture
def maquina(tmp_path):
    """llama_cpp_dir e models_dir de mentira, com um GGUF vazio."""
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / f"llama-server{EXE}").write_bytes(b"")
    (llama / f"llama-bench{EXE}").write_bytes(b"")
    modelos = tmp_path / "models"
    (modelos / "tiny").mkdir(parents=True)
    (modelos / "tiny" / "tiny-Q4_K_M.gguf").write_bytes(b"x" * 1024)
    (modelos / "tiny" / "mmproj-tiny-F16.gguf").write_bytes(b"x" * 512)
    return {"raiz": tmp_path, "llama": llama, "models": modelos}
