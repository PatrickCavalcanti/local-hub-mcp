import json
import os

import pytest

from local_hub_mcp.perfis import (Catalogo, ErroPerfil, Medido, Perfil, Voz, argv, caminho_seguro,
                                  partes, tamanho_gb, validar)


def _perfil(**kw):
    base = dict(alias="tiny", arquivo="tiny/tiny-Q4_K_M.gguf", onde="gpu")
    base.update(kw)
    return Perfil(**base)


def test_ida_e_volta_dict():
    p = _perfil(voz=Voz(), medido=Medido(1.5, 0.5, 3.0, "2026-10-28"), capacidades=["llm.chat", "tts"])
    assert Perfil.de_dict(p.para_dict()) == p


def test_de_dict_recusa_campo_desconhecido():
    with pytest.raises(ErroPerfil, match="desconhecidos"):
        Perfil.de_dict({"alias": "a", "arquivo": "x.gguf", "onde": "gpu", "porta": 1})


def test_caminho_seguro_recusa_fuga(maquina):
    with pytest.raises(ErroPerfil):
        caminho_seguro(maquina["models"], "../fora.gguf")
    with pytest.raises(ErroPerfil):
        caminho_seguro(maquina["models"], str(maquina["models"] / "tiny" / "tiny-Q4_K_M.gguf"))
    assert caminho_seguro(maquina["models"], "tiny/tiny-Q4_K_M.gguf").name == "tiny-Q4_K_M.gguf"


def test_partes_de_arquivo_dividido(tmp_path):
    ps = partes(tmp_path / "M-Q4-00001-of-00003.gguf")
    assert [p.name for p in ps] == ["M-Q4-00001-of-00003.gguf", "M-Q4-00002-of-00003.gguf", "M-Q4-00003-of-00003.gguf"]
    assert partes(tmp_path / "m.gguf") == [tmp_path / "m.gguf"]


def test_validar(maquina):
    validar(_perfil(), maquina["models"])
    with pytest.raises(ErroPerfil, match="alias"):
        validar(_perfil(alias="Com Espaço"), maquina["models"])
    with pytest.raises(ErroPerfil, match="onde"):
        validar(_perfil(onde="tpu"), maquina["models"])
    with pytest.raises(ErroPerfil, match="capacidades"):
        validar(_perfil(capacidades=["voar"]), maquina["models"])
    with pytest.raises(ErroPerfil, match="--port"):
        validar(_perfil(flags=["--port", "1"]), maquina["models"])
    with pytest.raises(ErroPerfil, match="bad_alloc"):
        validar(_perfil(flags=["--no-mmap"]), maquina["models"])
    with pytest.raises(ErroPerfil, match="não existe"):
        validar(_perfil(arquivo="tiny/outro.gguf"), maquina["models"])
    with pytest.raises(ErroPerfil, match="voz"):
        validar(_perfil(capacidades=["tts"]), maquina["models"])


def test_argv_gpu_e_cpu(maquina):
    llama = maquina["llama"] / "llama-server"
    cmd = argv(_perfil(flags=["-c", "4096"], mmproj="tiny/mmproj-tiny-F16.gguf"), llama, maquina["models"], 8180, None)
    assert cmd[0] == str(llama)
    assert cmd[cmd.index("--alias") + 1] == "tiny"
    assert cmd[-4:] == ["--host", "127.0.0.1", "--port", "8180"]
    assert "--mmproj" in cmd and "--device" not in cmd
    cpu = argv(_perfil(onde="cpu", mmproj="tiny/mmproj-tiny-F16.gguf"), llama, maquina["models"], 8181, None)
    assert cpu[cpu.index("--device") + 1] == "none"
    assert "--no-mmproj-offload" in cpu


def test_argv_video_com_ffmpeg(maquina, tmp_path):
    cmd = argv(_perfil(video=True), maquina["llama"] / "llama-server", maquina["models"], 8180, tmp_path)
    assert cmd[cmd.index("--video-ffmpeg-dir") + 1] == str(tmp_path)
    sem = argv(_perfil(video=True), maquina["llama"] / "llama-server", maquina["models"], 8180, None)
    assert "--video-ffmpeg-dir" not in sem


def test_tamanho_soma_mmproj(maquina):
    gb = tamanho_gb(_perfil(mmproj="tiny/mmproj-tiny-F16.gguf"), maquina["models"])
    assert gb == pytest.approx(1536 / 1024 ** 3)


def test_catalogo_crud(maquina):
    arq = maquina["raiz"] / "perfis.json"
    cat = Catalogo(arq, maquina["models"])
    assert cat.listar() == []
    cat.criar(_perfil())
    with pytest.raises(ErroPerfil, match="já existe"):
        cat.criar(_perfil())
    novo = cat.editar("tiny", {"flags": ["-c", "2048"]})
    assert novo.flags == ["-c", "2048"]
    with pytest.raises(ErroPerfil, match="alias"):
        cat.editar("tiny", {"alias": "outro"})
    cat.gravar_medido("tiny", Medido(1.0, 0.2, 2.5, "2026-10-28"))
    assert cat.obter("tiny").medido.vram_gb == 1.0
    assert json.loads(arq.read_text(encoding="utf-8"))["perfis"][0]["alias"] == "tiny"
    cat.apagar("tiny")
    with pytest.raises(ErroPerfil, match="não existe"):
        cat.obter("tiny")


def test_editar_voz_mescla_campo_a_campo(maquina):
    cat = Catalogo(maquina["raiz"] / "perfis.json", maquina["models"])
    vozes = {"female": {}, "male": {"seed": 42, "tom": -1.5, "velocidade": 1.1}}
    cat.criar(_perfil(capacidades=["tts"], voz=Voz(vozes=vozes, pedaco_tts=200)))
    novo = cat.editar("tiny", {"voz": {"idioma": "en"}})
    assert novo.voz.idioma == "en"
    assert novo.voz.vozes == vozes and novo.voz.pedaco_tts == 200


def test_editar_flags_descarta_medicao(maquina):
    cat = Catalogo(maquina["raiz"] / "perfis.json", maquina["models"])
    cat.criar(_perfil(flags=["-c", "4096"], medido=Medido(1.0, 0.2, 2.5, "2026-10-28")))
    assert cat.editar("tiny", {"notas": "x"}).medido is not None
    assert cat.editar("tiny", {"flags": ["-c", "4096"]}).medido is not None
    assert cat.editar("tiny", {"flags": ["-c", "8192"]}).medido is None


def test_perfis_exemplo_carregam():
    from pathlib import Path
    arq = Path(__file__).resolve().parents[1] / "perfis.exemplo.json"
    dados = json.loads(arq.read_text(encoding="utf-8"))
    perfis = [Perfil.de_dict(d) for d in dados["perfis"]]
    assert {p.alias for p in perfis} == {"qwen3-4b", "gemma3-4b", "voz"}
    assert all("--" in p.arquivo.split("/")[0] for p in perfis)

@pytest.mark.parametrize("caminho", [
    pytest.param("\\x.gguf", marks=pytest.mark.skipif(os.name != "nt", reason="'\\' só é separador no Windows")),
    "/x.gguf",
    pytest.param("C:x.gguf", marks=pytest.mark.skipif(os.name != "nt", reason="drive só existe no Windows")),
])
def test_recusa_caminho_com_raiz_ou_drive(maquina, caminho):
    with pytest.raises(ErroPerfil):
        caminho_seguro(maquina["models"], caminho)


def test_catalogo_limpa_temporario_se_replace_falha(maquina, monkeypatch):
    def falhar(*args):
        raise PermissionError("bloqueado")
    monkeypatch.setattr("local_hub_mcp.perfis.os.replace", falhar)
    cat = Catalogo(maquina["raiz"] / "perfis.json", maquina["models"])
    with pytest.raises(ErroPerfil, match="não consegui gravar .*bloqueado .*arquivo está aberto"):
        cat.criar(_perfil())
    assert not list(maquina["raiz"].glob(".perfis-*.tmp"))


def test_voz_precisa_de_vozes(maquina):
    with pytest.raises(ErroPerfil, match="precisa de pelo menos uma voz"):
        validar(_perfil(voz=Voz(vozes={})), maquina["models"])



def test_alias_atual_e_reservado(maquina):
    with pytest.raises(ErroPerfil, match="reservado"):
        validar(_perfil(alias="atual"), maquina["models"])


def test_voz_sem_numeros_por_extenso_usa_padrao_desligado():
    from local_hub_mcp.perfis import Voz
    assert Voz().numeros_por_extenso is False
    p = Perfil.de_dict({"alias": "v", "arquivo": "a/b.gguf", "onde": "cpu",
                        "voz": {"vozes": {"female": {}}, "idioma": "pt"}})
    assert p.voz.numeros_por_extenso is False
