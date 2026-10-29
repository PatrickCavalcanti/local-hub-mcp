from local_hub_mcp.recursos import GB, parse_nvidia_smi, parse_typeperf, parse_versao

TYPEPERF = (
    '\r\n'
    '"(PDH-CSV 4.0)","\\\\PC\\GPU Process Memory(pid_100_luid_0x0_0x1_phys_0)\\Dedicated Usage",'
    '"\\\\PC\\GPU Process Memory(pid_100_luid_0x0_0x2_phys_0)\\Dedicated Usage",'
    '"\\\\PC\\GPU Process Memory(pid_100_luid_0x0_0x1_phys_0)\\Shared Usage",'
    '"\\\\PC\\GPU Process Memory(pid_200_luid_0x0_0x1_phys_0)\\Shared Usage"\r\n'
    f'"10/28/2026 20:56:22.651","{2 * GB:.6f}","{1 * GB:.6f}","{0.5 * GB:.6f}","0.000000"\r\n'
    'Saindo, por favor aguarde...\r\n'
)


def test_parse_typeperf_soma_por_pid():
    m = parse_typeperf(TYPEPERF)
    assert round(m[100].dedicada_gb, 3) == 2.0
    assert round(m[100].compartilhada_gb, 3) == 0.5
    assert m[200].dedicada_gb == 0.0


def test_parse_typeperf_vazio():
    assert parse_typeperf("") == {}


def test_parse_nvidia_smi():
    g = parse_nvidia_smi("6144, 4209\n")
    assert g.total_gb == 6.0
    assert round(g.livre_gb, 2) == 4.11
    assert parse_nvidia_smi("NVIDIA-SMI has failed") is None


def test_parse_versao():
    texto = "ggml_cuda_init: ...\nversion: 0.8.0-dev (build 11968, commit ac8e1694d)\nbuilt with Clang"
    assert parse_versao(texto) == "0.8.0-dev (build 11968, commit ac8e1694d)"

def test_parse_typeperf_seleciona_adapter_com_mais_dedicada():
    import csv
    import io
    dados = io.StringIO()
    csv.writer(dados, quoting=csv.QUOTE_ALL).writerows([
        ["data", *[
            f"GPU Process Memory(pid_300_luid_{luid}_phys_0)\\{tipo} Usage"
            for luid in ("0x0_0x1", "0x0_0x2")
            for tipo in ("Dedicated", "Shared")
        ]],
        ["agora", GB, 0.1 * GB, 0, 0.6 * GB],
    ])
    m = parse_typeperf(dados.getvalue())[300]
    assert m.dedicada_gb == 1.0
    assert m.compartilhada_gb == 0.1
