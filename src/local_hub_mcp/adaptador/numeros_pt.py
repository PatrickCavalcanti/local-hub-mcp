"""Numeros por extenso (pt-BR) antes do TTS.

O LFM3 le numero escrito em digito errado ("29 de outubro de 2026" vira "2006";
"R$ 1.250,50" vira lixo). Escrito por extenso ele le certo (medido em 28/10/2026).
Este modulo troca, no texto que vai para a voz: dinheiro, porcentagem, data, hora,
ordinal (1o, 2a), decimal e inteiro. Numero colado em letra (gpt-4o, v2, mp3) fica
como esta.

Uso:  from numeros_pt import por_extenso;  por_extenso("Sao 3 tarefas")
Teste: python numeros_pt.py
"""
import re

UNIDADES = ["zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez",
            "onze", "doze", "treze", "quatorze", "quinze", "dezesseis", "dezessete", "dezoito", "dezenove"]
DEZENAS = ["", "", "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta", "oitenta", "noventa"]
CENTENAS = ["", "cento", "duzentos", "trezentos", "quatrocentos", "quinhentos", "seiscentos", "setecentos",
            "oitocentos", "novecentos"]
MESES = ["", "janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro",
         "outubro", "novembro", "dezembro"]
ORDINAIS = {1: "primeir", 2: "segund", 3: "terceir", 4: "quart", 5: "quint", 6: "sext", 7: "sétim",
            8: "oitav", 9: "non", 10: "décim"}
# substantivos femininos comuns depois de numero: "2 horas" -> "duas horas"
FEMININOS = {"hora", "horas", "pessoa", "pessoas", "vez", "vezes", "semana", "semanas", "mensagem",
             "mensagens", "tarefa", "tarefas", "reunião", "reuniões", "coisa", "coisas", "página", "páginas",
             "linha", "linhas", "sessão", "sessões", "nota", "notas", "conta", "contas", "opção", "opções",
             "etapa", "etapas", "fase", "fases", "pendência", "pendências", "ligação", "ligações"}


def _feminino(txt):
    return (txt.replace("um", "uma", 1) if txt.endswith("um") else txt) \
        .replace("dois", "duas").replace("entos", "entas")


def _ate_mil(n):
    """1..999"""
    if n == 100:
        return "cem"
    c, r = divmod(n, 100)
    partes = [CENTENAS[c]] if c else []
    if r:
        if r < 20:
            partes.append(UNIDADES[r])
        else:
            d, u = divmod(r, 10)
            partes.append(DEZENAS[d] + (" e " + UNIDADES[u] if u else ""))
    return " e ".join(partes)


def cardinal(n, feminino=False):
    if n < 0:
        return "menos " + cardinal(-n, feminino)
    if n < 20:
        txt = UNIDADES[n]
        return _feminino(txt) if feminino and n in (1, 2) else txt
    grupos = []                                   # [(valor, escala)] do maior para o menor
    for escala in (10 ** 12, 10 ** 9, 10 ** 6, 10 ** 3, 1):
        v, n = divmod(n, escala)
        if v:
            grupos.append((v, escala))
    nomes = {10 ** 12: ("trilhão", "trilhões"), 10 ** 9: ("bilhão", "bilhões"), 10 ** 6: ("milhão", "milhões")}
    partes = []
    for v, escala in grupos:
        txt = _ate_mil(v)
        if escala == 1 and feminino:
            txt = _feminino(txt)
        if escala == 10 ** 3:
            txt = "mil" if v == 1 else (_feminino(txt) if feminino else txt) + " mil"
        elif escala > 10 ** 3:
            txt = f"{txt} {nomes[escala][0] if v == 1 else nomes[escala][1]}"
        partes.append((v, escala, txt))
    out = partes[0][2]
    for v, escala, txt in partes[1:]:
        # "e" antes do ultimo grupo quando ele e < 100 ou centena redonda: mil e vinte, mil e duzentos
        ultimo = (v, escala) == grupos[-1][:2]
        out += (" e " if ultimo and (v < 100 or v % 100 == 0) else " ") + txt
    return out


def ordinal(n, fem=False):
    if n in ORDINAIS:
        return ORDINAIS[n] + ("a" if fem else "o")
    return cardinal(n, fem)


def _inteiro(s):
    return int(s.replace(".", ""))


def _decimal(inteiro, frac, feminino=False):
    txt = cardinal(_inteiro(inteiro), feminino)
    if frac is None:
        return txt
    # "1,5" -> "um vírgula cinco"; "1,05" -> "um vírgula zero cinco"
    if frac.startswith("0"):
        return txt + " vírgula " + " ".join(UNIDADES[int(d)] for d in frac)
    return txt + " vírgula " + cardinal(int(frac))


NUM = r"\d{1,3}(?:\.\d{3})+|\d+"            # 1.250.000 ou 1250
SEM_LETRA_ANTES = r"(?<![\w.,/:-])"
SEM_LETRA_DEPOIS = r"(?![\w%º°ª]|[.,:/-]\d)"   # 1.2.3 e 10-20 ficam como estao


def _dinheiro(m):
    reais, cent = _inteiro(m.group(1)), int((m.group(2) or "0").ljust(2, "0")[:2])
    partes = []
    if reais:
        partes.append(cardinal(reais) + (" de" if reais >= 10 ** 6 and reais % 10 ** 6 == 0 else "")
                      + (" real" if reais == 1 else " reais"))
    if cent:
        partes.append(cardinal(cent) + (" centavo" if cent == 1 else " centavos"))
    return " e ".join(partes) or "zero reais"


def _data(m):
    d, mes, ano = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= d <= 31 and 1 <= mes <= 12):
        return m.group(0)
    txt = ("primeiro" if d == 1 else cardinal(d)) + " de " + MESES[mes]
    if ano:
        a = int(ano)
        a = a + 2000 if len(ano) == 2 else a
        txt += " de " + cardinal(a)
    return txt


def _hora(m):
    h, mi = int(m.group(1)), int(m.group(2) or 0)
    if h > 24 or mi > 59:
        return m.group(0)
    hs = cardinal(h, feminino=True)
    if mi == 0:
        return hs + (" hora" if h == 1 else " horas")
    return f"{hs} e {cardinal(mi)}"


def por_extenso(texto):
    t = texto
    # dinheiro: R$ 1.250,50 · R$ 10 · R$0,99
    t = re.sub(r"R\$\s?(" + NUM + r")(?:,(\d{1,2}))?" + SEM_LETRA_DEPOIS, _dinheiro, t)
    # porcentagem: 15% · 2,5 %
    t = re.sub(SEM_LETRA_ANTES + r"(" + NUM + r")(?:,(\d+))?\s?%",
               lambda m: _decimal(m.group(1), m.group(2)) + " por cento", t)
    # data ISO: 2026-10-29
    t = re.sub(SEM_LETRA_ANTES + r"(\d{4})-(\d{2})-(\d{2})(?![\d-])",
               lambda m: _data(re.match(r"(\d+)/(\d+)/(\d+)", f"{m.group(3)}/{m.group(2)}/{m.group(1)}")), t)
    # negativo: " -5" -> "menos cinco"
    t = re.sub(r"(?<![\w.,/-])-(?=\d)", "menos ", t)
    # intervalo: 10-20 -> 10 a 20 (os numeros viram extenso logo abaixo)
    t = re.sub(SEM_LETRA_ANTES + r"(\d+)\s?-\s?(\d+)(?![\d/-])", r"\1 a \2", t)
    # data: 29/10/2026 · 29/10/26 · 29/10
    t = re.sub(SEM_LETRA_ANTES + r"(\d{1,2})/(\d{1,2})(?:/(\d{4}|\d{2}))?(?![\d/])", _data, t)
    # "29 de outubro" e "dia 1": dia 1 -> dia primeiro
    t = re.sub(r"\b(dia|Dia)\s1\b(?![,.]\d)", r"\1 primeiro", t)
    t = re.sub(r"(?<![\d,.])\b1\s(?=de\s(?:" + "|".join(MESES[1:]) + r")\b)", "primeiro ", t)
    # hora: 14h30 · 14:30 · 9h · 18 h
    t = re.sub(SEM_LETRA_ANTES + r"(\d{1,2})\s?h(?:(\d{2})(?:min)?)?\b", _hora, t)
    t = re.sub(SEM_LETRA_ANTES + r"(\d{1,2}):(\d{2})(?!:\d)\b", _hora, t)
    # ordinal: 1º 2ª 3o
    t = re.sub(SEM_LETRA_ANTES + r"(\d+)\s?([ºª°])",
               lambda m: ordinal(int(m.group(1)), m.group(2) == "ª"), t)
    # sequencia longa de digitos (telefone, codigo): digito a digito
    t = re.sub(SEM_LETRA_ANTES + r"(\d{7,})" + SEM_LETRA_DEPOIS,
               lambda m: " ".join(UNIDADES[int(d)] for d in m.group(1)), t)

    # decimal e inteiro; feminino quando a palavra seguinte pede ("2 horas", "1 pessoa")
    def _numero(m):
        prox = (m.group(3) or "").lower()
        return _decimal(m.group(1), m.group(2), feminino=prox in FEMININOS) + (m.group(3) and " " + m.group(3) or "")
    t = re.sub(SEM_LETRA_ANTES + r"(" + NUM + r")(?:,(\d+))?" + SEM_LETRA_DEPOIS + r"(?:\s(\w+))?", _numero, t)
    return t


if __name__ == "__main__":
    casos = {
        "Você tem 3 compromissos amanhã, dia 29 de outubro de 2026, e o primeiro custa R$ 1.250,50 para remarcar.":
            "Você tem três compromissos amanhã, dia vinte e nove de outubro de dois mil e vinte e seis, e o primeiro custa mil duzentos e cinquenta reais e cinquenta centavos para remarcar.",
        "A reunião é às 14h30 e dura 2 horas.": "A reunião é às quatorze e trinta e dura duas horas.",
        "Às 9h tem café; às 18:00 acaba.": "Às nove horas tem café; às dezoito horas acaba.",
        "Prazo: 05/11/2026.": "Prazo: cinco de novembro de dois mil e vinte e seis.",
        "Desde 1/11.": "Desde primeiro de novembro.",
        "Subiu 15% e depois 2,5%.": "Subiu quinze por cento e depois dois vírgula cinco por cento.",
        "Custa R$ 1,00 ou R$ 0,50.": "Custa um real ou cinquenta centavos.",
        "São 1.000.000 de linhas.": "São um milhão de linhas.",
        "Ano 2000, 1999 e 1001.": "Ano dois mil, mil novecentos e noventa e nove e mil e um.",
        "Tem 1 pessoa e 1 projeto.": "Tem uma pessoa e um projeto.",
        "Use o gpt-4o e o mp3 da v2.": "Use o gpt-4o e o mp3 da v2.",
        "É o 1º lugar e a 2ª vez.": "É o primeiro lugar e a segunda vez.",
        "Temperatura de 23,5 graus.": "Temperatura de vinte e três vírgula cinco graus.",
        "Ligue para 67999887766.": "Ligue para seis sete nove nove nove oito oito sete sete seis seis.",
        "Faltam 200 páginas e 101 testes.": "Faltam duzentas páginas e cento e um testes.",
        "O dia 1 é sábado.": "O dia primeiro é sábado.",
        "Entre 10-20 itens, em 2026-10-28, faltando -5 graus.":
            "Entre dez a vinte itens, em vinte e oito de outubro de dois mil e vinte e seis, faltando menos cinco graus.",
        "Versão 1.2.3 do app.": "Versão 1.2.3 do app.",
        "São 2 mil reais e R$ 2.500.000.": "São dois mil reais e dois milhões e quinhentos mil reais.",
    }
    falhas = 0
    for entrada, esperado in casos.items():
        saida = por_extenso(entrada)
        ok = saida == esperado
        falhas += not ok
        print(("OK  " if ok else "ERRO") + " | " + entrada + ("" if ok else f"\n     saiu:     {saida}\n     esperado: {esperado}"))
    print(f"{len(casos) - falhas}/{len(casos)} ok")
    raise SystemExit(1 if falhas else 0)
