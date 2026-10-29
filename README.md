# local-hub-mcp

**English:** [README.en.md](README.en.md)

Nomes de ferramentas, campos de config e mensagens do serviço ficam em português; este README explica cada um.

## O que é

Uma conexão para usar e gerenciar modelos GGUF locais com o llama.cpp.
O gateway compatível com OpenAI (`/v1`) recebe as conversas.
As ferramentas MCP (`/mcp`) baixam, configuram, medem, sobem e derrubam os modelos.
O serviço usa o llama.cpp que você instala; não instala os binários nem baixa modelos no boot.

```text
Cliente ──→ :8079/v1  ──→ llama-server por perfil
        └─→ :8079/mcp ──→ gerenciamento dos perfis e processos
```

## Plataformas

| Plataforma | Suporte e limites |
|---|---|
| Windows | Suporte completo: Job Object para encerrar os processos filhos e contadores de memória dedicada/compartilhada para detectar transbordo de VRAM. |
| Linux/macOS | Sem Job Object e sem medida de transbordo de VRAM. Matar o serviço à força pode deixar `llama-server` vivo. |
| GPU | NVIDIA detectada por `nvidia-smi`. Sem NVIDIA detectável, use perfis com `onde: "cpu"`. |

Para usar GPU NVIDIA, instale uma build do llama.cpp com backend CUDA. A presença da GPU não basta: uma build sem CUDA pode executar o modelo na CPU e ficar lenta. O serviço verifica a presença do `llama-server` e consulta a GPU via `nvidia-smi`; não valida o backend CUDA da build. Depois de subir um perfil, use `logs(alias="qwen3-4b")` e confira se o log mostra o dispositivo CUDA em uso.

## Requisitos

- Python **3.12 ou superior** e [uv](https://docs.astral.sh/uv/).
- Git para clonar o repositório.
- [llama.cpp](https://github.com/ggml-org/llama.cpp/releases): uma pasta com `llama-server` e `llama-bench` (`.exe` no Windows). No Windows com NVIDIA, escolha o zip `...-cuda-...` e mantenha os arquivos da distribuição juntos. No Linux, compile com `-DGGML_CUDA=ON` ou use uma build CUDA. Sem GPU NVIDIA, use uma build para CPU.
- Qualquer cliente com MCP por Streamable HTTP — por exemplo Claude Code, Claude Desktop, Cursor ou VS Code — para chamar as ferramentas de gerenciamento (veja [Ligar um cliente](#ligar-um-cliente)).
- [ffmpeg](https://ffmpeg.org/download.html), opcional: necessário para voz e anexos de áudio/vídeo. Instale no Windows com `winget install Gyan.FFmpeg`, no Linux com `sudo apt install ffmpeg` ou no macOS com `brew install ffmpeg`.
- Conta do Hugging Face, opcional: disponibilize `HF_TOKEN` no ambiente do serviço para acessar repositórios fechados aos quais sua conta tem acesso.

## Instalação passo a passo

### 1. Clonar e instalar as dependências

Nos dois sistemas:

```sh
git clone https://github.com/PatrickCavalcanti/local-hub-mcp.git
cd local-hub-mcp
uv sync
```

### 2. Criar a configuração

Windows PowerShell:

```powershell
copy config.exemplo.json config.json
New-Item -ItemType Directory -Force models
```

Linux/macOS:

```sh
cp config.exemplo.json config.json
mkdir -p models
```

Edite `config.json`. Aponte `llama_cpp_dir` para a pasta dos binários do llama.cpp e `models_dir` para uma pasta existente que receberá os GGUF. Para usar a pasta criada acima, defina `"models_dir": "models"`. Caminhos relativos partem da pasta de `config.json`. No Windows, use `/` nos caminhos JSON ou escape cada `\` como `\\`.

Mantenha `bind: "127.0.0.1"` para uso local. Não coloque a porta do gateway dentro de `portas_internas`.

### 3. Iniciar o serviço

Windows PowerShell:

```powershell
.\iniciar.bat
```

Linux/macOS:

```sh
sh iniciar.sh
```

Ou, nos dois sistemas:

```sh
uv run local-hub-mcp --config config.json
```

Na primeira execução sem `perfis.json`, o serviço copia `perfis.exemplo.json` da pasta do `config.json` (por isso a config fica na pasta do projeto). São três receitas de exemplo: `qwen3-4b`, `gemma3-4b` e `voz`. Os pesos ainda precisam ser baixados.

A tela mostra a versão do llama.cpp, os endereços `/v1` e `/mcp`, a quantidade de perfis, o ffmpeg encontrado e o modo de Job Object. Sem ffmpeg, informa que voz e anexos de áudio/vídeo estão desligados. Deixe o terminal aberto; use Ctrl+C para encerrar normalmente.

### 4. Baixar e subir o primeiro modelo

Com o serviço rodando, conecte um cliente MCP conforme a próxima seção. Os exemplos abaixo são **chamadas de ferramentas MCP**, feitas pelo cliente; não são comandos de terminal nem uma API Python.

Para usar o perfil de chat `qwen3-4b`, peça ao cliente que execute:

```text
baixar_modelo(
  repo_id="unsloth/Qwen3-4B-Instruct-2507-GGUF",
  arquivos=["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"]
)
```

A ferramenta devolve uma tarefa. Consulte `tarefa(id="<id devolvido>")` até concluir com sucesso. O arquivo fica em `models_dir/unsloth--qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, caminho já usado pelo perfil de exemplo.

O perfil vem com `onde: "gpu"`. Sem NVIDIA, antes de subir use `editar_perfil(alias="qwen3-4b", onde="cpu", flags=["-ngl", "0", "-c", "4096", "-np", "1", "--jinja"])`. Com GPU, confira a build CUDA e a memória disponível com `status()`.

Depois execute `subir_modelo(alias="qwen3-4b")`. Se a carga virar tarefa, acompanhe o ID devolvido. Confira `status()` e `logs(alias="qwen3-4b")`; só então faça a primeira conversa pelo `/v1`. Para encerrar o modelo, use `derrubar_modelo(alias="qwen3-4b")`.

Alternativa: baixe o GGUF à mão e coloque-o no caminho exato indicado pelo campo `arquivo` do perfil, dentro de `models_dir`. Para outro caminho ou modelo, use `criar_perfil` ou `editar_perfil`. Modelos com visão ou voz podem precisar também do arquivo `mmproj`; veja as notas de `perfis.exemplo.json`. Os exemplos são pontos de partida, não garantia de que cabem na sua máquina. Use `medir` para comparar flags.

## Ligar um cliente

### MCP

Os exemplos usam a porta padrão `8079`. Se você mudou `porta` no `config.json`, troque `8079` pela sua porta em todos eles.

O gerenciamento usa **Streamable HTTP** em `http://127.0.0.1:8079/mcp`. O serviço devolve uma `tarefa` em até `espera_s` (90 s por padrão); o timeout de chamada só precisa ser maior que esse intervalo, por exemplo **120 segundos**.

Exemplo com Claude Code:

```sh
claude mcp add --transport http local-hub http://127.0.0.1:8079/mcp
```

Se configurar `chaves`, acrescente `--header "Authorization: Bearer <admin>"` ao comando acima, substituindo `<admin>` por `chaves.admin`.

No Claude Code, configure a variável de ambiente [`MCP_TOOL_TIMEOUT`](https://code.claude.com/docs/en/env-vars) em milissegundos (ex.: `MCP_TOOL_TIMEOUT=120000`). Em outros clientes, ajuste o timeout na configuração do servidor MCP de cada um.

Configuração JSON genérica (a forma de registrar pode variar conforme o cliente):

```json
{
  "mcpServers": {
    "local-hub": {
      "type": "http",
      "url": "http://127.0.0.1:8079/mcp"
    }
  }
}
```

Se configurar `chaves`, envie o header `Authorization: Bearer <admin>` nas chamadas MCP, substituindo `<admin>` pelo valor de `chaves.admin`. Teste a conexão chamando `status` e `listar_modelos`.

Para scripts, o SDK `mcp` já vem nas dependências do projeto. Exemplo mínimo em Python (salve como `cliente.py` e rode com `uv run python cliente.py` na pasta do projeto):

```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def main():
    async with streamable_http_client("http://127.0.0.1:8079/mcp") as (leitura, escrita):
        async with ClientSession(leitura, escrita) as sessao:
            await sessao.initialize()
            r = await sessao.call_tool("status", {})
            print(r.content[0].text)  # o serviço responde com um texto JSON

asyncio.run(main())
```

### OpenAI

Configure `base_url = http://127.0.0.1:8079/v1`, `api_key` com qualquer texto se não houver `chaves`, ou com o valor de `chaves.inference` se houver. Use `model` igual ao alias do perfil ou ao coringa `atual`.

Com o perfil no ar, Linux/macOS:

```sh
curl http://127.0.0.1:8079/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer local' \
  -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"Olá!"}]}'
```

Windows PowerShell:

```powershell
$headers = @{ Authorization = 'Bearer local' }
$body = @{ model = 'qwen3-4b'; messages = @(@{ role = 'user'; content = 'Olá!' }) } | ConvertTo-Json -Depth 4
Invoke-RestMethod -Uri http://127.0.0.1:8079/v1/chat/completions -Method Post -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

Troque `local` pela chave `inference` se houver autenticação. `atual` seleciona o último perfil que subiu compatível com a rota: chat para `/v1/responses` e `/v1/chat/completions`, STT para `/v1/audio/transcriptions`, TTS para `/v1/audio/speech` e embedding para `/v1/embeddings`. `GET /v1/models/atual` informa `aponta_para`. Declare no cliente apenas capacidades que os modelos usados com esse coringa oferecem.

Para carregar um modelo ao iniciar o serviço, inclua seu alias em `subir_no_boot`. O cliente ainda precisa esperar a carga terminar.

**Exemplo: Igris.** Configure o provider com `protocol: openai`, `base_url: http://127.0.0.1:8079/v1`, `api_style: responses` e `model_id` igual ao alias ou `atual`. Cadastre o MCP no endereço acima com `call_timeout_ms` ≥ 120000. Com `chaves`, use `inference` no provider e `Authorization: Bearer <admin>` no MCP.

## Referência

### Ferramentas MCP

| Ferramenta e parâmetros | O que faz |
|---|---|
| `status()` | Processos, falhas recentes de subida, recursos livres, versão, ffmpeg e tarefas. |
| `listar_modelos()` | Perfis e GGUF sem perfil em `models_dir`. |
| `procurar_modelo(texto, limite=10, ordem="trending")` | Busca originais e conversões GGUF no Hugging Face. Ordem: `trending`, `downloads`, `recentes` ou `likes`. |
| `conversoes_gguf(repo_id, ordem="downloads", limite=20)` | Lista conversões ligadas ao original por `base_model`. |
| `detalhes_repo(repo_id)` | Lista quants, tamanhos, partes, mmproj e estimativa de onde cabe. |
| `baixar_modelo(repo_id, arquivos)` | Baixa a lista de arquivos e todas as partes de GGUF divididos; devolve tarefa. |
| `criar_perfil(alias, arquivo, onde, ...)` | Cria receita; aceita `flags`, `capacidades`, `mmproj`, `anexos`, `video`, `voz` e `notas`. |
| `editar_perfil(alias, ...)` | Altera os mesmos campos opcionais; só os informados mudam. Flags substituem a lista inteira; `voz` altera só as chaves informadas; `mmproj=""` remove o projetor. Vale na próxima subida. |
| `apagar_perfil(alias)` | Remove o perfil, não os pesos. Recusa perfil no ar. |
| `apagar_arquivos(caminhos, confirmacao=null)` | Planeja exclusão dentro de `models_dir`; mostre a lista ao usuário e, só após seu OK, repita com o código em `confirmacao`. Recusa arquivos usados por perfil. |
| `subir_modelo(alias)` | Verifica VRAM/RAM e inicia `llama-server`; carga longa vira tarefa. Sem `medido`, recusa um segundo perfil de GPU (`gpu` ou `gpu+cpu`) enquanto outro de GPU está no ar: derrube o primeiro. Perfis `cpu` podem conviver, respeitando a RAM disponível. |
| `derrubar_modelo(alias)` | Encerra o processo e confere liberação de porta e VRAM. |
| `medir(alias, ncmoe=null, ub=null, gravar=false)` | Compara `-ncmoe`/`-ub` via `llama-bench`; `gravar=true` salva as melhores flags. Exige GPU livre. |
| `testar_modelo(alias, capacidades=null)` | Testes curtos de texto, visão, TTS e STT contra perfil no ar. |
| `logs(alias, linhas=100)` | Últimas linhas do log do perfil (de 1 a 2000 linhas). |
| `info_conexao()` | URLs, aliases, capacidades e vozes para configurar o cliente. |
| `tarefa(id)` | Estado e resultado: `em_andamento`, `ok`, `erro` ou `cancelada`; progresso do download. |

Confira `ok` e eventuais erros nas respostas. Os nomes e parâmetros das ferramentas continuam em português.

### `config.json`

| Campo | Padrão | Uso |
|---|---|---|
| `llama_cpp_dir` | obrigatório | Pasta dos binários do llama.cpp. O boot verifica `llama-server`; `medir` usa `llama-bench`. |
| `models_dir` | obrigatório | Pasta existente dos pesos; downloads e caminhos de modelos dos perfis ficam dentro dela. |
| `ffmpeg_dir` | `null` | Pasta do ffmpeg; sem valor, procura no PATH e na pasta do winget. |
| `bind` / `porta` | `127.0.0.1` / `8079` | Endereço do gateway. Bind fora do local exige as duas chaves. |
| `portas_internas` | `[8180, 8199]` | Faixa inclusiva das portas dos processos `llama-server`. |
| `chaves` | `{}` | Objeto com `admin` e `inference`; veja Segurança. |
| `perfis` | `"perfis.json"` | Arquivo do catálogo; não é a pasta dos pesos. |
| `logs_dir` | `"logs"` | Logs dos perfis e `auditoria.log`. |
| `load_timeout_s` | `600` | Limite de espera pela carga de um modelo, em segundos. |
| `espera_s` | `90` | Espera de uma ferramenta longa antes de devolver tarefa. |
| `subir_no_boot` | `[]` | Aliases carregados em sequência ao iniciar; falhas ficam nos logs e no status. |

### Formato do perfil

`perfis.json` contém um objeto com a lista `perfis`. Exemplo mínimo de estrutura:

```json
{
  "perfis": [
    {
      "alias": "meu-chat",
      "arquivo": "meu-modelo/modelo.gguf",
      "onde": "cpu",
      "flags": ["-ngl", "0", "-c", "4096"],
      "capacidades": ["llm.chat"]
    }
  ]
}
```

Substitua `arquivo` pelo caminho real antes de subir. Prefira editar o catálogo pelas ferramentas MCP.

| Campo | Padrão / regra |
|---|---|
| `alias` | Obrigatório; até 64 caracteres, começa com letra minúscula ou número; aceita também `.`, `_`, `-`. `atual` é reservado. |
| `arquivo` | Obrigatório; GGUF relativo a `models_dir`. Para modelo dividido, indique a primeira parte. |
| `onde` | Obrigatório: `gpu`, `gpu+cpu` ou `cpu`. As flags controlam o offload do llama.cpp; este campo orienta o planejamento de recursos. |
| `flags` | `[]`; cada token separado. Porta, host, arquivo, mmproj, alias e `--video-ffmpeg-dir` são controlados pelo serviço. `--no-mmap` e `--mlock` são proibidos. |
| `capacidades` | `["llm.chat"]`; valores permitidos abaixo. |
| `mmproj` | `null`; projetor relativo a `models_dir`. |
| `anexos` / `video` | `false` / `false`; opções do adaptador de anexos e vídeo. |
| `voz` | `null`; obrigatório quando `capacidades` inclui `stt` ou `tts`. Se presente, `vozes` precisa conter pelo menos uma voz. |
| `notas` | `""`; descrição livre. |
| `medido` | `null`; medição com `vram_gb`, `ram_gb`, `carga_s` e `em`. Mudanças em arquivo, local de execução ou flags descartam a medição. |

Capacidades permitidas: `llm.chat`, `llm.vision`, `llm.fast`, `embedding`, `stt`, `tts`, `text.generate`, `audio.in`, `video.in`. Declarar uma capacidade não a acrescenta ao modelo.

Bloco `voz`, com os padrões:

```json
{
  "vozes": {"female": {}, "male": {}},
  "idioma": "pt",
  "pedaco_tts": 250,
  "pedaco_stt_s": 45.0,
  "numeros_por_extenso": true
}
```

`numeros_por_extenso` converte números em palavras antes do TTS somente para idioma iniciado por `pt`. Use `editar_perfil(alias="voz", voz={"numeros_por_extenso": false})` para enviar dígitos. Perfis sem esse campo usam `true`.

Cada voz aceita `seed`, enviado ao modelo, e `tom` (semitons) e `velocidade`, aplicados com o ffmpeg. Exemplo de `vozes`: `{"male": {"tom": -1.5}}`.

`criar_perfil` e `editar_perfil` conferem na hora se o GGUF (todas as partes, se dividido) e o `mmproj`, quando informado, existem em `models_dir`. Baixe os pesos antes de criar ou editar o perfil.

## Segurança

O bind padrão é local (`127.0.0.1`). Sem `chaves`, não há autenticação por chave. Para ativá-la, defina valores próprios em `chaves.admin` e `chaves.inference`: `/mcp` exige `admin`; `/v1` aceita ambas. Não compartilhe essas chaves.

Bind fora do local exige as duas chaves na configuração e permite acesso de rede ao `/v1`. O `/mcp` só aceita chamadas locais, por proteção de DNS rebinding do SDK; mudar o bind não libera MCP remoto.

## Testes

```sh
uv run pytest -q
```

Os testes usam um `llama-server` falso e não exigem GPU. Testes marcados `ffmpeg` dependem do ffmpeg instalado. O CI em `.github/workflows/testes.yml` executa pytest no Windows e no Linux.

## Limitações conhecidas

- Modelo fora do ar recebe 409 no `/v1`, sem carga sob demanda; isso inclui o boot enquanto `subir_no_boot` ainda carrega.
- O limite de corpo de 512 MB depende de `Content-Length`; corpo chunked sem esse header não tem esse limite.
- Download não tem cancelamento pela ferramenta MCP. Download interrompido deixa o temporário em `models_dir/<pasta>/.cache/huggingface/download/` e recomeça do zero na próxima vez.
- Antes da primeira medição, a checagem de RAM não conta o KV cache: com contexto grande (`-c`) e pouca RAM livre, o sistema pode paginar. Suba uma vez com contexto menor (a subida grava a medição) ou reduza `-c`.
- Conversão de números por extenso para voz só funciona em português.
- Planejamento de GPU só reconhece NVIDIA via `nvidia-smi`.
- Linux/macOS não têm Job Object nem medida de transbordo de VRAM; encerramento forçado pode deixar processos vivos.
- `/mcp` só aceita chamadas locais.

## Licença

MIT.
