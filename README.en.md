# local-hub-mcp

**Português:** [README.md](README.md)

Tool names, config fields and service messages are in Portuguese; this README explains each one.

## What it is

A connection to use and manage local GGUF models with llama.cpp.
The OpenAI-compatible gateway (`/v1`) receives conversations.
The MCP tools (`/mcp`) download, configure, benchmark, start, and stop models.
The service uses your installed llama.cpp; it does not install binaries or download models at boot.

```text
Cliente ──→ :8079/v1  ──→ llama-server por perfil
        └─→ :8079/mcp ──→ gerenciamento dos perfis e processos
```

## Platforms

| Platform | Support and limitations |
|---|---|
| Windows | Full support: Job Object to terminate child processes and dedicated/shared memory counters to detect VRAM overflow. |
| Linux/macOS | No Job Object and no VRAM overflow measurement. Force-killing the service may leave `llama-server` running. |
| GPU | NVIDIA detected via `nvidia-smi`. Without a detectable NVIDIA GPU, use profiles with `onde: "cpu"`. |

To use an NVIDIA GPU, install a llama.cpp build with CUDA backend. The presence of a GPU alone is not enough: a build without CUDA may run the model on CPU and run slowly. The service checks for the presence of `llama-server` and queries the GPU via `nvidia-smi`; it does not validate the build's CUDA backend. After starting a profile, use `logs(alias="qwen3-4b")` and check whether the log shows the CUDA device in use.

## Requirements

- Python **3.12 or higher** and [uv](https://docs.astral.sh/uv/).
- Git to clone the repository.
- [llama.cpp](https://github.com/ggml-org/llama.cpp/releases): a directory containing `llama-server` and `llama-bench` (`.exe` on Windows). On Windows with NVIDIA, choose the `...-cuda-...` zip and keep the distribution files together. On Linux, compile with `-DGGML_CUDA=ON` or use a CUDA build. Without an NVIDIA GPU, use a CPU build.
- Any client supporting MCP over Streamable HTTP — for example Claude Code, Claude Desktop, Cursor, or VS Code — to call management tools (see [Connecting a client](#connecting-a-client)).
- [ffmpeg](https://ffmpeg.org/download.html), optional: required for voice and audio/video attachments. Install on Windows with `winget install Gyan.FFmpeg`, on Linux with `sudo apt install ffmpeg`, or on macOS with `brew install ffmpeg`.
- Hugging Face account, optional: provide `HF_TOKEN` in the service environment to access gated repositories your account has access to.

## Step-by-step installation

### 1. Clone and install dependencies

On both systems:

```sh
git clone https://github.com/PatrickCavalcanti/local-hub-mcp.git
cd local-hub-mcp
uv sync
```

### 2. Create the configuration

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

Edit `config.json`. Point `llama_cpp_dir` to the directory containing llama.cpp binaries and `models_dir` to an existing directory that will store GGUFs. To use the directory created above, set `"models_dir": "models"`. Relative paths are resolved from the directory containing `config.json`. On Windows, use `/` in JSON paths or escape each `\` as `\\`.

Keep `bind: "127.0.0.1"` for local use. Do not include the gateway port inside `portas_internas`.

### 3. Start the service

Windows PowerShell:

```powershell
.\iniciar.bat
```

Linux/macOS:

```sh
sh iniciar.sh
```

Or, on both systems:

```sh
uv run local-hub-mcp --config config.json
```

On first run without `perfis.json`, the service copies `perfis.exemplo.json`. There are three example recipes: `qwen3-4b`, `gemma3-4b`, and `voz`. Weights still need to be downloaded.

The console displays the llama.cpp version, `/v1` and `/mcp` endpoints, profile count, detected ffmpeg, and Job Object mode. Without ffmpeg, it informs that voice and audio/video attachments are disabled. Keep the terminal open; use Ctrl+C to shut down cleanly.

### 4. Download and launch the first model

With the service running, connect an MCP client according to the next section. The examples below are **MCP tool calls**, performed by the client; they are not terminal commands or a Python API.

To use the `qwen3-4b` chat profile, ask the client to execute:

```text
baixar_modelo(
  repo_id="unsloth/Qwen3-4B-Instruct-2507-GGUF",
  arquivos=["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"]
)
```

The tool returns a task. Query `tarefa(id="<returned id>")` until it completes successfully. The file is saved to `models_dir/unsloth--qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, the path already used by the example profile.

The profile comes with `onde: "gpu"`. Without NVIDIA, before launching use `editar_perfil(alias="qwen3-4b", onde="cpu", flags=["-ngl", "0", "-c", "4096", "-np", "1", "--jinja"])`. With a GPU, check the CUDA build and available memory with `status()`.

Then execute `subir_modelo(alias="qwen3-4b")`. If loading becomes a background task, follow the returned ID. Check `status()` and `logs(alias="qwen3-4b")`; only then start the first conversation over `/v1`. To stop the model, use `derrubar_modelo(alias="qwen3-4b")`.

Alternative: download the GGUF manually and place it at the exact path indicated by the profile's `arquivo` field, inside `models_dir`. For another path or model, use `criar_perfil` or `editar_perfil`. Models with vision or voice may also require an `mmproj` file; see the notes in `perfis.exemplo.json`. The examples are starting points, not guarantees they fit your machine. Use `medir` to compare flags.

## Connecting a client

### MCP

Management uses **Streamable HTTP** at `http://127.0.0.1:8079/mcp`. The service returns a `tarefa` within `espera_s` (90 s by default); the call timeout only needs to exceed that interval, for example **120 seconds**.

Claude Code example:

```sh
claude mcp add --transport http local-hub http://127.0.0.1:8079/mcp
```

If you configure `chaves`, append `--header "Authorization: Bearer <admin>"` to the command above, replacing `<admin>` with `chaves.admin`.

In Claude Code, set the [`MCP_TOOL_TIMEOUT`](https://code.claude.com/docs/en/env-vars) environment variable in milliseconds (e.g. `MCP_TOOL_TIMEOUT=120000`). In other clients, adjust the timeout in their MCP server configuration.

Generic JSON configuration (registration format may vary depending on the client):

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

If you configure `chaves`, include the `Authorization: Bearer <admin>` header in MCP calls, replacing `<admin>` with the value of `chaves.admin`. Test the connection by calling `status` and `listar_modelos`.

### OpenAI

Configure `base_url = http://127.0.0.1:8079/v1`, `api_key` with any text if `chaves` is not set, or with the value of `chaves.inference` if set. Use `model` matching the profile alias or the wildcard `atual`.

With the profile running, Linux/macOS:

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

Replace `local` with the `inference` key if authentication is enabled. `atual` selects the last launched profile compatible with the route: chat for `/v1/responses` and `/v1/chat/completions`, STT for `/v1/audio/transcriptions`, TTS for `/v1/audio/speech`, and embedding for `/v1/embeddings`. `GET /v1/models/atual` reports `aponta_para`. In the client, declare only capabilities supported by the models used with this wildcard.

To load a model when the service starts, add its alias to `subir_no_boot`. The client still needs to wait for loading to finish.

**Example: Igris.** Configure the provider with `protocol: openai`, `base_url: http://127.0.0.1:8079/v1`, `api_style: responses`, and `model_id` matching the alias or `atual`. Register the MCP at the address above with `call_timeout_ms` ≥ 120000. With `chaves`, use `inference` in the provider and `Authorization: Bearer <admin>` in MCP.

## Reference

### MCP tools

| Tool and parameters | What it does |
|---|---|
| `status()` | Processes, recent startup failures, free resources, version, ffmpeg, and tasks. |
| `listar_modelos()` | Profiles and unassigned GGUF files in `models_dir`. |
| `procurar_modelo(texto, limite=10, ordem="trending")` | Searches originals and GGUF conversions on Hugging Face. Order: `trending`, `downloads`, `recentes`, or `likes`. |
| `conversoes_gguf(repo_id, ordem="downloads", limite=20)` | Lists conversions linked to the original via `base_model`. |
| `detalhes_repo(repo_id)` | Lists quants, sizes, parts, mmproj, and fit estimates. |
| `baixar_modelo(repo_id, arquivos)` | Downloads file list and all parts of split GGUFs; returns a task. |
| `criar_perfil(alias, arquivo, onde, ...)` | Creates recipe; accepts `flags`, `capacidades`, `mmproj`, `anexos`, `video`, `voz`, and `notas`. |
| `editar_perfil(alias, ...)` | Modifies the same optional fields; only provided fields change. Flags replace the entire list; `voz` updates only provided keys; `mmproj=""` removes the projector. Takes effect on next launch. |
| `apagar_perfil(alias)` | Removes profile, not the weights. Rejects running profile. |
| `apagar_arquivos(caminhos, confirmacao=null)` | Plans deletion inside `models_dir`; show the list to the user and, only after their confirmation, repeat with code in `confirmacao`. Rejects files used by a profile. |
| `subir_modelo(alias)` | Checks VRAM/RAM and starts `llama-server`; long loads become a task. Without `medido`, rejects a second GPU profile (`gpu` or `gpu+cpu`) while another GPU profile is running: stop the first one. Profiles using `cpu` can coexist, subject to available RAM. |
| `derrubar_modelo(alias)` | Terminates process and checks port and VRAM release. |
| `medir(alias, ncmoe=null, ub=null, gravar=false)` | Compares `-ncmoe`/`-ub` via `llama-bench`; `gravar=true` saves best flags. Requires free GPU. |
| `testar_modelo(alias, capacidades=null)` | Short text, vision, TTS, and STT tests against running profile. |
| `logs(alias, linhas=100)` | Last lines of profile log (from 1 to 2000 lines). |
| `info_conexao()` | URLs, aliases, capabilities, and voices to configure client. |
| `tarefa(id)` | Status and result: `em_andamento`, `ok`, `erro`, or `cancelada`; download progress. |

Check `ok` and any errors in responses. Tool names and parameters remain in Portuguese.

### `config.json`

| Field | Default | Usage |
|---|---|---|
| `llama_cpp_dir` | required | llama.cpp binaries directory. Boot checks `llama-server`; `medir` uses `llama-bench`. |
| `models_dir` | required | Existing weights directory; downloads and profile model paths reside inside it. |
| `ffmpeg_dir` | `null` | ffmpeg directory; if empty, searches PATH and winget directory. |
| `bind` / `porta` | `127.0.0.1` / `8079` | Gateway address. Non-local bind requires both keys. |
| `portas_internas` | `[8180, 8199]` | Inclusive port range for `llama-server` processes. |
| `chaves` | `{}` | Object with `admin` and `inference`; see Security. |
| `perfis` | `"perfis.json"` | Catalog file; not the weights directory. |
| `logs_dir` | `"logs"` | Profile logs and `auditoria.log`. |
| `load_timeout_s` | `600` | Model load timeout in seconds. |
| `espera_s` | `90` | Long-running tool wait time before returning a task. |
| `subir_no_boot` | `[]` | Aliases loaded sequentially on startup; failures are logged and shown in status. |

### Profile format

`perfis.json` contains an object with a `perfis` list. Minimal structural example:

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

Replace `arquivo` with the real path before starting. Prefer editing the catalog using MCP tools.

| Field | Default / rule |
|---|---|
| `alias` | Required; up to 64 characters, starts with lowercase letter or number; also allows `.`, `_`, `-`. `atual` is reserved. |
| `arquivo` | Required; GGUF relative to `models_dir`. For split models, specify the first part. |
| `onde` | Required: `gpu`, `gpu+cpu`, or `cpu`. Flags control llama.cpp offload; this field guides resource planning. |
| `flags` | `[]`; each token separate. Port, host, file, mmproj, alias, and `--video-ffmpeg-dir` are managed by the service. `--no-mmap` and `--mlock` are prohibited. |
| `capacidades` | `["llm.chat"]`; allowed values below. |
| `mmproj` | `null`; projector relative to `models_dir`. |
| `anexos` / `video` | `false` / `false`; attachment and video adapter options. |
| `voz` | `null`; required when `capacidades` includes `stt` or `tts`. When present, `vozes` must contain at least one voice. |
| `notas` | `""`; free-form description. |
| `medido` | `null`; measurement with `vram_gb`, `ram_gb`, `carga_s`, and `em`. Changes to file, execution location, or flags discard the measurement. |

Allowed capabilities: `llm.chat`, `llm.vision`, `llm.fast`, `embedding`, `stt`, `tts`, `text.generate`, `audio.in`, `video.in`. Declaring a capability does not add it to the model.

`voz` block, with defaults:

```json
{
  "vozes": {"female": {}, "male": {}},
  "idioma": "pt",
  "pedaco_tts": 250,
  "pedaco_stt_s": 45.0,
  "numeros_por_extenso": true
}
```

`numeros_por_extenso` spells out numbers in words prior to TTS only for languages starting with `pt`. Use `editar_perfil(alias="voz", voz={"numeros_por_extenso": false})` to pass digits. Profiles without this field default to `true`.

Each voice accepts `seed`, sent to the model, and `tom` (semitones) and `velocidade`, applied with ffmpeg. Example `vozes`: `{"male": {"tom": -1.5}}`.

`criar_perfil` and `editar_perfil` immediately check that the GGUF (all parts if split) and `mmproj`, when specified, exist in `models_dir`. Download the weights before creating or editing the profile.

## Security

Default bind is local (`127.0.0.1`). Without `chaves`, there is no key authentication. To enable it, set your own values in `chaves.admin` and `chaves.inference`: `/mcp` requires `admin`; `/v1` accepts either. Do not share these keys.

Non-local bind requires both keys in configuration and allows network access to `/v1`. `/mcp` only accepts local calls due to DNS rebinding protection in the SDK; changing bind does not enable remote MCP.

## Tests

```sh
uv run pytest -q
```

Tests use a mock `llama-server` and do not require a GPU. Tests marked `ffmpeg` depend on ffmpeg being installed. CI in `.github/workflows/testes.yml` runs pytest on Windows and Linux.

## Known limitations

- Inactive model receives 409 on `/v1`, with no on-demand loading; this includes boot while `subir_no_boot` is still loading.
- 512 MB body limit depends on `Content-Length`; chunked body without this header has no such limit.
- Download cannot be cancelled via MCP tool.
- Number-to-words conversion for voice only works in Portuguese.
- GPU planning only recognizes NVIDIA via `nvidia-smi`.
- Linux/macOS have no Job Object and no VRAM overflow measurement; force-killing may leave processes running.
- `/mcp` only accepts local calls.

## License

MIT.
