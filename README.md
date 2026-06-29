# coworking

A multi-agent experiment harness on top of [pydantic-ai](https://ai.pydantic.dev/).
The current scenario (`cwrk chatbox`) launches a small team of agents that explore
a shared paper corpus and try to figure out how the papers relate to each other.

## Configuration

Before the first run, point the harness at the model you want to use.

### 1. Edit `configs/chatbox_models.yaml`

The file expects exactly three model entries under the `models:` map. Each entry
describes one OpenAI-compatible endpoint. The fields are:

| Field          | Meaning                                                       |
| -------------- | ------------------------------------------------------------- |
| `model_name`   | The model id sent in the request                              |
| `base_url`     | OpenAI-compatible base URL                                    |
| `api_key_env`  | Name of the env var that holds the API key (see step 2)       |
| `max_tokens`   | Max output tokens                                             |
| `temperature`  | Sampling temperature (omit to use the server default)         |
| `top_p`        | Nucleus sampling                                              |
| `presence_penalty` | Presence penalty                                           |
| `extra_body`   | Free-form JSON merged into every request body                 |

You can have one real entry and stub the other two, e.g.

```yaml
models:
  my-model:
    model_name: gpt-4o
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    max_tokens: 16384
  stub-a: { model_name: x, api_key_env: STUB_KEY }
  stub-b: { model_name: x, api_key_env: STUB_KEY }
```

`uv` automatically loads `.env` from the project root.

### 2. Put the API key in `.env`

For whichever entry you intend to use, create `.env` with the variable named in
its `api_key_env` field. For the snippet above:

```sh
OPENAI_API_KEY=sk-...
STUB_KEY=sk-stub
```

(The two `STUB_KEY` stubs just need any non-empty value to satisfy the loader.)

### 3. Pick which entry drives the run

The `--model-alias` flag selects the YAML entry that is used as the model for
**all three agents**. There is no per-agent model selection - the whole team
runs against the same endpoint.

## Agentic Trace Recording

Optional - if you want to capture structured traces of agent runs, you need an
OTLP collector running before you start the harness.

One option is to use [Vector](https://vector.dev/) as a local collector. A
homelab-ready config is included at `./configs/vector_collector.yaml`:

```sh
vector -c ./configs/vector_collector.yaml
```

Then set the OTLP endpoint env var via export `export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318` or .env (`cp template.env .env` and edit it) and run the pipeline as usual.

Traces are written to `./data/traces/` in zstd-compressed JSONL format.

I've published a [small post](https://jezzarax.github.io/vector-otlp-tracing/) about how to set up vector for collecting traces from Codex/Claude Code/Pydantic AI.

## Running

```sh
uv run cwrk chatbox
```

Flags:

- `--llm-config PATH` - YAML model config (default `./configs/chatbox_models.yaml`)
- `--model-alias TEXT` - which YAML entry to use (default `qwen`)
- `--papers-dir DIRECTORY` - paper corpus (default `./data/papers`)

Example using the `my-model` entry from above:

```sh
uv run cwrk chatbox --model-alias my-model
```

## Setup

Three agents - `Alice`, `Bob`, `Carol` - run concurrently against the same model:

| Agent | Role                          |
| ----- | ----------------------------- |
| Alice | explorer, peer of Bob + Carol |
| Bob   | explorer, peer of Alice + Carol |
| Carol | explorer, peer of Alice + Bob  |

All three receive the same task instruction; each is told who its peers are so
it can address them by name.

### Shared vs private toolsets

- **`Papers`** (one instance, shared by all agents): `list_files`, `info`,
  `read`, `grep`. Tools take a `paper_id` (the file stem, e.g. `2510.10185`),
  not a path.
- **`Notebook`** (one instance per agent): `jot`, `reread`. Private scratchpad.
- **`send_message`** (built into every agent): coordinates with peers. Omit
  `to` to broadcast; pass a peer name to direct the message. Backed by the
  `Switchboard`, which delivers via pydantic-ai's `enqueue(priority="asap")`
  so the peer sees the message in its next request even if it was about to end.

## Goal

The `data/papers/` folder contains markdown extracts of publicly available
papers. The selection is arbitrary for now - it's a small grab-bag rather than
a curated corpus - and is expected to grow and change as the project evolves.

Each agent independently explores `data/papers/` (15 markdown papers), records
intermediate observations in its private Notebook, and coordinates with peers
to identify how the papers are connected - shared topics, methods, datasets,
citations, contradictions, or evolution of ideas. The run ends with each agent
replying with a concise synthesis of the connections it found.
