# Inference

Answers a query over one episode by planning what to retrieve, retrieving it, and reasoning over the result.

```bash
python scripts/run_inference.py conf/emotion_hybrid_all.yaml [--resume]
```

Reads `inference.queries`, writes to `inference.predictions`; `--resume` skips queries already there.

## The agent loop

`HARPAgent.__call__` runs four steps per query:

| Step | Where | What |
|---|---|---|
| plan | `HARPAgent._plan` | LLM emits JSON: which windows to localize, which searches to run |
| dispatch | `HARPAgent._dispatch` → `tools.py` | Execute that plan against `segments/` and `faiss/` |
| gather | `format.format_evidence_sets` | Render segments into prompt text; attach clips if `audio` is on |
| answer | `HARPAgent._answer` | LLM returns prediction + rationale |

Each record keeps `plan` and `evidence_sets` so evaluation can score retrieval separately from answering.

`OracleAgent` subclasses `HARPAgent` and derives the plan from ground truth (`inference.oracle: true`) — an upper bound on answering when retrieval is perfect.

## Layout

```
inference/
├── agent.py       # HARPAgent, OracleAgent
├── tools.py       # localize, search_by_*, gather_evidence
├── format.py      # evidence sets → prompt text
├── preprocess.py  # optional re-ASR / re-labeling of retrieved clips
├── llm/           # backbone: base.py, qwen_text.py, qwen3_omni.py
└── prompts/       # task: emotion.py, music.py, health.py
```

**`llm/` is the backbone, `prompts/` is the task.** Swap either independently.

## `llm/` — the backbone

Anything satisfying `LLMBase` (`__call__(prompt, **kwargs) -> str`) works. The agent passes `max_tokens=` when planning and `audio_clips=` when answering with the `audio` modality — a text-only backbone can ignore the latter, but then drop `audio` from `modalities` or clips are retrieved and discarded.

| Class | Model | Audio |
|---|---|---|
| `QwenText` | Qwen3-30B-A3B-Instruct | no |
| `Qwen3Omni` | Qwen3-Omni-30B-A3B-Instruct | yes |

Both talk to a local vLLM server (`scripts/serve_vllm.sh [text|omni]`); set `base_url: null` to load via HF instead.

## `prompts/` — the task

`inference.task` selects the module. Each defines five symbols, read by name in `agent.py`: `TASK_DESC`, `SCHEMA_DESC`, `TOOL_DESCS`, `PLAN_PROMPT`, `ANSWER_PROMPT`. `TOOL_DESCS` is keyed by modality name — omit a key for a tool the task never uses. The two templates are `str.format` targets; see the `.format(...)` calls in `HARPAgent._plan` and `._answer` for the required fields.

### Prompt tuning across backbones

**Expect to retune prompts when you change the backbone.** The plan step must come back as parseable JSON, and models differ in how reliably they do that. If planning degrades after a swap, look there before suspecting retrieval: an unparseable plan falls back to `{"sets": []}`, which retrieves nothing and makes the answer look confidently uninformed.

Prompts are per-task, not per-model, so a second backbone's variant means either a new module selected by `task:` or a branch inside the existing one.

## Adding a task

Create `inference/prompts/{task}.py` with the five symbols, set `inference.task`, and add the matching `evaluate/prompts/{task}.py`. Nothing else changes — `agent.py` and `tools.py` treat `domain_labels` as opaque.
