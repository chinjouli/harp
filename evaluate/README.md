# Evaluate

Scores predictions against ground truth with an LLM judge.

```bash
python scripts/run_eval.py conf/emotion_hybrid_all.yaml [--resume]
python scripts/run_score.py data/mspemotion          # aggregate by query type
```

Scores mirror the predictions filename: `predictions_text.jsonl` → `scores_text.jsonl`.

## Scores

`run_inference.py` saves the `plan` and `evidence_sets` alongside the answer, so retrieval is scored separately from answering.

| Score | Type | Question |
|---|---|---|
| `answer` | 0/1 | Does the prediction match the GT answer? |
| `factual` | 0/1 | Are the rationale's claims consistent with GT evidence? |
| `faithful` | 0/1 | Does the rationale stay grounded in the *retrieved* evidence, without contradicting or inventing beyond it? |
| `rationale` | 0/1 | `factual & faithful` |
| `retrieval_hit_rate` | float | Fraction of GT evidence segments the retrieval found |
| `plan_hit_rate` | float | Fraction of GT evidence segments the plan's windows covered |

`factual` and `faithful` differ in what they compare against: `_score_factual` checks the rationale against ground truth, `_score_faithful` against what the agent actually retrieved. Audio-only setups have no text evidence to be faithful to, so `run_eval.py` sets `skip_faithful` for them and `rationale` falls back to `factual` alone.

## Layout

```
evaluate/
├── agent.py             # EvalAgent — load GT, retrieve evidence, judge
├── closed_api_agent.py  # ClosedAPIEvalAgent — batch/resume for paid APIs
├── metrics.py           # score_answer / _rationale / _retrieval / _plan / _all
├── gt_retriever.py      # generic GT loading + formatting
├── gt/                  # task-specific GT retrieval (emotion, music)
├── judge/               # ClaudeJudge, OpenAIJudge, GeminiJudge, QwenJudge
└── prompts/             # task-specific judge prompts
```

## Two paths

`EvalAgent` (default) loads GT from `evaluate.gt_dir`, retrieves the evidence for each query, and judges it. This is also what writes the GT evidence file.

`ClosedAPIEvalAgent` (`--closed-api`) reuses that evidence file to score against a paid API, so **run the default path once first**. It adds `--resume` and `--batch`.

```bash
python scripts/run_eval.py conf/emotion_hybrid_all.yaml --closed-api --batch \
    --judge evaluate.judge.openai.OpenAIJudge --judge-model gpt-4o
```

`--judge` / `--judge-model` override the config, so comparing judges needs no edits. The default is `GeminiJudge` — a different family from the Qwen candidate, since models favour their own outputs; `scripts/run_judgeeval.py` quantifies that across judges. Only `OpenAIJudge` implements `batch()`.

## Extending

**A judge** subclasses `JudgeBase` (`__call__(prompt, **kwargs) -> str`), same contract as `LLMBase`. Add `batch()` for `--batch`. Keys come from the environment: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.

**A task** needs `evaluate/prompts/{task}.py`, set as `evaluate.prompts`. Prompt modules are duck-typed — the agents read `SCHEMA_DESC`, `SNIPPET_NOTE`, `STRIP_SPEAKER_TYPES`, and `format_gt_segments` by name and fall back to generic defaults. Add `evaluate/gt/{task}.py` only if the standard segment format isn't enough.
