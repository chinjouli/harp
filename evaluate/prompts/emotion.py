from __future__ import annotations

from evaluate.gt.emotion import format_gt_segments  # re-exported for EvalAgent

# Used in factual prompt (schema + scale context for judge)
SCHEMA_DESC = (
    "Task: emotion analysis on MSP-Podcast/MSP-Conversation.\n"
    "Evidence segment format:\n"
    "  - speaker: diarization cluster label (e.g. SPEAKER_00). These are "
    "episode-local IDs; the agent matched target speakers by voice "
    "similarity, NOT by label.\n"
    "  - domain.scores: raw SER model probability outputs (each 0–1, "
    "sum ~1). These are predicted scores, NOT GT annotations. "
    "Primary class = the highest-scoring category.\n"
    "  - domain V/A/D: continuous emotion model outputs, each 0–1 "
    "(higher = more positive / more active / more dominant). These come "
    "from a different model than the GT annotations, so absolute values "
    "may differ; check direction only.\n"
    "GT annotation scale: primary label = human-annotated emotion class; "
    "GT V/A/D = human SAM ratings (1–7) normalized to 0–1. A model "
    "citing V=0.11 while GT shows V=0.568 may still be correct if the "
    "relative ordering (vs other speakers/times) agrees.\n"
    "For comparison queries (which speaker has higher X), the key fact "
    "is which speaker ranks higher — do NOT penalize absolute value "
    "differences between model evidence and GT."
)

# Prepended to factual and faithful prompts for closed-source text judges.
SNIPPET_NOTE = (
    "Context for closed-source text judges:\n"
    "Audio placeholders: [audio example], [speaker example], "
    "[A's audio example], [B's audio example] = speaker voice clips "
    "given to the model as audio input. [emotion example] = audio clip of "
    "the target emotion. [Snippet E] in model responses = that emotion clip. "
    "The model heard these; they are not reproduced here.\n"
    "Claims about audio input clips (e.g. '[Snippet E] has happiness score "
    "0.72', '[audio example] is Speaker A') are based on the model's direct "
    "audio perception — they are NOT in the retrieved evidence text; do NOT "
    "mark faithful=NO for such claims.\n"
    "Speaker IDs: SPEAKER_XX in evidence are diarization cluster labels — "
    "they do NOT directly map to [audio example] handles or to "
    "'Speaker A'/'Speaker B' in the query. The agent matched target "
    "speakers by voice similarity.\n"
    "Evidence sets: when multiple sets exist, Set N = evidence for the "
    "N-th mentioned item. When only one set exists, the model may use "
    "different time-windows within it to compare different speakers.\n"
    "Value scales: SER scores in evidence (e.g. Happy=0.62) are raw model "
    "probabilities, not GT labels. V/A/D values in evidence come from a "
    "continuous SER model; GT V/A/D come from human annotations — both "
    "are 0–1 but from different sources, so absolute values may differ. "
    "Accept if the relative ordering or direction is correct."
)
