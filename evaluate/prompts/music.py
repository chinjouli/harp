from __future__ import annotations


# GT segment retrieval is not meaningful for music — scores come from judge_info
SKIP_GT_EVIDENCE = True
# Internal SONG_XX speaker labels are noise in time-only queries
STRIP_SPEAKER_TYPES: set = {"time"}

# Used in factual prompt (schema + scale context for judge)
SCHEMA_DESC = (
    "Task: music aesthetic evaluation on SongEval.\n"
    "Evidence segment format:\n"
    "  - speaker: section label (e.g. SONG_03). SONG_N is the N-th "
    "song (1-indexed), so SONG_03 = 3rd song.\n"
    "Evidence set mapping (same as inference): Set 1 = evidence for the "
    "song mentioned FIRST in the query; Set 2 = song mentioned SECOND; "
    "and so on.\n"
    "  - domain.scores: raw predictions from laion/music-aesthetics model "
    "(Naturalness, Clarity, Musicality, Coherence, Memorability, each 1–5). "
    "These are model estimates from automatically extracted sections; "
    "they will NOT match GT scores numerically. Raw model values can be "
    "substantially higher or lower than GT-annotated means for the same song.\n"
    "GT label scores are averaged human annotations on the same 1–5 scale.\n"
    "For comparison/audio/time/position queries: the primary check is "
    "whether the model identifies the correct WINNER (Song A vs Song B). "
    "When raw model scores and GT means differ, prefer relative ordering "
    "over exact values — accept if the winning side matches GT."
)

# Prepended to factual and faithful prompts for closed-source text judges.
SNIPPET_NOTE = (
    "Context for closed-source text judges:\n"
    "Audio placeholders: [A's audio example] and [B's audio example] in the "
    "query are audio clips of Song A and Song B given to the model. "
    "[Snippet A] / [Snippet B] in model responses = those same clips.\n"
    "Claims about audio input clips (e.g. '[Snippet A] sounds like high "
    "quality') are based on the model's direct audio perception — they are "
    "NOT in the retrieved evidence text; do NOT mark faithful=NO for such "
    "claims.\n"
    "Song IDs: SONG_XX in evidence speaker field is 1-indexed — "
    "SONG_N = N-th song (e.g. SONG_03 = 3rd song).\n"
    "Evidence sets: Set 1 = first song mentioned in the query, Set 2 = "
    "second song mentioned, and so on — same convention as inference. "
    "When only one set exists, the model uses different segments within "
    "it to compare songs.\n"
    "Value scales: aesthetic scores in evidence are raw model outputs (1–5); "
    "GT label scores are human annotations (1–5). Scores may differ "
    "between raw predictions and GT means — when they differ, accept "
    "based on relative ordering (which song has higher scores)."
)
