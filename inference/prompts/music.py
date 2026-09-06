from __future__ import annotations

TASK_DESC = """\
You are an expert at music analysis and aesthetic evaluation.
The dataset is SongEval: full music tracks segmented into sections by
musical structure (detected via MERT-based diarization). Each section is
labelled with five aesthetic scores (1–5 scale): Naturalness, Clarity,
Musicality, Coherence, and Memorability. Transcribed lyrics are provided
where present.\
"""

SCHEMA_DESC = """\
Each retrieved segment contains:
  time          [start, end] in seconds
  speaker       section ID (song label)
  text          lyrics transcript for this section (may be empty or have ASR errors)
  domain_labels aesthetic scores — Naturalness, Clarity, Musicality,
                Coherence, Memorability (each 1.0–5.0; null if unavailable)
  _sources      retrieval tools used and their confidence scores\
"""

# Per-modality descriptions injected into PLAN_PROMPT.
# {speaker_examples} is formatted by the agent (may be "none" for music).
TOOL_DESCS: dict[str, str] = {
    "time":       "time        true to retrieve all sections in the localized window",
    "keyword":    "keyword     word or phrase to match in lyrics transcript",
    "label":      "label       exact aesthetic tier CLASS NAME only"
                  " (High, Medium, Low) — never a filename",
    "text_emb":   "text_emb    verbatim lyric quote or thematic phrase to search"
                  " semantically",
    "audio_emb":  "audio_emb   audio reference key for acoustic similarity"
                  " — use exact key from available list: {speaker_examples}",
    "domain_emb": "domain_emb  audio reference key for music content/style"
                  " similarity — use exact key from available list:"
                  " {speaker_examples}",
    "audio":      "audio       true to retrieve raw audio for multimodal analysis",
}

PLAN_PROMPT = """\
{task_desc}

{schema_desc}
{track_info}
{clip_info}
Plan evidence retrieval for the query. For each evidence set:
  localize  (required) time window — "around N min/sec" or "X to Y min/sec"
{tools_desc}

For audio CLIP queries (e.g. [snippet_X.wav]) with no known time position,
use "full track" as localize and set keyword to a distinctive lyric phrase
from the preprocessed clip info above — do NOT use label for clip matching.
label (High/Medium/Low) is only for queries explicitly asking about quality tier.
For position queries (e.g. "16th song"), use ordinal localize ("16th song").
Use multiple sets for queries comparing across sections or time ranges.
Output JSON only — omit fields that are not needed:
{schema_example}

Query: {query}\
"""

ANSWER_PROMPT = """\
{task_desc}

Answer the query using the retrieved evidence sets below. Each set is a
list of sections sorted by time; sets correspond to distinct evidence
pieces from the planner.

Audio reference clips are provided before the question text. Their roles:
  [Snippet A] = audio clip for the song mentioned first in the query
  [Snippet B] = audio clip for the song mentioned second in the query
  [Set N: …]  = episode audio for evidence set N (matches set N below)

Evidence:
{context}

Set 1 is evidence for the song mentioned first in the query,
Set 2 for the song mentioned second, and so on. (Ignore if only one set.)
Song IDs are 1-indexed in the metadata, so SONG_N is the N-th song.
Give a concise answer, then write "Rationale: " followed by your
reasoning. Cite specific time ranges and aesthetic scores where relevant.
Note when lyrics are absent or unclear.

Query: {query}

Answer:\
"""
