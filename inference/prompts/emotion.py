from __future__ import annotations

TASK_DESC = """\
You are an expert at emotion analysis in conversational audio.
The dataset is MSP-Podcast/MSP-Conversation: naturalistic conversations
annotated with a primary emotion class (Happy, Sad, Angry, Neutral,
Disgust, Fear, Contempt, Surprise) and continuous traces of valence,
arousal, and dominance (V/A/D, normalised 0–1).\
"""

SCHEMA_DESC = """\
Each retrieved segment contains:
  time          [start, end] in seconds
  speaker       speaker ID
  text          ASR transcript (may contain errors)
  domain_labels two complementary emotion signals:
                - categorical (SER): primary class + per-class scores
                  (Happy, Sad, Angry, Neutral, Disgust, Fear,
                   Contempt, Surprise)
                - continuous (CSER): valence, arousal, dominance each
                  as {mean, std} in [0, 1]; null if not available
  _sources      retrieval tools used and their confidence scores\
"""

# Per-modality one-line descriptions injected into PLAN_PROMPT.
# {speaker_examples} is formatted by the agent before injection.
TOOL_DESCS: dict[str, str] = {
    "time":       "time        true to retrieve all segments in the localized window",
    "keyword":    "keyword     word or phrase to match in transcript",
    "label":      "label       exact emotion CLASS NAME only"
                  " (Happy, Sad, Angry, Neutral, Disgust, Fear, Contempt, Surprise)"
                  " — never a filename",
    "text_emb":   "text_emb    key phrases and topics to search semantically",
    "audio_emb":  "audio_emb   speaker voice-similarity search —"
                  " set to a key string from: {audio_examples}",
    "domain_emb": "domain_emb  emotion similarity search —"
                  " set to a key string from: {audio_examples}",
    "audio":      "audio       true to retrieve raw audio for multimodal analysis",
}

# {tools_desc} is built by the agent from enabled modalities.
PLAN_PROMPT = """\
{task_desc}

{schema_desc}
{track_info}
{clip_info}
Plan evidence retrieval for the query. For each evidence set:
  localize  (required) time window — "around N min/sec" or "X to Y min/sec"
{tools_desc}

For comparison queries (which speaker has higher X), create one set per
speaker, each with that speaker's search key.

Use multiple sets whenever evidence spans different time ranges or subjects.
Output JSON only — omit fields that are not needed:
{schema_example}

Query: {query}\
"""

ANSWER_PROMPT = """\
{task_desc}

Answer the query using the retrieved evidence sets below. Each set is a
list of segments sorted by time; sets correspond to distinct evidence
pieces from the planner.

Audio reference clips are provided before the question text. Their roles:
  [Snippet A] = [speaker example] / [audio example]  — the target speaker's voice
  [Snippet B] = second speaker     — used in comparison queries
  [Snippet E] = [emotion example] / [audio example] — the reference emotion
  [Set N: …]  = episode audio for evidence set N (matches set N below)
When there is one evidence set, all its segments are the target speaker's
utterances. For comparison queries, Set 1 is for the first speaker and
Set 2 for the second. Speaker IDs in the evidence (e.g. SPEAKER_02) are
diarization labels and do not correspond to the clip handles above.

Evidence:
{context}

Give a concise answer, then write "Rationale: " followed by your
reasoning. Cite time ranges and speaker IDs. Note when ASR errors or
missing labels limit confidence.

Query: {query}

Answer:\
"""
