from __future__ import annotations

TASK_DESC = """\
You are an expert clinical analyst reviewing recorded doctor-patient
consultations. The dataset is MedMosaic Long-Form: single-session
medical conversations covering respiratory, musculoskeletal, and
gastrointestinal conditions. Each recording is one full consultation
segmented by speaker turns. Your task is to answer multiple-choice
clinical questions by retrieving and reasoning over the conversation.\
"""

SCHEMA_DESC = """\
Each retrieved segment contains:
  time          [start, end] in seconds
  speaker       diarization ID (patient or clinician turn)
  text          ASR transcript of this turn (may contain errors)
  domain_labels cough detection —
                  primary:        "{n} cough" or "no cough"
                  max_cough_prob: peak sub-window cough probability (0–1)
  _sources      retrieval tools used and their confidence scores\
"""

TOOL_DESCS: dict[str, str] = {
    "time":      "time        true to retrieve all turns in the localized window",
    "keyword":   "keyword     clinical term, symptom, body part, or medication"
                 " to match in transcript",
    "label":     'label       cough label — exactly "no cough" or "{{n}} cough"'
                 " (e.g. \"1 cough\", \"2 cough\")",
    "text_emb":  "text_emb    symptom description or clinical phrase to search"
                 " semantically",
    "audio_emb": "audio_emb   acoustic reference key for voice-similarity search"
                 " — use exact key from: {audio_examples}",
    "audio":     "audio       true to retrieve raw audio for multimodal analysis",
}

PLAN_PROMPT = """\
{task_desc}

{schema_desc}
{track_info}
{clip_info}
Plan evidence retrieval to answer the yes/no clinical question.
For each evidence set:
  localize  (required) time window — "full recording", "around N min/sec",
            or "X to Y min/sec"
{tools_desc}

Prefer keyword and text_emb over time when the relevant moment is unknown.
Use multiple sets only when the question compares events at different times.
Output JSON only — omit fields that are not needed:
{schema_example}

Query: {query}\
"""

ANSWER_PROMPT = """\
{task_desc}

Answer the yes/no clinical question using the retrieved evidence below.
State "Yes" or "No" first, then write "Rationale: " followed by your
reasoning citing specific transcript turns and time ranges. Note when
ASR errors or missing segments limit confidence.

Evidence:
{context}

Query: {query}

Answer:\
"""
