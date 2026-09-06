from __future__ import annotations

# GT is embedded in query_item directly; no segment retrieval needed
SKIP_GT_EVIDENCE = True

# Used in factual prompt
SCHEMA_DESC = (
    "Task: clinical question answering on MedMosaic Long-Form.\n"
    "Evidence segment format:\n"
    "  - speaker: diarization label (e.g. SPEAKER_00, SPEAKER_01) — "
    "one label per participant (clinician or patient); context and transcript "
    "content determines which is which.\n"
    "  - domain: cough detection output — primary '{n} cough'/'no cough', "
    "max_cough_prob 0–1.\n"
    "Ground truth: 'Yes'/'No' annotation from clinical review. "
    "'Exhibited' means the patient physically showed or reported a symptom; "
    "a clinician asking about a symptom the patient denied does NOT count.\n"
    "Proxy cases: some consultations involve a parent/caregiver speaking on "
    "behalf of a child patient. In these cases the speaker in the transcript "
    "may be the parent, not the patient — use GT label and transcript content "
    "to determine who the actual patient is.\n"
    "Negative answers: for 'reported/exhibited' queries with a No GT label, "
    "reasoning of the form 'this is a non-[X] episode / episode is about [Y] "
    "complaint, so no [X] symptoms were reported/exhibited' is valid factual "
    "grounding. A GT label confirming a different episode type (e.g. knee "
    "injury, abdominal pain) IS evidence of absence — do NOT mark factual=NO "
    "just because the GT does not explicitly mention the absent symptom."
)

# No audio clip placeholders in health queries; minimal note for consistency
SNIPPET_NOTE = (
    "Context: speaker labels (SPEAKER_00, SPEAKER_01) in evidence are "
    "diarization IDs — use transcript content to determine clinician vs patient."
)
