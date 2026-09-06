"""Generate reported/exhibited query pairs for all 106 long-form episodes.

Reads res.md for RES annotations, Long_Form_metadata.json for MCQ context.
Outputs queries.jsonl (one JSON object per line).

Usage:
    python dataprep/health/generate_queries.py
"""

import json
import re
from pathlib import Path

import argparse

from dataprep import paths


REPORTED_Q = "Did the patient report any respiratory symptoms during the consultation?"
EXHIBITED_Q = "Were respiratory symptoms exhibited during the consultation?"

BEAUTIFY = {
    "self, coughed":                             "Patient (speaker) coughed",
    "self, coughed ":                            "Patient (speaker) coughed",
    "self, coughed, coarse voice":               "Patient (speaker) coughed; coarse voice noted",
    "self, no cough, coarse voice":              "Patient (speaker) had coarse voice; no cough",
    "self, coarse voice":                        "Patient (speaker) had coarse voice",
    "self, no cough":                            "No audible respiratory event from patient",
    "self, breathed once":                       "Single audible breath from patient; no cough",
    "self, no cough, stuffy nose":               "Patient (speaker) had audible nasal congestion; no cough",
    "self, stuffy nose":                         "Patient (speaker) had audible nasal congestion",
    "daughter":                                  "Proxy: speaker's daughter is the patient; no cough heard",
    "son":                                       "Proxy: speaker's son is the patient; no cough heard",
    "son, but speaker coughed":                  "Proxy: speaker's son is the patient; speaker (parent) coughed",
    "son, coughed":                              "Proxy: speaker's son is the patient; speaker (parent) coughed",
    "daughter, but speaker coughed":             "Proxy: speaker's daughter is the patient; speaker (parent) coughed",
    "daughter, but speaker wheezed and coughed": "Proxy: speaker's daughter is the patient; speaker (parent) wheezed and coughed",
    "someone, but speaker coughed":              "Proxy: speaker coughed; described patient is a third party",
}

def _wire(args) -> None:
    """Resolve path constants from --src / --out."""
    src, out = paths.resolve(args, "medmosaic")
    globals().update(
        RESMD=src / "res.md",
        META_PATH=src / "long_form/Long_Form_metadata.json",
        AUDIO_DIR=src / "long_form",
        OUTPUT=out / "queries.jsonl",
    )


def parse_resmd():
    """Return {filename: (annotation, group)} for the 69 RES entries."""
    result = {}
    for line in RESMD.read_text(encoding="utf-8").splitlines():
        if ".wav" not in line or "[" not in line:
            continue
        parts = line.strip().split("    ")
        fname = parts[0].strip()
        group = parts[-1].strip().strip("[]")
        ann   = "    ".join(parts[1:-1]).strip()
        result[fname] = (ann, group)
    return result


def parse_metadata():
    """Return {filename: {question, answer}} from Long_Form_metadata.json."""
    with open(META_PATH, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for e in data:
        fname = Path(e["audio_path"]).name
        m = re.search(r'\(([a-j])\)', e["ground_truth"])
        ans_text = ""
        if m:
            letter = m.group(1)
            for opt in e["options"]:
                if opt.startswith(f"({letter})"):
                    ans_text = opt
        out[fname] = {"question": e["question"], "answer": ans_text}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    paths.add_path_args(ap)
    _wire(ap.parse_args())
    res_entries = parse_resmd()
    metadata    = parse_metadata()
    all_wav     = sorted(AUDIO_DIR.glob("*.wav"))

    query_id = 1
    stats = {}

    with open(OUTPUT, "w", encoding="utf-8") as fout:
        for wav in all_wav:
            fname = wav.name
            if fname in res_entries:
                ann, group = res_entries[fname]
            else:
                ann, group = "Non-respiratory episode", "NON_RES"

            mcq = metadata.get(fname, {})
            audio_path = f"long_form/{fname}"

            reported_gt = "Yes" if group != "NON_RES" else "No"
            exhibited_gt = "Yes" if group in ("RES_SELF_ACOUSTIC",
                                             "RES_PROXY_SPEAKER_COUGH") else "No"

            judge_info = {
                "group":        group,
                "annotation":   BEAUTIFY.get(ann.strip(), ann.strip()),
                "mcq_question": mcq.get("question", ""),
                "mcq_answer":   mcq.get("answer", ""),
            }

            for query_type, question, gt in [
                ("reported", REPORTED_Q, reported_gt),
                ("exhibited", EXHIBITED_Q, exhibited_gt),
            ]:
                rec = {
                    "query_id":   query_id,
                    "episode_id": wav.stem,
                    "audio_path": audio_path,
                    "query_type": query_type,
                    "query":      question,
                    "ground_truth": {"answer": gt},
                    "judge_info": judge_info,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                query_id += 1

            stats[group] = stats.get(group, 0) + 1

    total = query_id - 1
    print(f"Written {total} queries ({len(all_wav)} audios x 2) -> {OUTPUT}")
    print("\nGroup distribution:")
    for g, n in sorted(stats.items()):
        print(f"  {g:<30s} {n}")
    reported_yes = sum(1 for g in stats if g != "NON_RES") * 0  # recount below
    # recount from file
    lines = OUTPUT.read_text(encoding="utf-8").splitlines()
    r_yes = sum(1 for l in lines if '"reported"' in l and '"Yes"' in l)
    e_yes = sum(1 for l in lines if '"exhibited"' in l and '"Yes"' in l)
    print(f"\nReported  Yes: {r_yes} / {len(all_wav)}")
    print(f"Exhibited Yes: {e_yes} / {len(all_wav)}")


if __name__ == "__main__":
    main()
