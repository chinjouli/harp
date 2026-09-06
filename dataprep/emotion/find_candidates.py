"""Find candidate events for query generation in one MSP-Podcast episode.

Output: candidates/{ep}.json with baselines and a candidate list tagged by
shape: st_event | ct_trend | multi_spk_window.

Usage:
    python dataprep/emotion/find_candidates.py --episode MSP-PODCAST_0002
    python dataprep/emotion/find_candidates.py  # all test episodes
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from dataprep import paths

from transformers import pipeline as hf_pipeline

CONVO     = BASE / "msp_conversation_v2.0"
POD_V2    = BASE / "msp_podcast_v2.0"

TOPIC_MODEL   = "Qwen/Qwen3.5-0.8B"

ST_CF_MIN      = 0.50
ST_SPK_THRESH  = 0.50
CT_SPK_THRESH  = 0.70
CT_TREND_DELTA = 0.08
MULTI_DIFF     = 0.10
SKIP_COLS      = {"Central_Time", "Start_Time", "End_Time", "Mean"}

EMO_CODE = {"Neutral": "N", "Angry": "A", "Sad": "S", "Happy": "H",
            "Surprise": "U", "Fear": "F", "Disgust": "D",
            "Contempt": "C", "Other": "O"}


# ── Segment helpers ───────────────────────────────────────────────────────────

def _wire(args) -> None:
    """Resolve path constants from --src / --out."""
    src, out = paths.resolve(args, "longemo_dataset")
    globals().update(
        BASE=src / "msppodcast_full",
        CAND_DIR=out / "candidates",
        JSONL_DIR=out / "jsons",
        EPISODES_FILE=out / "src_dataset/test_episodes.jsonl",
    )


def load_segs(ep):
    segs = [json.loads(l)
            for l in open(JSONL_DIR / f"{ep}.jsonl", encoding="utf-8")]
    segs.sort(key=lambda s: s["start"])
    return segs


def segs_in(segs, t0, t1):
    return [s for s in segs if s["end"] > t0 and s["start"] < t1]


def dominant_spk(segs, t0, t1, thresh):
    cov = defaultdict(float)
    for s in segs_in(segs, t0, t1):
        g = s.get("global_speaker2", [])
        if len(g) == 1:
            cov[g[0]] += min(s["end"], t1) - max(s["start"], t0)
    if not cov:
        return None
    total = sum(cov.values())
    best, bc = max(cov.items(), key=lambda x: x[1])
    return best if bc / total >= thresh else None


def ctx_snippet(segs, t0, t1, chars=120):
    return " ".join(s.get("text", "") for s in segs_in(segs, t0, t1))[:chars]


def speaker_baselines(segs):
    avd = defaultdict(lambda: {"A": [], "V": [], "D": []})
    for s in segs:
        g = s.get("global_speaker2", [])
        if len(g) == 1 and s.get("num_speakers", 2) == 1:
            avd[g[0]]["A"].append(s.get("arousal", 0))
            avd[g[0]]["V"].append(s.get("valence", 0))
            avd[g[0]]["D"].append(s.get("dominance", 0))
    return {spk: {k: round(statistics.mean(v), 4) for k, v in dims.items()}
            for spk, dims in avd.items() if len(dims["A"]) >= 5}


# ── ST labels ─────────────────────────────────────────────────────────────────

def load_st_labels(ep):
    prefix = ep + "_"
    st = {}
    with open(POD_V2 / "Labels/labels_consensus.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["FileName"].startswith(prefix):
                continue
            n = row["FileName"][len(prefix):-4]
            st[n] = {"emo": row["EmoClass"], "spkr": row["SpkrID"],
                     "gender": row["Gender"],
                     "act": round((float(row["EmoAct"]) - 1) / 6, 3),
                     "val": round((float(row["EmoVal"]) - 1) / 6, 3),
                     "dom": round((float(row["EmoDom"]) - 1) / 6, 3),
                     "votes": {}}
    with open(POD_V2 / "Labels/labels_detailed.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["FileName"].startswith(prefix):
                continue
            n = row["FileName"][len(prefix):-4]
            if n in st:
                em = EMO_CODE.get(row["EmoClass_Major"], row["EmoClass_Major"])
                st[n]["votes"][em] = st[n]["votes"].get(em, 0) + 1
    for lab in st.values():
        tot = sum(lab["votes"].values())
        lab["cf"] = round(lab["votes"].get(lab["emo"], 0) / tot, 3) if tot else 0.0
    return st


def load_st_times(ep, segs, convo_offset):
    """Exact times from segments.json; text-match fallback for missing."""
    prefix = ep + "_"
    exact  = {}
    if convo_offset is not None:
        with open(CONVO / "Time_Labels/segments.json", encoding="utf-8") as f:
            seg_map = json.load(f)
        exact = {key[len(prefix):]: (float(sv["Start_Time"]) + convo_offset,
                                      float(sv["End_Time"])   + convo_offset)
                 for key, sv in seg_map.items() if key.startswith(prefix)}

    trans_dir = POD_V2 / "Transcripts"
    seg_words = [set(s.get("text", "").lower().split()) for s in segs]
    times  = dict(exact)
    prev_i = 0
    for n in sorted(set(exact) | {p.stem[len(prefix):]
                                   for p in trans_dir.glob(f"{prefix}*.txt")}):
        if n in times:
            t0     = times[n][0]
            prev_i = min(range(len(segs)),
                         key=lambda i: abs(segs[i]["start"] - t0))
            continue
        tp = trans_dir / f"{prefix}{n}.txt"
        if not tp.exists():
            continue
        pod_words = set(tp.read_text(encoding="utf-8").lower().split())
        if not pod_words:
            continue
        best_i = max(range(len(seg_words)),
                     key=lambda i: len(pod_words & seg_words[i]))
        if len(pod_words & seg_words[best_i]) > 0:
            idx    = max(best_i, prev_i)
            times[n] = (segs[idx]["start"], segs[idx]["end"])
            prev_i = idx
    return times


# ── CT traces ─────────────────────────────────────────────────────────────────

def load_ct_traces(convo_id, parts_info):
    ct = {}
    for dim in ("Arousal", "Valence", "Dominance"):
        pts = []
        for pid, t0 in parts_info:
            p = CONVO / "Traces" / dim / f"{pid}.csv"
            if not p.exists():
                continue
            with open(p, encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    abs_t = t0 + float(row["Central_Time"])
                    mean  = float(row["Mean"]) / 100
                    anns  = [float(row[k]) / 100 for k in row if k not in SKIP_COLS]
                    std   = statistics.stdev(anns) if len(anns) > 1 else 0.0
                    pts.append((abs_t, mean, std))
        ct[dim] = pts
    return ct


# ── Candidate finders ─────────────────────────────────────────────────────────

def find_st_events(segs, st, st_times, ep):
    trans_dir = POD_V2 / "Transcripts"
    prefix    = ep + "_"
    out = []
    for n, lab in sorted(st.items()):
        if lab["cf"] < ST_CF_MIN or n not in st_times:
            continue
        t0, t1 = st_times[n]
        spk = int(lab["spkr"]) if lab.get("spkr", "").isdigit() else None
        if spk is None:
            spk = dominant_spk(segs, t0, t1, ST_SPK_THRESH)
        if spk is None:
            continue
        ctx_before = [s for s in segs_in(segs, t0 - 60, t0)
                      if spk in (s.get("global_speaker2") or [])
                      and s.get("text", "")]
        if len(ctx_before) < 2:
            continue
        tp       = trans_dir / f"{prefix}{n}.txt"
        seg_text = tp.read_text(encoding="utf-8").strip() if tp.exists() else ""
        ctx      = " ".join(p for p in [
            ctx_snippet(segs, t0 - 20, t0, chars=300),
            seg_text,
            ctx_snippet(segs, t1, t1 + 20, chars=300),
        ] if p)
        out.append({"shape": "st_event", "n": n,
                    "t0": round(t0, 2), "t1": round(t1, 2), "spk": spk,
                    "emo": lab["emo"], "cf": lab["cf"], "votes": lab["votes"],
                    "act": lab["act"], "val": lab["val"], "dom": lab["dom"],
                    "seg_text": seg_text, "ctx": ctx})
    return out


def _flat_window(ct, segs, dim, excluded):
    """Flattest 30 s window for dim, avoiding excluded (t0, t1) ranges."""
    TWIN, TSTEP, HALF = 30.0, 10.0, 15.0
    all_t = [t for t, _, _ in ct[dim]]
    best, t = None, all_t[0]
    while t + TWIN <= all_t[-1]:
        if any(t < ex1 + 60 and t + TWIN > ex0 - 60 for ex0, ex1 in excluded):
            t += TSTEP
            continue
        spk = dominant_spk(segs, t, t + TWIN, CT_SPK_THRESH)
        if spk is None:
            t += TSTEP
            continue
        first  = [m for (at, m, _) in ct[dim] if t        <= at < t + HALF]
        second = [m for (at, m, _) in ct[dim] if t + HALF <= at <= t + TWIN]
        if not first or not second:
            t += TSTEP
            continue
        m1, m2 = statistics.mean(first), statistics.mean(second)
        if best is None or abs(m2 - m1) < best[0]:
            best = (abs(m2 - m1), t, t + TWIN, round(m1, 3), round(m2, 3), spk)
        t += TSTEP
    if best is None:
        return None
    _, ft0, ft1, fh1, fh2, spk = best
    return {"t0": ft0, "t1": ft1, "spk": spk, "half1": fh1, "half2": fh2}


def find_ct_trends(segs, ct):
    TWIN, TSTEP, HALF = 30.0, 10.0, 15.0
    raw = []
    for dim in ("Valence", "Arousal", "Dominance"):
        all_t = [t for t, _, _ in ct[dim]]
        t = all_t[0]
        while t + TWIN <= all_t[-1]:
            first  = [m for (at, m, _) in ct[dim] if t        <= at < t + HALF]
            second = [m for (at, m, _) in ct[dim] if t + HALF <= at <= t + TWIN]
            if first and second:
                m1, m2 = statistics.mean(first), statistics.mean(second)
                delta  = m2 - m1
                if abs(delta) > CT_TREND_DELTA:
                    spk = dominant_spk(segs, t, t + TWIN, CT_SPK_THRESH)
                    if spk:
                        raw.append({"dim": dim, "t0": t, "t1": t + TWIN,
                                    "spk": spk, "delta": delta,
                                    "half1": round(m1, 3), "half2": round(m2, 3)})
            t += TSTEP

    out = []
    for dim in ("Valence", "Arousal", "Dominance"):
        dim_cands = sorted([c for c in raw if c["dim"] == dim],
                           key=lambda c: abs(c["delta"]), reverse=True)
        kept = []
        for c in dim_cands:
            if any(c["t0"] < k["t1"] + 60 and c["t1"] > k["t0"] - 60
                   for k in kept):
                continue
            kept.append(c)
        for c in kept:
            c = dict(c)
            c["shape"]     = "ct_trend"
            c["delta"]     = round(c["delta"], 3)
            c["direction"] = "rising" if c["delta"] > 0 else "falling"
            c["ctx"]       = ctx_snippet(segs, c["t0"], c["t1"])
            out.append(c)

    excluded = [(c["t0"], c["t1"]) for c in out]
    for dim in ("Valence", "Arousal", "Dominance"):
        fw = _flat_window(ct, segs, dim, excluded)
        if fw:
            fw.update({"shape": "ct_trend", "dim": dim, "direction": "none",
                       "delta": round(fw["half2"] - fw["half1"], 3),
                       "ctx": ctx_snippet(segs, fw["t0"], fw["t1"])})
            out.append(fw)
    return out


def find_multi_spk_windows(segs):
    WIN, STEP = 60.0, 30.0
    out = []
    t = segs[0]["start"]
    while t + WIN <= segs[-1]["end"]:
        cov = defaultdict(float)
        for s in segs_in(segs, t, t + WIN):
            g = s.get("global_speaker2", [])
            if len(g) == 1:
                cov[g[0]] += min(s["end"], t + WIN) - max(s["start"], t)
        active = [sp for sp, c in cov.items() if c > 10]
        if len(active) >= 2:
            best_dim, best_diff, best_avd = None, 0.0, {}
            for dim, key in (("Valence", "valence"), ("Arousal", "arousal"),
                             ("Dominance", "dominance")):
                avd = {sp: statistics.mean(
                           [s.get(key, 0) for s in segs_in(segs, t, t + WIN)
                            if s.get("global_speaker2") == [sp]
                            and s.get("num_speakers", 2) == 1])
                       for sp in active
                       if any(s.get("global_speaker2") == [sp]
                              and s.get("num_speakers", 2) == 1
                              for s in segs_in(segs, t, t + WIN))}
                if len(avd) >= 2:
                    diff = max(avd.values()) - min(avd.values())
                    if diff > best_diff:
                        best_dim, best_diff, best_avd = dim, diff, avd
            if best_dim and best_diff > MULTI_DIFF:
                out.append({"shape": "multi_spk_window",
                            "t0": t, "t1": t + WIN, "dim": best_dim,
                            "spk_val": {k: round(v, 3)
                                        for k, v in best_avd.items()},
                            "val_diff": round(best_diff, 3),
                            "ctx": ctx_snippet(segs, t, t + WIN)})
        t += STEP
    return out


# ── Topic generation ──────────────────────────────────────────────────────────

def gen_topic(pipe, ctx):
    messages = [
        {"role": "system", "content":
            "Give a topic phrase (3-7 words) for what is being discussed. "
            "Output only the phrase, nothing else."},
        {"role": "user", "content": ctx[:200]},
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    out = pipe(prompt, return_full_text=False)[0]["generated_text"]
    return out.strip().strip("\".,: '").split("\n")[0].strip()[:60] or "this topic"


# ── Episode processing ────────────────────────────────────────────────────────

def process_episode(ep, out_dir, pipe, excluded_segs=None):
    ep_num   = ep.split("_")[-1]
    convo_id = f"MSP-Conversation_{ep_num}"

    convo_offset = None
    with open(CONVO / "Time_Labels/conversations.txt") as f:
        for line in f:
            p = line.strip().split(";")
            if p[0] == convo_id:
                convo_offset = float(p[1])
                break
    if convo_offset is None:
        print(f"  {ep}: not in MSP-Conversation — ct_trend unavailable")

    parts_info = []
    if convo_offset is not None:
        for fname in ("conversation_parts.txt", "test2_conversation_parts.txt"):
            fpath = CONVO / "Time_Labels" / fname
            if not fpath.exists():
                continue
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    pid, t0, _ = line.split(";")
                    if pid.startswith(convo_id + "_"):
                        parts_info.append((pid, float(t0)))

    segs      = load_segs(ep)
    baselines = speaker_baselines(segs)
    st        = load_st_labels(ep)
    st_times  = load_st_times(ep, segs, convo_offset)
    excl      = set(excluded_segs or [])

    candidates = [c for c in find_st_events(segs, st, st_times, ep)
                  if ep + "_" + c["n"] not in excl]
    if parts_info:
        candidates += find_ct_trends(segs, load_ct_traces(convo_id, parts_info))
    candidates += find_multi_spk_windows(segs)

    shapes  = [c["shape"] for c in candidates]
    summary = "  ".join(f"{s}:{shapes.count(s)}" for s in sorted(set(shapes)))
    print(f"  {ep}: {len(candidates)} candidates  [{summary}]")

    if pipe is not None:
        for c in candidates:
            c["topic"] = gen_topic(pipe, c["ctx"]) if c.get("ctx") else "this topic"

    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"{ep}.json", "w", encoding="utf-8") as f:
        json.dump({"episode": ep, "convo_id": convo_id,
                   "convo_offset": convo_offset,
                   "baselines": baselines,
                   "candidates": candidates},
                  f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_path_args(ap)
    ap.add_argument("--episode", default=None)
    ap.add_argument("--rerun", action="store_true")
    args    = ap.parse_args()
    _wire(args)
    out_dir = CAND_DIR

    episodes, excluded_map = [], {}
    if args.episode:
        episodes = [args.episode]
    else:
        for line in open(EPISODES_FILE, encoding="utf-8"):
            if not line.strip() or line.startswith("//"):
                continue
            rec = json.loads(line)
            ep  = f"MSP-PODCAST_{rec['conversation'].split('_')[-1]}"
            episodes.append(ep)
            if rec.get("pod_train_segs"):
                excluded_map[ep] = rec["pod_train_segs"]

    if not args.rerun:
        episodes = [ep for ep in episodes
                    if not (out_dir / f"{ep}.json").exists()]

    if not episodes:
        print("Nothing to do.")
        return

    pipe = hf_pipeline("text-generation", model=TOPIC_MODEL,
                       device_map="auto", max_new_tokens=20, do_sample=False)
    print(f"Processing {len(episodes)} episodes ...")
    for i, ep in enumerate(episodes, 1):
        print(f"[{i}/{len(episodes)}] {ep}")
        process_episode(ep, out_dir, pipe,
                        excluded_segs=excluded_map.get(ep))
    print("Done.")


if __name__ == "__main__":
    main()
