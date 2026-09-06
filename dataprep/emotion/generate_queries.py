"""Generate queries for one episode.

Reads assignments/{ep}.json, outputs queries/{ep}.jsonl.

Usage:
    python dataprep/emotion/generate_queries.py --episode MSP-PODCAST_0002
    python dataprep/emotion/generate_queries.py  # all test episodes
"""
import argparse
import json
import random
import re
import statistics
from collections import defaultdict

from dataprep import paths

from transformers import pipeline as hf_pipeline

QWEN_MODEL    = "Qwen/Qwen3.5-0.8B"

MAX_LOCATE_EXPAND      = 150.0  # 2.5 min each direction → 5 min max window
CMP_PAD_TOTAL          = 240    # total padding added to comparison window
CMP_PAD_MIN            = 10    # minimum padding per side
LOCATE_HARD_WIN    = 300.0  # max cluster span
LOCATE_HARD_PAD    = 30.0   # padding around cluster edges
LOCATE_HARD_CAP    = 3

_SENT_STARTS = {
    "i", "we", "you", "he", "she", "they", "it",
    "welcome", "this", "that", "there", "so", "and", "but",
}


# ── Text helpers ──────────────────────────────────────────────────────────────

def _wire(args) -> None:
    """Resolve path constants from --src / --out."""
    src, out = paths.resolve(args, "longemo_dataset")
    globals().update(
        ASSIGN_DIR=out / "assignments",
        CAND_DIR=out / "candidates",
        OUT_DIR=out / "queries",
        EPISODES_FILE=out / "src_dataset/test_episodes.jsonl",
        SPK_INDEX=out / "spk_clip_index.json",
        EMO_INDEX=out / "emo_clip_index.json",
        JSONL_DIR=out / "jsons",
        CONVO_TIMES=src / "msppodcast_full/msp_conversation_v2.0/Time_Labels/conversations.txt",
    )


def pick_clause(text, rng, min_w=4, max_w=12):
    parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+|,\s+', text)
             if p.strip()]
    fits  = [p for p in parts if min_w <= len(p.split()) <= max_w]
    return rng.choice(fits or parts) if (fits or parts) else ""


def fmt_time(secs):
    half = round(secs / 30)
    mins = half // 2
    return (f"around {mins} minute{'s' if mins != 1 else ''}"
            if half % 2 == 0 else f"around {mins} and a half minutes")


def format_topic(topic, pipe):
    if not topic:
        return "this topic"
    lower = " ".join(
        w if (len(w.strip(".,!?;:'\"")) > 1
              and w.strip(".,!?;:'\"").isupper()) else w.lower()
        for w in topic.split()
    )
    if topic.split()[0].lower() not in _SENT_STARTS:
        return lower
    if pipe is None:
        return lower
    messages = [
        {"role": "system", "content":
            "Convert to a short noun phrase (3-6 words) that fits after "
            "'talking about'. Output only the phrase."},
        {"role": "user", "content": topic},
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    out = pipe(prompt, return_full_text=False)[0]["generated_text"]
    out = out.strip().strip("\".,: '").split("\n")[0].lower()
    return out or "this topic"


# ── Audio clip pickers ────────────────────────────────────────────────────────

def _spk_gender_in_ep(global_spk, ep):
    # global_speaker2 clusters can mix genders across episodes; look up the
    # speaker's actual gender in this specific episode and use it to filter
    # the example pool so the provided clip matches who the annotators heard.
    fp = JSONL_DIR / f"{ep}.jsonl"
    if not fp.exists():
        return None
    counts: dict[str, int] = {}
    for line in open(fp, encoding="utf-8"):
        s = json.loads(line)
        if global_spk in s.get("global_speaker2", []):
            g = s.get("gender", "?")
            counts[g] = counts.get(g, 0) + 1
    return max(counts, key=counts.get) if counts else None


def pick_spk_clip(spk_idx, global_spk, exclude_ep, rng):
    gender = _spk_gender_in_ep(global_spk, exclude_ep)
    pool = [x for x in spk_idx.get(str(global_spk), [])
            if x.get("wav") and x.get("ep") != exclude_ep
            and (gender is None or x.get("gender") == gender)]
    if not pool:
        # fallback: ignore gender filter if no matching clip available
        pool = [x for x in spk_idx.get(str(global_spk), [])
                if x.get("wav") and x.get("ep") != exclude_ep]
    if not pool:
        raise ValueError(f"No speaker clip: global_spk={global_spk}, ep={exclude_ep}")
    return rng.choice(pool)["wav"]


def pick_emo_clip(emo_idx, emo_code, exclude_ep, exclude_spk, rng):
    pool = [x for x in emo_idx.get(emo_code, [])
            if x.get("wav") and x.get("ep") != exclude_ep
            and x.get("global_spk") != exclude_spk]
    if not pool:
        raise ValueError(f"No emotion clip: emo={emo_code}, ep={exclude_ep}")
    return rng.choice(pool)["wav"]


def pick_emo_clip_closest(emo_idx, emo_code, target_cf, target_act, target_val,
                          target_dom, exclude_ep, exclude_spk, rng):
    """Pick clip closest to target in (cf, act, val, dom) space, diff speaker."""
    pool = [x for x in emo_idx.get(emo_code, [])
            if x.get("wav") and x.get("ep") != exclude_ep
            and x.get("global_spk") != exclude_spk]
    if not pool:
        raise ValueError(f"No emotion clip: emo={emo_code}, ep={exclude_ep}")

    def dist(x):
        return (abs(x.get("cf", 0) - target_cf)
                + ((x.get("act", 0.5) - target_act) ** 2
                   + (x.get("val", 0.5) - target_val) ** 2
                   + (x.get("dom", 0.5) - target_dom) ** 2) ** 0.5)

    pool.sort(key=dist)
    return rng.choice(pool[:min(3, len(pool))])["wav"]


# ── Window helpers ────────────────────────────────────────────────────────────

def load_convo_bounds(ep):
    ep_num   = ep.split("_")[-1]
    convo_id = f"MSP-Conversation_{ep_num}"
    with open(CONVO_TIMES, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(";")
            if p[0] == convo_id:
                return float(p[1]), float(p[2])
    return None, None


def expand_locate_window(c, all_st, conv_start, conv_end):
    """Push locate window outward until same-spk/same-emo neighbor or conv edge."""
    emo, spk = c["emo"], c["spk"]
    same = [e for e in all_st
            if e.get("shape") == "st_event" and e.get("emo") == emo
            and e.get("spk") == spk and e.get("n") != c["n"]
            and e.get("cf", 0) >= 0.5]
    before = [e for e in same if e["t1"] <= c["t0"]]
    after  = [e for e in same if e["t0"] >= c["t1"]]
    # Only use conv bounds when they are on the correct side of the target event
    left_default  = (conv_start if conv_start is not None and conv_start < c["t0"]
                     else max(0, c["t0"] - MAX_LOCATE_EXPAND))
    right_default = (conv_end if conv_end is not None and conv_end > c["t1"]
                     else c["t1"] + MAX_LOCATE_EXPAND)
    left  = max((e["t1"] for e in before), default=left_default)
    right = min((e["t0"] for e in after),  default=right_default)
    win_t0 = max(left,  c["t0"] - MAX_LOCATE_EXPAND)
    win_t1 = min(right, c["t1"] + MAX_LOCATE_EXPAND)
    return win_t0, win_t1


# ── locate_hard cluster finder ────────────────────────────────────────────────

def find_locate_hard_clusters(all_st, conv_start, conv_end):
    """Find windows with 3+ same-spk/same-emo events; one cluster per (spk,emo)."""
    NEUTRAL = {"N", "X"}
    by_spk_emo = defaultdict(list)
    for evt in all_st:
        if (evt.get("shape") != "st_event" or evt.get("emo") in NEUTRAL
                or evt.get("cf", 0) < 0.5 or evt.get("spk") is None):
            continue
        by_spk_emo[(evt["spk"], evt["emo"])].append(evt)

    out = []
    for (spk, emo), events in by_spk_emo.items():
        events = sorted(events, key=lambda e: e["t0"])
        for i in range(len(events)):
            j = i + 1
            while (j < len(events)
                   and events[j]["t1"] - events[i]["t0"] <= LOCATE_HARD_WIN):
                j += 1
            group = events[i:j]
            if len(group) < 3:
                continue
            centroid = {d: statistics.mean(e[d] for e in group)
                        for d in ("act", "val", "dom")}
            target = max(group, key=lambda e: sum(
                (e[d] - centroid[d]) ** 2 for d in ("act", "val", "dom")))
            left   = (conv_start if conv_start is not None and conv_start < group[0]["t0"]
                      else max(0, group[0]["t0"] - LOCATE_HARD_PAD))
            right  = (conv_end if conv_end is not None and conv_end > group[-1]["t1"]
                      else group[-1]["t1"] + LOCATE_HARD_PAD)
            win_t0 = max(group[0]["t0"] - LOCATE_HARD_PAD, left)
            win_t1 = min(group[-1]["t1"] + LOCATE_HARD_PAD, right)
            if win_t0 >= win_t1:
                continue
            out.append({
                "emo": emo, "spk": spk,
                "t0": win_t0, "t1": win_t1,
                "target_n":   target["n"],
                "target_t0":  target["t0"],  "target_t1":  target["t1"],
                "target_cf":  target["cf"],
                "target_act": target["act"], "target_val": target["val"],
                "target_dom": target["dom"],
                "topic":    target.get("topic", "this topic"),
                "seg_text": target.get("seg_text", ""),
                "n_events": len(group),
            })
            break  # one cluster per (spk, emo)
    return out


# ── Query builder ─────────────────────────────────────────────────────────────

def make_query(qt, c, topic, labels, ep, spk_idx, emo_idx, rng):
    def lbl(gid):
        return labels.get(str(gid), str(gid))

    spk_id  = c.get("spk")
    spk_lbl = lbl(spk_id) if spk_id is not None else None
    t0      = c.get("t0") or c.get("t_boundary")
    audio_spk, audio_spk2, audio_emo = None, None, None

    if qt == "state":
        audio_spk = pick_spk_clip(spk_idx, spk_id, ep, rng)
        quote   = pick_clause(c.get("seg_text") or c.get("ctx", ""), rng)
        primary = c["emo"]
        q     = (f"Speaker: [audio example]. {fmt_time(t0).capitalize()}, "
                 f"this speaker was talking about {topic} and mentioned "
                 f"\"{quote}\". What emotion(s) did this speaker show?")
        gt    = {"emo": primary, "cf": c["cf"], "votes": c["votes"],
                 "secondary": {k: v for k, v in c["votes"].items()
                               if k != primary},
                 "act": c.get("act"), "val": c.get("val"), "dom": c.get("dom")}
        judge = {"type": "categorical", "primary": primary,
                 "secondary": gt["secondary"], "votes": c["votes"],
                 "cf": c["cf"],
                 "valence": c.get("val"), "arousal": c.get("act"),
                 "dominance": c.get("dom")}

    elif qt == "change":
        audio_spk = pick_spk_clip(spk_idx, spk_id, ep, rng)
        q     = (f"Speaker: [audio example]. From {int(c['t0'])} to "
                 f"{int(c['t1'])} seconds, how did this speaker's "
                 f"{c['dim'].lower()} change while discussing {topic}?")
        gt    = {"direction": c["direction"], "delta": c["delta"],
                 "half1": c["half1"], "half2": c["half2"]}
        judge = {"type": "trend", "direction": c["direction"],
                 "half1": c["half1"], "half2": c["half2"]}

    elif qt == "comparison":
        spks       = sorted(c["spk_val"].keys(), key=int)
        winner     = max(c["spk_val"], key=c["spk_val"].get)
        audio_spk  = pick_spk_clip(spk_idx, int(spks[0]), ep, rng)
        audio_spk2 = pick_spk_clip(spk_idx, int(spks[1]), ep, rng)
        pad_l  = rng.randint(CMP_PAD_MIN, CMP_PAD_TOTAL - CMP_PAD_MIN)
        pad_r  = CMP_PAD_TOTAL - pad_l
        q_t0   = max(0, int(c["t0"]) - pad_l)
        q_t1   = int(c["t1"]) + pad_r
        pos_winner = "A" if winner == spks[0] else "B"
        pos_spk_val = {"A": c["spk_val"][spks[0]], "B": c["spk_val"][spks[1]]}
        q      = (f"Speaker A: [A's audio example].\n"
                  f"Speaker B: [B's audio example].\n"
                  f"From {q_t0} to {q_t1} seconds discussing "
                  f"{topic}, which speaker had higher {c['dim'].lower()} "
                  f"— Speaker A or Speaker B?")
        gt     = {"winner": pos_winner, "dim": c["dim"], "spk_val": pos_spk_val}
        judge  = {"type": "speaker_id", "dim": c["dim"],
                  "winner": pos_winner, "spk_val": pos_spk_val}

    elif qt == "locate":
        audio_spk = pick_spk_clip(spk_idx, spk_id, ep, rng)
        audio_emo = pick_emo_clip(emo_idx, c["emo"], ep, spk_id, rng)
        quote     = pick_clause(c.get("seg_text") or c.get("ctx", ""), rng)
        win_t0    = c.get("win_t0", c["t0"])
        win_t1    = c.get("win_t1", c["t1"])
        q     = (f"Speaker: [speaker example].\nEmotion: [emotion example].\n"
                 f"Between {int(win_t0)} and {int(win_t1)} seconds, when did "
                 f"this speaker show the same emotion as in the audio clip, "
                 f"and what were they saying?")
        gt    = {"time_ref": fmt_time(t0), "t0": int(c["t0"]),
                 "t1": int(c["t1"]), "quote": quote, "emo": c["emo"]}
        judge = {"type": "time_and_quote", "t0": int(c["t0"]),
                 "t1": int(c["t1"]), "emo": c["emo"],
                 "seg_text": c.get("seg_text", ""), "time_tolerance_s": 15}

    elif qt == "locate_hard":
        audio_emo = pick_emo_clip_closest(
            emo_idx, c["emo"],
            c["target_cf"], c["target_act"], c["target_val"], c["target_dom"],
            ep, spk_id, rng)
        quote = pick_clause(c.get("seg_text", ""), rng)
        q     = (f"Emotion: [audio example].\nBetween {int(c['t0'])} and "
                 f"{int(c['t1'])} seconds, when did someone show the same "
                 f"emotion as in the audio clip, and what were they saying?")
        gt    = {"time_ref": fmt_time(c["target_t0"]),
                 "t0": int(c["target_t0"]), "t1": int(c["target_t1"]),
                 "quote": quote, "emo": c["emo"], "n_events": c["n_events"]}
        judge = {"type": "time_and_quote",
                 "t0": int(c["target_t0"]), "t1": int(c["target_t1"]),
                 "emo": c["emo"], "seg_text": c.get("seg_text", ""),
                 "time_tolerance_s": 15, "n_events": c["n_events"]}

    else:
        raise ValueError(f"Unknown query type: {qt}")

    return q, audio_spk, audio_spk2, audio_emo, gt, judge


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(ep, pipe, rng, spk_idx, emo_idx):
    src = ASSIGN_DIR / f"{ep}.json"
    if not src.exists():
        print(f"  {ep}: missing assignments, skipping.")
        return 0

    data   = json.load(open(src, encoding="utf-8"))
    labels = {sid: chr(65 + i)
              for i, sid in enumerate(
                  sorted(data.get("baselines", {}).keys(), key=int))}

    # Load candidates + conversation bounds for window expansion
    all_st, conv_start, conv_end = [], None, None
    cand_src = CAND_DIR / f"{ep}.json"
    if cand_src.exists():
        all_st = json.load(open(cand_src, encoding="utf-8")).get("candidates", [])
        conv_start, conv_end = load_convo_bounds(ep)

    OUT_DIR.mkdir(exist_ok=True)
    count = 0
    with open(OUT_DIR / f"{ep}.jsonl", "w", encoding="utf-8") as fout:

        for a in data["assignments"]:
            c     = a["candidate"]
            topic = format_topic(c.get("topic", "this topic"), pipe)
            # Expand window for locate queries
            if a["query_type"] == "locate" and all_st and conv_start is not None:
                win_t0, win_t1 = expand_locate_window(c, all_st, conv_start, conv_end)
                c = dict(c, win_t0=win_t0, win_t1=win_t1)
            try:
                q, audio_spk, audio_spk2, audio_emo, gt, judge = make_query(
                    a["query_type"], c, topic, labels, ep, spk_idx, emo_idx, rng)
            except ValueError as e:
                print(f"  {ep} [{a['query_type']}] skipped: {e}")
                continue
            rec = {
                "episode_id":   ep,
                "query_type":   a["query_type"],
                "query":        q,
                "time_ref":     c.get("t0") or c.get("t_boundary") or 0,
                "topic":        topic,
                "audio_spk":    audio_spk,
                "ground_truth": gt,
                "judge_info":   judge,
            }
            if audio_spk2 is not None:
                rec["audio_spk2"] = audio_spk2
            if audio_emo is not None:
                rec["audio_emo"] = audio_emo
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

        # locate_hard: mine clusters from existing candidates, cap at 3
        if all_st and conv_start is not None:
            clusters = find_locate_hard_clusters(all_st, conv_start, conv_end)
            for cluster in clusters[:LOCATE_HARD_CAP]:
                topic = format_topic(cluster.get("topic", "this topic"), pipe)
                try:
                    q, _, _, audio_emo, gt, judge = make_query(
                        "locate_hard", cluster, topic, labels, ep,
                        spk_idx, emo_idx, rng)
                except ValueError as e:
                    print(f"  {ep} [locate_hard] skipped: {e}")
                    continue
                rec = {
                    "episode_id":   ep,
                    "query_type":   "locate_hard",
                    "query":        q,
                    "time_ref":     cluster["target_t0"],
                    "topic":        topic,
                    "ground_truth": gt,
                    "judge_info":   judge,
                }
                if audio_emo is not None:
                    rec["audio_emo"] = audio_emo
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1

    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_path_args(ap)
    ap.add_argument("--episode", default=None)
    ap.add_argument("--dry-run", action="store_true", help="Skip Qwen calls")
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()
    _wire(args)

    if args.episode:
        episodes = [args.episode]
    else:
        episodes = [
            f"MSP-PODCAST_{json.loads(l)['conversation'].split('_')[-1]}"
            for l in open(EPISODES_FILE, encoding="utf-8")
            if l.strip() and not l.startswith("//")
        ]

    if not args.rerun:
        episodes = [ep for ep in episodes
                    if not (OUT_DIR / f"{ep}.jsonl").exists()]

    if not episodes:
        print("Nothing to do.")
        return

    spk_idx = json.load(open(SPK_INDEX, encoding="utf-8")) \
        if SPK_INDEX.exists() else {}
    emo_idx = json.load(open(EMO_INDEX, encoding="utf-8")) \
        if EMO_INDEX.exists() else {}
    pipe = None if args.dry_run else hf_pipeline(
        "text-generation", model=QWEN_MODEL,
        device_map="auto", max_new_tokens=20, do_sample=False)
    rng = random.Random(42)

    print(f"Processing {len(episodes)} episodes ...")
    total = sum(
        run_episode(ep, pipe, rng, spk_idx, emo_idx)
        for ep in episodes
    )
    print(f"Done. {total} queries written.")


if __name__ == "__main__":
    main()
