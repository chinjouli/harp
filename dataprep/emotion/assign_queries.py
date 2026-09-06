"""Assign query types to candidate events for one episode.

Reads candidates/{ep}.json, outputs assignments/{ep}.json.

Shape → eligible query types:
  st_event         → state, locate  (reusable across both)
  ct_trend         → change
  multi_spk_window → comparison

Usage:
    python dataprep/emotion/assign_queries.py --episode MSP-PODCAST_0002
    python dataprep/emotion/assign_queries.py  # all test episodes
"""
import argparse
import json
import random

from dataprep import paths


NEUTRAL = {"N", "X"}

# Most-constrained first so scarce candidates fill harder slots first.
QUERY_ORDER = ["comparison", "change", "state", "locate"]

def _wire(args) -> None:
    """Resolve path constants from --src / --out."""
    src, out = paths.resolve(args, "longemo_dataset")
    globals().update(
        CAND_DIR=out / "candidates",
        OUT_DIR=out / "assignments",
        EPISODES_FILE=out / "src_dataset/test_episodes.jsonl",
    )


def score(c):
    if c["shape"] == "st_event":       return c.get("cf", 0)
    if c["shape"] == "ct_trend":       return abs(c.get("delta", 0))
    if c["shape"] == "multi_spk_window": return c.get("val_diff", 0)
    return 0


def eligible(qtype, c):
    if qtype == "state":      return c["shape"] == "st_event"
    if qtype == "locate":     return c["shape"] == "st_event" and c.get("emo") not in NEUTRAL
    if qtype == "change":     return c["shape"] == "ct_trend"
    if qtype == "comparison": return c["shape"] == "multi_spk_window"
    return False


def stratify(pool, cap, seed):
    if len(pool) <= cap:
        return pool
    rng    = random.Random(seed)
    bucket = len(pool) / cap
    return [rng.choice(pool[int(k * bucket):int((k + 1) * bucket)])
            for k in range(cap)]


def assign(candidates, cap, seed=42):
    used    = set()
    result  = []
    indexed = list(enumerate(candidates))

    for qtype in QUERY_ORDER:
        pool = [(i, c) for i, c in indexed
                if eligible(qtype, c) and (c["shape"] == "st_event" or i not in used)]
        pool.sort(key=lambda x: score(x[1]), reverse=True)

        if qtype == "change":
            none_pool = [(i, c) for i, c in pool if c.get("direction") == "none"]
            real_pool = [(i, c) for i, c in pool if c.get("direction") != "none"]
            none_cap  = min(1, len(none_pool))
            chosen    = (stratify(none_pool, none_cap, seed) +
                         stratify(real_pool, cap - none_cap, seed))
        else:
            chosen = stratify(pool, cap, seed)

        for i, c in chosen:
            if c["shape"] != "st_event":
                used.add(i)
            result.append({"query_type": qtype, "candidate": c})

    return result


def process_episode(ep, cap, seed):
    src = CAND_DIR / f"{ep}.json"
    if not src.exists():
        print(f"  {ep}: missing candidates, skipping.")
        return
    data        = json.load(open(src, encoding="utf-8"))
    assignments = assign(data["candidates"], cap=cap, seed=seed)
    counts      = {qt: sum(1 for a in assignments if a["query_type"] == qt)
                   for qt in QUERY_ORDER}
    print(f"  {ep}: {len(assignments)}  "
          f"[{'  '.join(f'{qt}:{counts[qt]}' for qt in QUERY_ORDER)}]")
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / f"{ep}.json", "w", encoding="utf-8") as f:
        json.dump({"episode": ep, "cap": cap,
                   "baselines": data.get("baselines", {}),
                   "assignments": assignments},
                  f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_path_args(ap)
    ap.add_argument("--episode", default=None)
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
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
                    if not (OUT_DIR / f"{ep}.json").exists()]

    if not episodes:
        print("Nothing to do.")
        return

    print(f"Processing {len(episodes)} episodes ...")
    for ep in episodes:
        process_episode(ep, args.cap, args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
