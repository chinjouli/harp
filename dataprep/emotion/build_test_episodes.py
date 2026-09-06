"""
Filter Convo-Test1 episodes for query generation:
  - < 10 pod-Train segments
  - pod-Train segs < 10% of pod-Test1 segs
Also records the pod-Train segment IDs to exclude per episode.
Outputs: test_episodes.json
"""
import argparse
import json
from collections import defaultdict

from dataprep import paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    paths.add_path_args(ap)
    src, out = paths.resolve(ap.parse_args(), "longemo_dataset")
    CONV_DIR = src / "msppodcast_full/msp_conversation_v2.0"
    POD_DIR  = src / "msppodcast_full/msp_podcast_v2.0"
    OUT      = out / "src_dataset/test_episodes.jsonl"
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # -- 1. Convo-Test1 episodes -------------------------------------------------
    convo_test = set()
    with open(CONV_DIR / "partitions.txt") as f:
        for line in f:
            ep, split = line.strip().split(";")
            if split == "Test1":
                convo_test.add(ep.strip())

    # -- 2. Pod split per segment -------------------------------------------------
    pod_split = {}
    with open(POD_DIR / "Partitions.txt") as f:
        for line in f:
            split, fname = line.strip().split(";")
            seg_id = fname.strip().replace(".wav", "")
            pod_split[seg_id] = split.strip()

    # -- 3. Map pod segments -> conversation --------------------------------------
    with open(CONV_DIR / "Time_Labels" / "segments.json") as f:
        seg_map = json.load(f)  # seg_id -> {Conversation, ...}

    # per convo-test episode: lists of pod-train / pod-test1 seg IDs
    ep_train = defaultdict(list)
    ep_test1 = defaultdict(list)

    for seg_id, info in seg_map.items():
        conv = info["Conversation"]
        if conv not in convo_test:
            continue
        split = pod_split.get(seg_id)
        if split == "Train":
            ep_train[conv].append(seg_id)
        elif split == "Test1":
            ep_test1[conv].append(seg_id)

    # -- 4. Filter ----------------------------------------------------------------
    results = []
    excluded = []

    for ep in sorted(convo_test):
        n_train = len(ep_train[ep])
        n_test1 = len(ep_test1[ep])
        ratio = n_train / n_test1 if n_test1 > 0 else float("inf")
        if n_train < 10 and ratio < 0.10:
            results.append({
                "conversation": ep,
                "pod_train_count": n_train,
                "pod_test1_count": n_test1,
                "pod_train_segs": sorted(ep_train[ep]),
            })
        else:
            excluded.append({
                "ep": ep,
                "pod_train": n_train,
                "pod_test1": n_test1,
                "ratio_pct": round(ratio * 100, 1),
            })

    print(f"Included: {len(results)} / {len(convo_test)} Convo-Test1 episodes")
    print(f"Excluded: {len(excluded)}")
    for e in excluded:
        print(f"  {e['ep']}  train={e['pod_train']}  test1={e['pod_test1']}"
              f"  ratio={e['ratio_pct']}%")

    with open(OUT, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
