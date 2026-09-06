"""
Split all_data.json into per-episode JSONL files.
Each line = one segment as JSON, with episode_id and segment_id added.
Output: jsons/{episode_id}.jsonl
"""
import argparse
import json

from dataprep import paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    paths.add_path_args(ap)
    src, out = paths.resolve(ap.parse_args(), "longemo_dataset")
    SRC = src / "msppodcast_full/all_data.json"
    OUT = out / "jsons"
    OUT.mkdir(exist_ok=True)

    print("Loading all_data.json ...", flush=True)
    with open(SRC, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    for i, (ep_id, ep) in enumerate(data.items(), 1):
        out_path = OUT / f"{ep_id}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for seg_id, (_, seg) in enumerate(ep.items()):
                row = {"episode_id": ep_id, "segment_id": seg_id, **seg}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if i % 500 == 0 or i == total:
            print(f"  {i}/{total} episodes written", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
