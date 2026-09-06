"""Score human evaluation outcome files in human_eval/outcome/.

Usage:
    python human_eval/score_human.py          # all outcome files
    python human_eval/score_human.py health   # task filter
    python human_eval/score_human.py emotion music
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

OUTCOME_DIR = Path("human_eval/outcome")
OPTIONS_PATH = Path("human_eval/query_options.json")

QUERIES_PATHS = {
    "health":  Path("data/medmosaic/queries.jsonl"),
    "emotion": Path("data/mspemotion/queries.jsonl"),
    "music":   Path("data/songeval/queries.jsonl"),
}

EXCLUDE_PATHS = {
    "emotion": Path("data/mspemotion/exclude.txt"),
}

# Tolerance in seconds for locate/locate_hard timestamp matching
LOCATE_TOL = 5

# "observed" is how the HTML labels "exhibited"
QT_ALIASES = {"observed": "exhibited"}

# display order per task
TASK_ORDERS = {
    "health":  ["reported", "exhibited"],
    "emotion": ["state", "change", "comparison", "locate", "locate_hard"],
    "music":   ["time", "position", "audio", "audio_top"],
}

# music: main files have full-song audio (topline); patch files have snippets
MUSIC_MAIN_TYPES  = {"time", "position", "audio"}
MUSIC_PATCH_TYPES = {"audio"}

# emotion state: MCQ label → GT emo code
_STATE_LABEL_TO_CODE = {
    "A": "N", "B": "H", "C": "S", "D": "A",
    "E": "U", "F": "F", "G": "D", "H": "C",
}

# emotion change: MCQ label → GT direction string
_CHANGE_LABEL_TO_DIR = {"A": "rising", "B": "falling", "C": "none"}


def load_options() -> dict:
    with open(OPTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_gt(task: str) -> dict[int, dict]:
    p = QUERIES_PATHS.get(task)
    if not p or not p.exists():
        return {}
    gt = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            gt[q["query_id"]] = q
    return gt


def load_exclude(task: str) -> set[int]:
    p = EXCLUDE_PATHS.get(task)
    if not p or not p.exists():
        return set()
    excluded = set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                excluded.add(int(line.split()[0]))
            except (ValueError, IndexError):
                pass
    return excluded


def _parse_mmss(s: str) -> float | None:
    """Parse 'mm:ss' or 'h:mm:ss' to seconds."""
    try:
        parts = s.strip().split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, AttributeError):
        pass
    return None


def is_correct(choice, qt: str, task: str, q: dict) -> bool | None:
    gt = q.get("ground_truth", {})

    if task == "emotion":
        if qt == "comparison":
            # choice "A"/"B" vs GT winner "A"/"B"
            return choice == gt.get("winner")
        if qt == "change":
            gt_dir = gt.get("direction", "")
            return _CHANGE_LABEL_TO_DIR.get(choice) == gt_dir
        if qt == "state":
            # multi-select: correct if GT primary emo code is among chosen labels
            if not isinstance(choice, list):
                choice = [choice]
            chosen_codes = {_STATE_LABEL_TO_CODE.get(c) for c in choice}
            return gt.get("emo") in chosen_codes
        if qt in ("locate", "locate_hard"):
            t = _parse_mmss(str(choice)) if choice else None
            if t is None:
                return None
            t0, t1 = gt.get("t0", 0), gt.get("t1", 0)
            return (t0 - LOCATE_TOL) <= t <= (t1 + LOCATE_TOL)
        return None

    if task == "music":
        winner = gt.get("winner", "")
        # A→"A", B→"B", C→"Tie"/"tie"
        chosen = {"A": "A", "B": "B", "C": "Tie"}.get(choice, choice)
        return chosen.lower() == winner.lower()

    if task == "health":
        chosen = {"A": "Yes", "B": "No"}.get(choice, choice)
        return chosen.lower() == gt.get("answer", "").lower()

    return None


def detect_task(filename: str) -> str | None:
    stem = filename.removesuffix(".json")
    for task in ("health", "emotion", "music"):
        if stem.startswith(task):
            return task
    return None


def score_file(
    path: Path, task: str, gt: dict, exclude: set[int]
) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    is_patch = path.stem.endswith("_patch")
    rows = []
    for ans in data.get("answers", []):
        qid = ans["query_id"]
        if qid in exclude:
            continue
        qt_raw = ans.get("query_type", "")
        qt = QT_ALIASES.get(qt_raw, qt_raw)
        choice = ans.get("choice")
        sure = ans.get("sure")
        difficulty = ans.get("difficulty")

        if choice is None:
            continue

        # music: filter by file type and relabel topline audio
        if task == "music":
            if is_patch and qt not in MUSIC_PATCH_TYPES:
                continue
            if not is_patch:
                if qt not in MUSIC_MAIN_TYPES:
                    continue
                if qt == "audio":
                    qt = "audio_top"

        q = gt.get(qid, {})
        correct = is_correct(choice, qt, task, q) if q else None

        rows.append({
            "query_id": qid,
            "query_type": qt,
            "correct": correct,
            "confident": sure == "yes",
            "difficulty": difficulty if difficulty is not None else 0,
            "source": path.stem,
        })
    return rows


def print_table(rows: list[dict], task: str) -> None:
    order = TASK_ORDERS.get(task, sorted({r["query_type"] for r in rows}))
    present = {r["query_type"] for r in rows}
    cols = [qt for qt in order if qt in present]

    by_qt: dict[str, list] = defaultdict(list)
    for r in rows:
        by_qt[r["query_type"]].append(r)

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    col_w = 12
    lbl_w = 14
    header = (f"{'Metric':<{lbl_w}}"
              + "".join(f"{qt:>{col_w}}" for qt in cols)
              + f"{'Avg':>{col_w}}")
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    metrics = [
        ("Accuracy",   lambda r: r["correct"]),
        ("Confidence", lambda r: float(r["confident"])),
        ("Difficulty", lambda r: float(r["difficulty"])),
    ]
    for label, fn in metrics:
        row = f"{label:<{lbl_w}}"
        all_vals = []
        for qt in cols:
            vals = [fn(r) for r in by_qt[qt]]
            v = avg(vals)
            row += f"{v:>{col_w}.3f}"
            all_vals.extend([x for x in vals if x is not None])
        row += f"{avg(all_vals):>{col_w}.3f}"
        print(row)
    print(sep)
    n = sum(len(by_qt[qt]) for qt in cols)
    print(f"  n={n} answers from {len({r['source'] for r in rows})} file(s)")


def main():
    filters = [a.lower() for a in sys.argv[1:]]

    by_task: dict[str, list] = defaultdict(list)
    for path in sorted(OUTCOME_DIR.glob("*.json")):
        task = detect_task(path.name)
        if not task:
            continue
        if filters and task not in filters:
            continue
        gt = load_gt(task)
        exclude = load_exclude(task)
        rows = score_file(path, task, gt, exclude)
        by_task[task].extend(rows)

    if not by_task:
        print("No outcome files found.")
        return

    for task, rows in sorted(by_task.items()):
        print(f"\n{'='*54}")
        print(f"  {task.upper()}  ({len(rows)} answers)")
        print(f"{'='*54}")
        print_table(rows, task)


if __name__ == "__main__":
    main()
