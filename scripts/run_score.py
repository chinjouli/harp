"""Aggregate scores_*.jsonl files and report metrics by query type."""
import json
import sys
from collections import defaultdict
from pathlib import Path

EMO_ORDER = ["state", "change", "comparison", "locate", "locate_hard"]
EMO_LABELS = ["State", "Change", "Comp.", "Locate", "Locate_hard"]
MUSIC_ORDER = ["time", "position", "audio"]
MUSIC_LABELS = ["Time", "Position", "Audio"]
HEALTH_ORDER = ["reported", "exhibited"]
HEALTH_LABELS = ["Reported", "Exhibited"]

METRICS = ["plan_hit_rate", "retrieval_hit_rate", "answer", "rationale"]
METRIC_LABELS = ["Plan HR", "Retr HR", "Ans Acc", "Rat Acc"]

# display order: (filename_stem, display_name)
FILE_ORDER = [
    ("scores_text",        "text"),
    ("scores_embed",       "emb"),
    ("scores",             "hybrid"),
    ("scores_audio","hybrid_audio"),
    ("scores_all",  "hybrid_all"),
    ("scores_oracle_audio","oracle_audio"),
    ("scores_oracle_all",  "oracle_all"),
]


def load_query_types(queries_path: Path) -> dict[int, str]:
    qt = {}
    with open(queries_path) as f:
        for line in f:
            q = json.loads(line)
            qt[q["query_id"]] = q["query_type"]
    return qt


_AUDIO_STEMS = {"scores_audio", "scores_oracle_audio"}

# stems from FILE_ORDER, longest first so prefix matching is unambiguous
_FILE_ORDER_STEMS = sorted(
    (stem for stem, _ in FILE_ORDER), key=len, reverse=True
)


def _base_stem(file_stem: str) -> str:
    """Return the FILE_ORDER stem that file_stem is a variant of."""
    for fo_stem in _FILE_ORDER_STEMS:
        if file_stem == fo_stem or file_stem.startswith(fo_stem + "_"):
            return fo_stem
    return file_stem


def _extra_scores(
    run_dir: Path, covered: set[str]
) -> list[tuple[Path, str]]:
    """Discover judge-variant score files not in FILE_ORDER."""
    stem_to_display = dict(FILE_ORDER)
    extras = []
    for f in sorted(run_dir.glob("scores_*.jsonl")):
        if f.stem in covered:
            continue
        base = _base_stem(f.stem)
        base_display = stem_to_display.get(base, base.replace("scores_", ""))
        if base != f.stem:
            judge = f.stem[len(base) + 1:]
            display = f"{base_display} ({judge})"
        else:
            display = f.stem.replace("scores_", "")
        extras.append((f, display))
    return extras


def score_row(row: dict, skip_faithful: bool = False) -> dict:
    if skip_faithful:
        rationale = int(bool(row.get("factual")))
    else:
        rationale = int(bool(row.get("factual")) and bool(row.get("faithful")))
    return {
        "plan_hit_rate": row.get("plan_hit_rate", float("nan")),
        "retrieval_hit_rate": row.get("retrieval_hit_rate", float("nan")),
        "answer": row.get("answer", 0),
        "rationale": rationale,
    }


def load_exclude(run_dir: Path) -> set[int]:
    """Load excluded query IDs from exclude.txt (lines starting with # ignored)."""
    p = run_dir / "exclude.txt"
    if not p.exists():
        return set()
    excluded = set()
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                excluded.add(int(line.split()[0]))
            except (ValueError, IndexError):
                pass
    return excluded


def aggregate(
    scores_path: Path,
    query_types: dict[int, str],
    exclude: set[int] | None = None,
) -> dict[str, dict]:
    skip_faithful = _base_stem(scores_path.stem) in _AUDIO_STEMS
    buckets: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with open(scores_path) as f:
        for line in f:
            row = json.loads(line)
            qid = row["query_id"]
            if exclude and qid in exclude:
                continue
            qtype = query_types.get(qid)
            if qtype is None:
                continue
            for k, v in score_row(row, skip_faithful=skip_faithful).items():
                buckets[qtype][k].append(v)
    return {
        qt: {k: sum(vs) / len(vs) for k, vs in metrics.items()}
        for qt, metrics in buckets.items()
    }


def detect_task(query_types: dict[int, str]) -> str:
    types = set(query_types.values())
    if types & {"state", "change", "comparison", "locate", "locate_hard"}:
        return "emotion"
    if types & {"reported", "exhibited"}:
        return "health"
    return "music"


def print_table(display_name: str, agg: dict, order, labels):
    lbl_w = 10
    col_w = max((len(lbl) for lbl in labels), default=5) + 2
    header = (f"{'Metric':<{lbl_w}}"
              + "".join(f"{lbl:>{col_w}}" for lbl in labels)
              + f"{'Avg':>{col_w}}")
    sep = "-" * len(header)
    print(f"\n  {display_name}")
    print(sep)
    print(header)
    print(sep)
    for m, ml in zip(METRICS, METRIC_LABELS):
        row = f"{ml:<{lbl_w}}"
        vals = []
        for qt in order:
            if qt not in agg:
                row += f"{'—':>{col_w}}"
            else:
                v = agg[qt][m]
                row += f"{v:>{col_w}.3f}"
                vals.append(v)
        avg = (f"{sum(vals)/len(vals):>{col_w}.3f}"
               if vals else f"{'—':>{col_w}}")
        print(row + avg)
    print(sep)


_VARIANT_ORDER = [
    "text", "emb", "hybrid", "hybrid_audio", "hybrid_all",
    "oracle_audio", "oracle_all",
]


def _tsv_sort_key(display_name: str) -> tuple:
    """Sort key: (judge_rank, variant_rank). Qwen first, then alphabetical."""
    if " (" in display_name:
        variant, judge = display_name.split(" (", 1)
        judge = judge.rstrip(")")
    else:
        variant, judge = display_name, ""
    judge_rank = (0, "") if judge == "" else (1, judge)
    var_rank = _VARIANT_ORDER.index(variant) if variant in _VARIANT_ORDER else 99
    return (judge_rank, var_rank)


def write_tsv(tsv_path: Path, rows: list[tuple[str, dict, list]]) -> None:
    """Write ans+rat interleaved by query type, sorted by judge then variant."""
    if not rows:
        return
    order = rows[0][2]
    header_cols = []
    for qt in order:
        header_cols += [f"{qt}_ans", f"{qt}_rat"]
    header_cols += ["avg_ans", "avg_rat"]
    sorted_rows = sorted(rows, key=lambda r: _tsv_sort_key(r[0]))
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("variant\t" + "\t".join(header_cols) + "\n")
        for display_name, agg, _ in sorted_rows:
            ans_vals, rat_vals = [], []
            cols = []
            for qt in order:
                if qt in agg:
                    a, r = agg[qt]["answer"], agg[qt]["rationale"]
                    cols += [f"{a:.3f}", f"{r:.3f}"]
                    ans_vals.append(a)
                    rat_vals.append(r)
                else:
                    cols += ["", ""]
            avg_a = f"{sum(ans_vals)/len(ans_vals):.3f}" if ans_vals else ""
            avg_r = f"{sum(rat_vals)/len(rat_vals):.3f}" if rat_vals else ""
            cols += [avg_a, avg_r]
            f.write(display_name + "\t" + "\t".join(cols) + "\n")


def process_run(run_dir: Path):
    queries_path = run_dir / "queries.jsonl"
    if not queries_path.exists():
        return
    query_types = load_query_types(queries_path)
    exclude = load_exclude(run_dir)
    task = detect_task(query_types)
    if task == "emotion":
        order, labels = EMO_ORDER, EMO_LABELS
    elif task == "health":
        order, labels = HEALTH_ORDER, HEALTH_LABELS
    else:
        order, labels = MUSIC_ORDER, MUSIC_LABELS

    print(f"\n{'='*54}")
    print(f"  {run_dir.name}  [{task}]")
    if exclude:
        print(f"  (excluding {len(exclude)} queries from exclude.txt)")
    print(f"{'='*54}")

    found_any = False
    covered: set[str] = set()
    tsv_rows: list[tuple[str, dict, list]] = []

    for stem, display_name in FILE_ORDER:
        scores_path = run_dir / f"{stem}.jsonl"
        if not scores_path.exists():
            continue
        found_any = True
        covered.add(stem)
        agg = aggregate(scores_path, query_types, exclude=exclude)
        print_table(display_name, agg, order, labels)
        tsv_rows.append((display_name, agg, order))

    for scores_path, display_name in _extra_scores(run_dir, covered):
        found_any = True
        agg = aggregate(scores_path, query_types, exclude=exclude)
        print_table(display_name, agg, order, labels)
        tsv_rows.append((display_name, agg, order))

    if not found_any:
        print("  (no scores files found)")
    else:
        tsv_path = run_dir / "scores.tsv"
        write_tsv(tsv_path, tsv_rows)
        print(f"\n  TSV written to {tsv_path}")


def main():
    data_dir = Path("data")
    if not data_dir.exists():
        print("data/ directory not found", file=sys.stderr)
        sys.exit(1)

    # if specific dirs given on CLI, use those; else all under data/
    if len(sys.argv) > 1:
        run_dirs = [Path(a) for a in sys.argv[1:]]
    else:
        run_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())

    for run_dir in run_dirs:
        if any(run_dir.glob("scores*.jsonl")):
            process_run(run_dir)


if __name__ == "__main__":
    main()
