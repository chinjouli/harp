"""Inter-judge consistency: Cohen's κ, Fleiss' κ, 3-way agreement, leniency."""
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

FILE_ORDER = [
    ("scores_text",         "text"),
    ("scores_embed",        "emb"),
    ("scores",              "hybrid"),
    ("scores_audio",        "hybrid_audio"),
    ("scores_all",          "hybrid_all"),
    ("scores_oracle_audio", "oracle_audio"),
    ("scores_oracle_all",   "oracle_all"),
]
_DISPLAY_MAP = dict(FILE_ORDER)
_FILE_ORDER_STEMS = sorted(
    (stem for stem, _ in FILE_ORDER), key=len, reverse=True
)


def _base_stem(file_stem: str) -> str:
    for fo_stem in _FILE_ORDER_STEMS:
        if file_stem == fo_stem or file_stem.startswith(fo_stem + "_"):
            return fo_stem
    return file_stem


def load_exclude(run_dir: Path) -> set[int]:
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


def load_scores(path: Path, exclude: set[int]) -> dict[int, dict]:
    """Load {qid: {answer, rationale}} from a score jsonl."""
    out = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            qid = row.get("query_id")
            if qid is None or int(qid) in exclude:
                continue
            out[int(qid)] = {
                "answer": int(bool(row.get("answer", 0))),
                "rationale": int(bool(row.get("rationale", 0))),
            }
    return out


def discover_groups(run_dir: Path) -> dict[str, dict[str, Path]]:
    """Group score files by base_stem → {judge_label: path}.

    Bare files (no suffix) are labeled 'qwen'; suffixed files use the suffix.
    """
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for f in sorted(run_dir.glob("scores*.jsonl")):
        base = _base_stem(f.stem)
        if base not in _DISPLAY_MAP:
            continue
        if f.stem == base:
            groups[base]["qwen"] = f
        else:
            judge = f.stem[len(base) + 1:]
            groups[base][judge] = f
    return dict(groups)


def cohen_kappa(y1: list[int], y2: list[int]) -> float:
    n = len(y1)
    if n == 0:
        return float("nan")
    p_o = sum(a == b for a, b in zip(y1, y2)) / n
    p1 = sum(y1) / n
    p2 = sum(y2) / n
    p_e = p1 * p2 + (1 - p1) * (1 - p2)
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def fleiss_kappa(ratings: list[tuple]) -> float:
    """Fleiss' κ for binary outcomes with k raters per item."""
    n = len(ratings)
    if n == 0:
        return float("nan")
    k = len(ratings[0])
    if k < 2:
        return float("nan")
    P_i_sum = 0.0
    n_cat1 = 0
    for r in ratings:
        n1 = sum(r)
        n0 = k - n1
        P_i_sum += n1 * (n1 - 1) + n0 * (n0 - 1)
        n_cat1 += n1
    P_bar = P_i_sum / (n * k * (k - 1))
    p1 = n_cat1 / (n * k)
    p0 = 1 - p1
    P_e = p1 ** 2 + p0 ** 2
    if P_e >= 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def agree_all(ratings: list[tuple]) -> float:
    if not ratings:
        return float("nan")
    return sum(len(set(r)) == 1 for r in ratings) / len(ratings)


def compute_stats(
    groups: dict[str, dict[str, Path]],
    exclude: set[int],
) -> list[dict]:
    results = []
    for base_stem, judge_files in groups.items():
        if len(judge_files) < 2:
            continue
        all_scores = {
            judge: load_scores(path, exclude)
            for judge, path in judge_files.items()
        }
        common_qids = sorted(
            set.intersection(*[set(s.keys()) for s in all_scores.values()])
        )
        n = len(common_qids)
        if n == 0:
            continue
        judges = sorted(all_scores.keys())

        leniency = {
            j: {
                m: sum(all_scores[j][qid][m] for qid in common_qids) / n
                for m in ("answer", "rationale")
            }
            for j in judges
        }

        pairwise: dict[tuple, dict] = {}
        for j1, j2 in combinations(judges, 2):
            pair = (j1, j2)
            pairwise[pair] = {}
            for m in ("answer", "rationale"):
                y1 = [all_scores[j1][qid][m] for qid in common_qids]
                y2 = [all_scores[j2][qid][m] for qid in common_qids]
                pairwise[pair][m] = cohen_kappa(y1, y2)

        multi: dict[str, dict] = {"fleiss": {}, "agree3": {}}
        if len(judges) >= 3:
            for m in ("answer", "rationale"):
                ratings = [
                    tuple(all_scores[j][qid][m] for j in judges)
                    for qid in common_qids
                ]
                multi["fleiss"][m] = fleiss_kappa(ratings)
                multi["agree3"][m] = agree_all(ratings)

        results.append({
            "variant": _DISPLAY_MAP.get(base_stem, base_stem),
            "base_stem": base_stem,
            "n": n,
            "judges": judges,
            "leniency": leniency,
            "pairwise": pairwise,
            **multi,
        })
    return results


def _fmt(v: float, pct: bool = False) -> str:
    if v != v:
        return "—"
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"


def print_report(name: str, results: list[dict], exclude: set[int]) -> None:
    if not results:
        return
    all_judges = sorted({j for r in results for j in r["judges"]})
    pairs = list(combinations(all_judges, 2))

    print(f"\n{'='*72}")
    print(f"  {name}")
    if exclude:
        print(f"  (excluding {len(exclude)} queries from exclude.txt)")
    print(f"{'='*72}")

    # --- Leniency ---
    print("\n#### Leniency (mean score)\n")
    j_abbr = [j[:12] for j in all_judges]
    col = 9
    hdr = f"{'Variant':<14} {'N':>5}"
    for ja in j_abbr:
        hdr += f"  {ja+'_ans':>{col}}  {ja+'_rat':>{col}}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        row = f"{r['variant']:<14} {r['n']:>5}"
        for j in all_judges:
            if j in r["leniency"]:
                row += (
                    f"  {r['leniency'][j]['answer']:>{col}.3f}"
                    f"  {r['leniency'][j]['rationale']:>{col}.3f}"
                )
            else:
                row += f"  {'—':>{col}}  {'—':>{col}}"
        print(row)

    # --- Pairwise Cohen's κ ---
    print("\n#### Pairwise Cohen's κ\n")
    pair_labels = [f"{j1[:4]}↔{j2[:4]}" for j1, j2 in pairs]
    for m in ("answer", "rationale"):
        print(f"  {m.capitalize()}")
        hdr2 = f"  {'Variant':<14}"
        for pl in pair_labels:
            hdr2 += f"  {pl:>12}"
        hdr2 += f"  {'avg':>8}"
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 2))
        avg_by_pair: dict[tuple, list] = {p: [] for p in pairs}
        for r in results:
            row = f"  {r['variant']:<14}"
            kappas = []
            for pair in pairs:
                k = r["pairwise"].get(pair, {}).get(m, float("nan"))
                row += f"  {_fmt(k):>12}"
                if k == k:
                    kappas.append(k)
                    avg_by_pair[pair].append(k)
            avg = sum(kappas) / len(kappas) if kappas else float("nan")
            row += f"  {_fmt(avg):>8}"
            print(row)
        # AVG row
        row = f"  {'AVG':<14}"
        avgs = []
        for pair in pairs:
            vs = avg_by_pair[pair]
            v = sum(vs) / len(vs) if vs else float("nan")
            row += f"  {_fmt(v):>12}"
            if v == v:
                avgs.append(v)
        row += f"  {_fmt(sum(avgs)/len(avgs) if avgs else float('nan')):>8}"
        print(row)
        print()

    # --- Fleiss' κ + 3-way ---
    if any(r.get("fleiss") for r in results):
        print("#### Fleiss' κ  &  3-way agreement\n")
        hdr3 = (
            f"{'Variant':<14} {'N':>5}"
            f"  {'ans-κ':>8}  {'ans-3way':>9}"
            f"  {'rat-κ':>8}  {'rat-3way':>9}"
        )
        print(hdr3)
        print("-" * len(hdr3))
        fleiss_ans, fleiss_rat, agree_ans, agree_rat = [], [], [], []
        for r in results:
            fk_a = r.get("fleiss", {}).get("answer", float("nan"))
            fk_r = r.get("fleiss", {}).get("rationale", float("nan"))
            a3_a = r.get("agree3", {}).get("answer", float("nan"))
            a3_r = r.get("agree3", {}).get("rationale", float("nan"))
            print(
                f"{r['variant']:<14} {r['n']:>5}"
                f"  {_fmt(fk_a):>8}  {_fmt(a3_a, pct=True):>9}"
                f"  {_fmt(fk_r):>8}  {_fmt(a3_r, pct=True):>9}"
            )
            for lst, v in [
                (fleiss_ans, fk_a), (fleiss_rat, fk_r),
                (agree_ans, a3_a), (agree_rat, a3_r),
            ]:
                if v == v:
                    lst.append(v)
        # AVG row
        def _avg(lst: list) -> float:
            return sum(lst) / len(lst) if lst else float("nan")
        print(
            f"{'AVG':<14} {'':<5}"
            f"  {_fmt(_avg(fleiss_ans)):>8}  {_fmt(_avg(agree_ans), pct=True):>9}"
            f"  {_fmt(_avg(fleiss_rat)):>8}  {_fmt(_avg(agree_rat), pct=True):>9}"
        )


def main():
    data_dir = Path("data")
    if not data_dir.exists():
        print("data/ not found", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) > 1:
        run_dirs = [Path(a) for a in sys.argv[1:]]
    else:
        run_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())

    stem_rank = {stem: i for i, (stem, _) in enumerate(FILE_ORDER)}

    out_path = data_dir / "judgeeval.md"
    with out_path.open("w", encoding="utf-8") as out_f:
        import contextlib, io
        for run_dir in run_dirs:
            if not any(run_dir.glob("scores*.jsonl")):
                continue
            exclude = load_exclude(run_dir)
            groups = discover_groups(run_dir)
            results = compute_stats(groups, exclude)
            results.sort(key=lambda r: stem_rank.get(r["base_stem"], 99))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_report(run_dir.name, results, exclude)
            text = buf.getvalue()
            sys.stdout.write(text)
            out_f.write(text)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
