"""Shared evidence formatting for inference prompts and judge prompts."""
from __future__ import annotations

from typing import Any, Dict, List


def _r(v: Any) -> Any:
    return round(v, 2) if isinstance(v, float) else v


def _fmt_vad(v: Any) -> str | None:
    """Format a valence/arousal/dominance value: float or {mean, std} dict."""
    if v is None:
        return None
    if isinstance(v, dict):
        return f"{_r(v['mean'])}±{_r(v['std'])}"
    return str(_r(v))


def format_evidence_sets(
    evidence_sets: List[List[Dict[str, Any]]],
    strip_domain: bool = False,
    strip_speaker: bool = False,
) -> str:
    """Format retrieved evidence sets as headed bullet lists."""
    if not evidence_sets:
        return "(no retrieved evidence)"
    blocks: List[str] = []
    for i, ev in enumerate(evidence_sets):
        seg_blocks: List[str] = []
        for j, s in enumerate(ev):
            t = s.get("time", [None, None])
            conf = s.get("_sources") or {}
            conf_str = (
                "score: " + ", ".join(f"{k}={_r(v)}" for k, v in conf.items())
            ) if conf else ""
            spk = "" if (strip_domain or strip_speaker) else s.get("speaker", "")
            lines = [
                f"[{t[0]:.1f}s–{t[1]:.1f}s]"
                + (f" ({conf_str})" if conf_str else "")
            ]
            if spk:
                lines.append(f"- speaker: {spk}")
            txt = "" if strip_domain else s.get("text", "")
            if txt:
                lines.append(f"- text: \"{txt}\"")
            dl = {} if strip_domain else (s.get("domain_labels") or {})
            primary = dl.get("primary")
            scores = dl.get("scores") or {}
            mean = dl.get("mean")
            vad = {
                k: _fmt_vad(dl.get(k))
                for k in ("valence", "arousal", "dominance")
            }
            if primary or scores or mean is not None or any(vad.values()):
                sc_str = ", ".join(
                    f"{k}={_r(v)}" for k, v in scores.items()
                    if v is not None
                )
                parts = []
                if primary:
                    parts.append(f"primary={primary}")
                if mean is not None:
                    parts.append(f"mean={_r(mean)}")
                if sc_str:
                    parts.append(f"[{sc_str}]")
                vad_str = " ".join(
                    f"{k[0].upper()}={v}"
                    for k, v in vad.items() if v is not None
                )
                if vad_str:
                    parts.append(vad_str)
                lines.append(f"- domain: {', '.join(parts)}")
            seg_blocks.append("\n".join(lines))
        blocks.append(f"### Set {i + 1}\n\n" + "\n\n".join(seg_blocks))
    return "\n\n".join(blocks)
