import difflib
import re
from typing import Any, Dict, List, Tuple

SNAP_THRESHOLD = 0.80

_GLYPHS = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u00a0": " ", "\u00b5": "\u03bc",
}


def normalize(text: str) -> str:
    """Whitespace / typographic-glyph normalisation only. Wording is untouched."""
    text = text or ""
    for src, dst in _GLYPHS.items():
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", normalize(text)) if s]


def _tokens(text: str) -> List[str]:
    return re.findall(r"[\w.%<>=\u03bc/°-]+", text.lower())


def _coverage(quote: str, sentence: str) -> float:
    """Share of the quote's words that appear in the sentence, in order (word-level)."""
    q, s = _tokens(quote), _tokens(sentence)
    if not q or not s:
        return 0.0
    sm = difflib.SequenceMatcher(None, q, s, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(q)


def verify_quote(quote: str, source_text: str) -> Tuple[str, str, float]:
    """Return (status, resolved_quote, similarity). Status is exact | snapped | unverified.

    "exact" means the quote occurs in the source up to whitespace, quote/dash glyphs and letter case.
    """
    nq = normalize(quote).strip(" \"'")
    ns = normalize(source_text)
    if nq and nq.casefold() in ns.casefold():
        return "exact", quote.strip(), 1.0

    best, best_sim = "", 0.0
    for sent in split_sentences(source_text):
        sim = _coverage(nq, sent)
        if sim > best_sim:
            best, best_sim = sent, sim
    if best and best_sim >= SNAP_THRESHOLD:
        return "snapped", best, round(best_sim, 3)
    return "unverified", quote, round(best_sim, 3)


def verify_edges(edges: List[Dict[str, Any]], source_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Annotate every edge with provenance_status. 'snapped' edges get the real source sentence."""
    out, counts = [], {"exact": 0, "snapped": 0, "unverified": 0}
    for edge in edges:
        e = dict(edge)
        status, resolved, sim = verify_quote(e.get("provenance_quote", ""), source_text)
        e["provenance_quote"] = resolved
        e["provenance_status"] = status
        e["provenance_similarity"] = sim
        counts[status] += 1
        out.append(e)
    total = len(out)
    stats = {
        "total": total,
        "exact": counts["exact"],
        "snapped": counts["snapped"],
        "unverified": counts["unverified"],
        "faithfulness": round(counts["exact"] / total, 4) if total else 0.0,
    }
    return out, stats
