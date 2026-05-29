"""
Merge and dedupe per-cluster Semantic Scholar bibs into one additions file
per cluster, filtered against the master refs.bib.

Run from anywhere; paths are absolute.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import bibtexparser
    from bibtexparser.bparser import BibTexParser
except ImportError:
    sys.exit("bibtexparser not installed — pip install bibtexparser")

ROOT = Path(r"C:\GitHub\Paradigm_Shift_Act_Inf")
LIT = Path(r"C:\GitHub\PaperPipeline\lit_reviews")
MASTER_BIB = ROOT / "paper" / "refs.bib"
OUT_DIR = ROOT / "paper" / "bib_additions"

DATE = "2026-05-29"

CLUSTERS = {
    "C1_motivated_reasoning": {
        "slug": "motivated_reasoning",
        "section": "PDF §2 — motivated belief at the single-agent level",
        "anchors": [
            ("kunda", 1990, "case for motivated reasoning"),
            ("kahan", None, "cultural cognition"),
            ("lord", 1979, "biased assimilation"),
            ("nickerson", 1998, "confirmation bias"),
        ],
    },
    "C2_variational_cost": {
        "slug": "variational_cost",
        "section": "PDF §2 + §5 — variational cost of belief change / bounded cognition",
        "anchors": [
            ("sims", 2003, "rational inattention"),
            ("genewein", 2015, "bounded rationality abstraction"),
            ("ortega", 2013, "thermodynamics decision-making"),
            ("friston", 2010, "free-energy principle"),
            ("da costa", 2020, "active inference discrete"),
        ],
    },
    "C3_bmr_lakatos": {
        "slug": "bmr",
        "section": "PDF §5 — Bayesian Model Reduction + Lakatos research programmes",
        "anchors": [
            ("friston", 2011, "post hoc bayesian model"),
            ("friston", None, "bayesian model reduction"),
            ("lakatos", None, "research programmes"),
            ("hartmann", None, "bayesian"),
            ("thagard", None, "explanatory coherence"),
            ("quine", None, "web of belief"),
        ],
    },
    "C4_social_network_epistemology": {
        "slug": "network_epistemology",
        "section": "PDF §3 + §6 — Zollman tradeoff, Fiedler, division of cognitive labor",
        "anchors": [
            ("zollman", 2007, "communication structure"),
            ("zollman", 2010, "transient diversity"),
            ("hegselmann", 2002, "bounded confidence"),
            ("degroot", 1974, "reaching consensus"),
            ("fiedler", 1973, "algebraic connectivity"),
            ("longino", None, "critical contextual empiricism"),
            ("kitcher", None, "division cognitive labor"),
            ("strevens", None, "priority"),
            ("bala", 1998, "learning from neighbours"),
            ("muldoon", None, "agent-based"),
            ("weisberg", None, "epistemic"),
            ("friedkin", None, "social influence"),
            ("johnsen", None, "opinion"),
            ("bloor", None, "strong programme"),
        ],
    },
    "C5_echo_chambers": {
        "slug": "echo_chambers",
        "section": "PDF §6 — bubble vs chamber distinction",
        "anchors": [
            ("nguyen", 2020, "echo chambers"),
            ("sunstein", None, "republic"),
            ("bail", 2018, "exposure opposing views"),
            ("del vicario", None, "echo chambers"),
        ],
    },
    "C6_theory_laden": {
        "slug": "theory_laden",
        "section": "PDF §1 + §4 — Kuhn's three faces, Lavoisier example",
        "anchors": [
            ("hanson", 1958, "patterns of discovery"),
            ("feyerabend", None, "against method"),
            ("chang", None, ""),
            ("bird", None, "incommensurability"),
            ("bogen", None, "saving phenomena"),
        ],
    },
    "C7_attention_precision": {
        "slug": "attention_precision",
        "section": "PDF §5 closing — active sampling under EFE",
        "anchors": [
            ("friston", None, "curiosity insight"),
            ("schwartenbeck", None, ""),
            ("sajid", 2021, "active inference demystified"),
            ("da costa", 2020, "discrete state-spaces"),
            ("smith", None, "step-by-step tutorial"),
        ],
    },
    "C8_algorithmic_curation": {
        "slug": "algorithmic_curation",
        "section": "PDF §7 — algorithmic curation of attention",
        "anchors": [
            ("pariser", None, "filter bubble"),
            ("stray", None, ""),
            ("carroll", None, "induced preference"),
            ("mansoury", None, "feedback loop"),
        ],
    },
}


def normalise(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\{|\}", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def first_author_last(entry: dict) -> str:
    a = entry.get("author", "")
    if not a:
        return ""
    first = a.split(" and ")[0].strip()
    if "," in first:
        last = first.split(",", 1)[0].strip()
    else:
        last = first.split()[-1] if first else ""
    return normalise(last)


def parse_bib(path: Path) -> list[dict]:
    if not path.exists():
        return []
    parser = BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            db = bibtexparser.load(f, parser=parser)
        return db.entries
    except Exception as e:
        print(f"  ! parse error in {path.name}: {e}")
        return []


def entry_key(e: dict) -> tuple[str, str, str]:
    return (
        first_author_last(e),
        str(e.get("year", "")).strip(),
        normalise(e.get("title", ""))[:30],
    )


def matches_anchor(e: dict, anchors: list[tuple]) -> str | None:
    fa = first_author_last(e)
    yr = str(e.get("year", "")).strip()
    title = normalise(e.get("title", ""))
    for a_name, a_year, a_title in anchors:
        if a_name and a_name not in fa:
            continue
        if a_year and yr != str(a_year):
            continue
        if a_title and a_title not in title:
            continue
        return f"{a_name} {a_year or ''} {a_title or ''}".strip()
    return None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading master bib: {MASTER_BIB}")
    master = parse_bib(MASTER_BIB)
    master_keys = {entry_key(e) for e in master}
    master_author_year = {(first_author_last(e), str(e.get("year", "")).strip()) for e in master}
    print(f"  master entries: {len(master)}")

    totals = {"found": 0, "kept": 0, "anchor_hits": 0}

    for cluster_name, cfg in CLUSTERS.items():
        slug = cfg["slug"]
        cluster_dir = LIT / f"{DATE}_{slug}"
        if not cluster_dir.exists():
            print(f"\n[{cluster_name}] no dir; skipping")
            continue

        print(f"\n[{cluster_name}]  {cfg['section']}")
        bibs = sorted(cluster_dir.glob("*.bib"))
        print(f"  input bibs: {[b.name for b in bibs]}")

        all_entries: list[dict] = []
        for b in bibs:
            es = parse_bib(b)
            all_entries.extend(es)
        print(f"  raw entries: {len(all_entries)}")

        seen: set[tuple] = set()
        kept: list[dict] = []
        dropped_master = 0
        dropped_internal = 0
        anchor_hits: list[str] = []

        for e in all_entries:
            k = entry_key(e)
            ay = (k[0], k[1])
            if not k[0]:
                continue
            if k in master_keys or ay in master_author_year:
                dropped_master += 1
                continue
            if k in seen:
                dropped_internal += 1
                continue
            seen.add(k)
            anchor = matches_anchor(e, cfg["anchors"])
            if anchor:
                anchor_hits.append(f"{anchor} -> {e.get('ID', '?')}")
            kept.append(e)

        kept.sort(key=lambda e: int(e.get("year") or 0), reverse=True)

        out_path = OUT_DIR / f"cluster_{cluster_name}.bib"
        memo_lines = [
            f"% Cluster {cluster_name}",
            f"% Section served: {cfg['section']}",
            f"% Generated: {DATE} (lit-review pass 1)",
            f"% Input bibs: {', '.join(b.name for b in bibs)}",
            f"% Raw entries: {len(all_entries)}",
            f"% Dropped (master dup): {dropped_master}",
            f"% Dropped (internal dup): {dropped_internal}",
            f"% Kept: {len(kept)}",
            f"%",
            f"% Anchor recovery checklist:",
        ]
        if anchor_hits:
            for h in anchor_hits:
                memo_lines.append(f"%   [HIT] {h}")
        else:
            memo_lines.append("%   (no anchor matches — review queries)")
        memo_lines.append("%")
        memo_lines.append("% NOTE: keys are Semantic Scholar auto-generated. Rename to hand-curated")
        memo_lines.append("% convention (e.g. firstauthorYYYYShortTitle) before merging into refs.bib.")
        memo_lines.append("")

        db = bibtexparser.bibdatabase.BibDatabase()
        db.entries = kept
        writer = bibtexparser.bwriter.BibTexWriter()
        writer.indent = "  "
        bib_str = writer.write(db)
        out_path.write_text("\n".join(memo_lines) + "\n" + bib_str, encoding="utf-8")
        print(f"  wrote: {out_path}  ({len(kept)} entries, {len(anchor_hits)} anchors)")
        totals["found"] += len(all_entries)
        totals["kept"] += len(kept)
        totals["anchor_hits"] += len(anchor_hits)

    print(
        f"\nTotals — found {totals['found']}, kept {totals['kept']}, anchor hits {totals['anchor_hits']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
