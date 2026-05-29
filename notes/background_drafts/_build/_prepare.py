"""
Prepare _build/ for compiling background.tex standalone.

- Concatenate master refs.bib + all 8 cluster_C*.bib into _build/refs.bib
- Normalise BibTeX keys with non-ASCII / non-portable characters
- Apply matching key renames to a copy of background.tex
- Emit _build/refs.bib and _build/background_compile.tex
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(r"C:\GitHub\Paradigm_Shift_Act_Inf")
BUILD = ROOT / "notes" / "background_drafts" / "_build"
BG = ROOT / "notes" / "background_drafts" / "background.tex"
MASTER = ROOT / "paper" / "refs.bib"
ADD_DIR = ROOT / "paper" / "bib_additions"


def sanitise_key(key: str) -> str:
    # Strip accents, remove apostrophes/quotes, replace remaining
    # non-alnum-or-underscore-or-hyphen with empty
    nfkd = unicodedata.normalize("NFKD", key)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    cleaned = re.sub(r"[^A-Za-z0-9_:-]", "", ascii_only)
    return cleaned


def collect_keys_from_bib(text: str) -> dict[str, str]:
    """Return map of original-key -> sanitised-key for every @entry{key,"""
    out: dict[str, str] = {}
    for m in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", text):
        orig = m.group(1)
        san = sanitise_key(orig)
        if san != orig:
            out[orig] = san
    return out


def rewrite_bib_keys(text: str, mapping: dict[str, str]) -> str:
    def repl(m: re.Match) -> str:
        head = m.group(1)
        key = m.group(2)
        tail = m.group(3)
        new_key = mapping.get(key, key)
        return f"{head}{new_key}{tail}"

    return re.sub(r"(@\w+\s*\{\s*)([^,\s]+)(\s*,)", repl, text)


def rewrite_cite_keys(text: str, mapping: dict[str, str]) -> str:
    """Rewrite \\cite*{a, b, c} keys to sanitised forms when present in mapping."""

    def repl_cmd(m: re.Match) -> str:
        cmd = m.group(1)
        keys_blob = m.group(2)
        keys = [k.strip() for k in keys_blob.split(",")]
        new_keys = [mapping.get(k, k) for k in keys]
        return f"\\{cmd}{{{', '.join(new_keys)}}}"

    return re.sub(r"\\(cite|citet|citep|citealt|citealp)\*?\{([^}]+)\}", repl_cmd, text)


def safe_print(s: str) -> None:
    print(s.encode("ascii", "backslashreplace").decode("ascii"))


def main() -> int:
    bibs = [MASTER] + sorted(ADD_DIR.glob("cluster_C*.bib"))
    print("Concatenating bibs:")
    for b in bibs:
        print(f"  - {b.relative_to(ROOT)}")

    chunks: list[str] = []
    mapping: dict[str, str] = {}
    for b in bibs:
        t = b.read_text(encoding="utf-8", errors="replace")
        m = collect_keys_from_bib(t)
        mapping.update(m)
        chunks.append(f"\n% ===== from {b.name} =====\n")
        chunks.append(t)

    print(f"\nKey renames (non-ASCII / quote / apostrophe -> ASCII): {len(mapping)}")
    for k, v in sorted(mapping.items())[:20]:
        safe_print(f"  {k!r}  ->  {v!r}")
    if len(mapping) > 20:
        print(f"  ... +{len(mapping) - 20} more")

    refs_text = "".join(chunks)
    refs_text = rewrite_bib_keys(refs_text, mapping)
    (BUILD / "refs.bib").write_text(refs_text, encoding="utf-8")

    bg_text = BG.read_text(encoding="utf-8")
    bg_text = rewrite_cite_keys(bg_text, mapping)
    (BUILD / "background_compile.tex").write_text(bg_text, encoding="utf-8")

    # Count distinct keys actually used in background.tex
    used_keys = set()
    for m in re.finditer(r"\\(?:cite|citet|citep|citealt|citealp)\*?\{([^}]+)\}", bg_text):
        for k in m.group(1).split(","):
            used_keys.add(k.strip())
    print(f"\nDistinct cite keys in background.tex (post-rewrite): {len(used_keys)}")

    # Check which used keys exist in the merged refs.bib
    refs_keys = {m.group(1) for m in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", refs_text)}
    missing = sorted(used_keys - refs_keys)
    print(f"  Resolved against refs.bib: {len(used_keys - set(missing))}")
    print(f"  MISSING (will show as ??): {len(missing)}")
    for k in missing:
        print(f"    - {k}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
