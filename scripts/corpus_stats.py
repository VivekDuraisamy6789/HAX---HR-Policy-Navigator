"""Honest corpus-size report for the '500-2,000 documents' requirement.

    python scripts/corpus_stats.py [--exclude trivy]
Reports: markdown files, section-level units (H1/H2 sections: the unit a user would call a
"document" in a handbook), and chunk counts per chunking strategy. State in the report which unit
you count and why; do not present chunks as files.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hr_policy_navigator.chunking import _sections, chunk_document
from hr_policy_navigator.config import Settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default="", help="comma-separated path substrings to drop (e.g. trivy: release-note noise)")
    a = ap.parse_args()
    root = Settings().data_root
    ex = [x for x in a.exclude.split(",") if x]
    files = [f for f in sorted(root.rglob("*.md")) if f.read_text(encoding="utf-8", errors="ignore").strip()
             and not any(e in f.as_posix() for e in ex)]
    words = secs = 0
    per = {"fixed_200": 0, "fixed_500": 0, "heading": 0, "heading_cap": 0}
    for f in files:
        raw = f.read_text(encoding="utf-8", errors="ignore")
        rel = f.relative_to(root).as_posix()
        words += len(raw.split())
        secs += sum(1 for crumb, _ in _sections(raw) if crumb.count(">") <= 1)   # H1/H2-level units
        for k in per:
            per[k] += len(chunk_document(rel, raw, k))
    print(f"corpus root      : {root}")
    print(f"markdown files   : {len(files)}" + (f"  (excluding: {ex})" if ex else ""))
    print(f"words            : {words:,}")
    print(f"H1/H2 sections   : {secs}")
    for k, v in per.items():
        print(f"chunks [{k:11s}]: {v}")
    lo, hi = 500, 2000
    print("\nrequirement 500-2,000 documents:")
    for label, n in (("files", len(files)), ("H1/H2 sections", secs), ("heading_cap chunks", per["heading_cap"])):
        print(f"  {label:20s} {n:6d}  {'within range' if lo <= n <= hi else ('BELOW' if n < lo else 'ABOVE')}")


if __name__ == "__main__":
    main()
