"""Classify every corpus file as core_hr / hr_adjacent / not_hr using ordered path rules (first match wins).
Rules are deliberately explicit so the team can argue with them; output: docs/corpus_classification.csv

    python scripts/classify_corpus.py
core_hr      employee-facing HR policy: benefits, leave, pay/equity, conduct & DEI, performance, onboarding/offboarding, People Ops
hr_adjacent  touches employees but is not HR policy: recruiting/talent acquisition, travel & expenses, remote-work norms, culture, IT acceptable use
not_hr       security/infosec, finance ops, IT tooling, product, writing/style guides, release notes
"""
import csv, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "sourcegraph_files"
RULES = [  # (regex on path relative to sourcegraph_files, category) - ordered, first match wins
    (r"^synthetic/", "synthetic_hr"),
    (r"^benefits-pay-perks/benefits-perks/(travel|spending-company-money)", "hr_adjacent"),
    (r"^benefits-pay-perks/pay-expenses/(expenses|invoices)", "hr_adjacent"),
    (r"^benefits-pay-perks/", "core_hr"),
    (r"^company-info-and-process/(respectful-workplace-policy|equality-of-opportunity|gender-diversity|personal-pronouns|queer)", "core_hr"),
    (r"^company-info-and-process/communication/(code_of_conduct|conflicts|convercent)", "core_hr"),
    (r"^company-info-and-process/community/code_of_conduct", "core_hr"),
    (r"^company-info-and-process/(working-at-sourcegraph|mentorship)/", "core_hr"),
    (r"^company-info-and-process/communication/(asynchronous-communication|1-1|seeking-and-giving-feedback|announcements|company_meeting|retrospectives|working-group)", "hr_adjacent"),
    (r"^company-info-and-process/(remote|values|about-sourcegraph)/", "hr_adjacent"),
    (r"^company-info-and-process/policies/acceptable-use-policy", "hr_adjacent"),
    (r"^company-info-and-process/(policies|communication|onboarding|community)/", "not_hr"),
    (r"^company-info-and-process/index", "hr_adjacent"),
    (r"^departments/people-talent/talent/", "hr_adjacent"),
    (r"^departments/people-talent/(resources_for_candidates|swag|git-intro)", "hr_adjacent"),
    (r"^departments/people-talent/people-ops/tools/internal-swag", "hr_adjacent"),
    (r"^departments/people-talent/", "core_hr"),
    (r"^departments/tech-ops/process/team_device_usage_privacy", "hr_adjacent"),
]
def classify(rel):
    for rx, cat in RULES:
        if re.search(rx, rel):
            return cat
    return "not_hr"

files = sorted(p for p in CORPUS.rglob("*.md"))
rows = [(p.relative_to(CORPUS).as_posix(), p.stat().st_size) for p in files]
cat = {r: classify(r) for r, _ in rows}
Path(ROOT / "docs").mkdir(exist_ok=True)
with open(ROOT / "docs/corpus_classification.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["path", "bytes", "category"]); [w.writerow([r, s, cat[r]]) for r, s in rows]
cnt = Counter(cat.values()); tot = len(rows)
print(f"total files: {tot}")
for c in ("core_hr", "hr_adjacent", "synthetic_hr", "not_hr"):
    print(f"  {c:12s} {cnt[c]:4d}  ({cnt[c]/tot:.0%})")
by_top = defaultdict(Counter)
for r, _ in rows:
    by_top["/".join(r.split("/")[:2]) if r.count("/") else r][cat[r]] += 1
print("\nby folder (core/adjacent/not):")
for k, v in sorted(by_top.items()):
    print(f"  {k:55s} {v['core_hr']:3d} /{v['hr_adjacent']:3d} /{v['not_hr']:3d}")
trivy = sum(1 for r, _ in rows if "/trivy/" in r)
print(f"\nof the not_hr files, {trivy} are Trivy release notes")
ev = set(s for f in ("evaluation_set.json",) for it in json.load(open(ROOT / f)) for s in it["expected_sources"])
print("\neval-set expected sources by category:", dict(Counter(cat.get(s, "MISSING") for s in ev)), f"({len(ev)} distinct)")
