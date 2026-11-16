"""Verify synthetic docs agree with the real corpus and with each other."""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from generate_synthetic_corpus import REAL_FACTS as R, SYNTH_FACTS as S

root = Path(sys.argv[1] if len(sys.argv) > 1 else "sourcegraph_files")
syn = list((root/"synthetic").rglob("*.md"))
real_text = " ".join(p.read_text(encoding="utf-8", errors="ignore") for p in root.rglob("*.md")
                     if "synthetic" not in p.parts)
errs = []
# 1. every REAL fact is present in the real corpus
checks = {"parental weeks": rf"{R['parental_weeks']} weeks", "receipt": rf"over \${R['receipt_threshold_usd']}\b",
  "wellness": rf"\${R['wellness_stipend_usd']} monthly wellness", "education": rf"\${R['education_usd']} USD/year",
  "software": rf"\${R['software_approval_usd']} per year", "premium": rf"{R['premium_pct']}% of premiums",
  "reimb": rf"{R['reimb_days'].replace('-','-')} day", "case mgr": rf"{R['leave_case_mgr_days']} business days"}
for k, rx in checks.items():
    if not re.search(rx, real_text): errs.append(f"REAL fact not found in real corpus: {k} /{rx}/")
# 2. synthetic docs: numbers tied to a keyword must match the fact table
rules = [(r"(\d+)% of premiums", R["premium_pct"]), (r"\$(\d+) monthly wellness stipend|\$(\d+) per month", R["wellness_stipend_usd"]),
  (r"\$(\d+) per year for education|\$(\d+) per year professional|\$(\d+) per year\*?\*? education|\$(\d+) per year budget|\$(\d+) per year for conferences", R["education_usd"]),
  (r"up to (\d+) weeks", R["parental_weeks"]), (r"over \$(\d+)", R["receipt_threshold_usd"]),
  (r"more than \$(\d+) per year", R["software_approval_usd"]), (r"within (\d+) months", R["parental_window_months"]),
  (r"Buddies are paired for (\d+) weeks|assigned for (\d+) weeks", S["buddy_weeks"]),
  (r"within (\d+) hours", None), (r"(\d+) business days to respond", S["offer_response_days"])]
titles = {}
for p in syn:
    t = p.read_text(encoding="utf-8")
    if "synthetic: true" not in t: errs.append(f"{p}: missing synthetic flag")
    titles.setdefault(re.search(r"^title: (.*)$", t, re.M).group(1), []).append(p)
    for rx, want in rules:
        for m in re.finditer(rx, t):
            v = next(int(g) for g in m.groups() if g)
            if want is not None and v != want: errs.append(f"{p.name}: '{m.group(0)}' != {want}")
            if want is None and v not in (S["manager_response_hours"], S["scorecard_hours"]): errs.append(f"{p.name}: hours {v}")
    # 3. links resolve
    for _, link in re.findall(r"- \[(.*?)\]\((.*?)\)", t):
        if not (p.parent/link).exists(): errs.append(f"{p.name}: broken link {link}")
    if len(re.findall(r"\$(\d+) ", t)) and re.search(r"\$(?!25\b|500\b|100\b|1000\b)\d+", t):
        errs.append(f"{p.name}: unexpected dollar figure")
errs += [f"duplicate title {t}" for t, ps in titles.items() if len(ps) > 1]
bodies = {}
for p in syn:
    b = re.sub(r"^---.*?---", "", p.read_text(encoding="utf-8"), flags=re.S)
    b = re.sub(r"^# .*$", "", b, flags=re.M)
    bodies.setdefault(b, []).append(p.name)
errs += [f"identical bodies: {v[:3]}" for v in bodies.values() if len(v) > 1]
print(f"synthetic docs: {len(syn)}; unique titles: {len(titles)}; unique bodies: {len(bodies)}")
print("OK" if not errs else "\n".join(errs[:40]) + f"\n{len(errs)} problems")
sys.exit(1 if errs else 0)
