"""Generate ~1,000 SYNTHETIC HR / HR-adjacent policy documents.

All numeric facts come from the single FACTS table below. REAL facts were
verified against the real corpus (see REAL_FACTS and verify_synthetic_consistency.py);
SYNTH facts are invented internal-process values, defined once so every document
agrees. Every file is labelled `synthetic: true` and lives under
sourcegraph_files/synthetic/. They are NOT real Sourcegraph policy.
"""
import argparse, hashlib, os, shutil
from pathlib import Path

REAL_FACTS = {  # each must be found in the real corpus (checked by the verifier)
    "parental_weeks": 16, "parental_window_months": 12,
    "receipt_threshold_usd": 25, "reimb_days": "7-10",
    "wellness_stipend_usd": 100, "education_usd": 1000,
    "software_approval_usd": 500, "premium_pct": 100,
    "leave_case_mgr_days": "1-2",
}
SYNTH_FACTS = {  # invented, process-level; used identically everywhere
    "checkin_days": [30, 60, 90], "buddy_weeks": 8, "offer_response_days": 5,
    "equipment_days": 3, "exit_survey_days": 7, "access_removal": "last working day",
    "manager_response_hours": 48, "scorecard_hours": 24,
}
R, S = REAL_FACTS, SYNTH_FACTS

REGIONS = ["United States","Canada","United Kingdom","Ireland","Germany","France","Netherlands","Spain",
 "Italy","Poland","Sweden","Norway","Denmark","Finland","Switzerland","Portugal","Belgium","Austria",
 "Czechia","Romania","India","Singapore","Australia","New Zealand","Japan","South Korea","Israel",
 "United Arab Emirates","South Africa","Nigeria","Kenya","Brazil","Mexico","Argentina","Chile",
 "Colombia","Philippines","Indonesia","Vietnam","Turkey"]
FUNCTIONS = ["Engineering","Product","Design","Customer Engineering","Sales","Sales Development","Marketing",
 "Finance","People","Legal","Security","IT","Support","Developer Relations","Data","Recruiting",
 "Revenue Operations","Customer Success","Technical Writing","Business Operations","Partnerships",
 "Procurement","Facilities","Executive Support","Analytics"]
EXPENSE_CATS = ["Home office equipment","Software subscriptions","Conference registration","Online courses",
 "Books and publications","Team meals","Client meals","Local travel","Air travel","Lodging","Ground transport",
 "Coworking space","Internet and phone","Wellness","Professional memberships","Certifications",
 "Visa and immigration fees","Relocation","Shipping and courier","Office supplies","Meetups hosted by teammates",
 "Training workshops","Gifts for customers","Recruiting events","Offsite activities","Parking and tolls",
 "Mobile device accessories","Ergonomic items","Language courses","Mentoring programs","Video equipment",
 "Printing","Postage","Team celebrations","Interview travel for candidates","Speaker fees","Sponsorships",
 "Translation services","Background checks","Software trials","Hardware repairs","Charging and cables",
 "Headsets","Keyboards and mice","Monitors","Desks and chairs","Lighting","Cloud credits","Security keys",
 "Passport photos","Notary services","Industry reports","Podcast gear","Community events","Hackathon supplies",
 "Remote team games","Welcome kits","Anniversary recognition","Volunteer activities","Charity matching"]
LEAVE_SCEN = [("birth","a birth"),("adoption","an adoption"),("foster","a foster placement")]
SITUATIONS = ["a teammate asks for leave","a teammate is struggling with workload","a teammate wants outside work",
 "a new hire misses a checkpoint","a teammate asks about equity","a teammate relocates countries",
 "a teammate needs an accommodation","a teammate raises a concern","a teammate wants a course funded",
 "a teammate is leaving","a teammate returns from leave","a teammate wants to change teams",
 "a candidate asks about benefits","a teammate reports a payroll error","a teammate needs equipment",
 "a teammate asks about wellness support","a teammate is promoted","a teammate asks for a reference",
 "a teammate has a family emergency","a contractor is converting to employee"]
SKILLS = ["Go","TypeScript","Rust","Python","SQL","Kubernetes","Public speaking","Technical writing",
 "Negotiation","Project management","Data analysis","Security fundamentals","Accessibility","Leadership",
 "Coaching","Product discovery","Customer interviewing","Financial modeling","Excel","Design systems",
 "Observability","Incident response","Cloud architecture","Machine learning basics","Sales methodology",
 "Presentation design","Time management","Mentoring","Interviewing","Remote collaboration"]
STAGES = ["Application review","Recruiter screen","Hiring manager screen","Technical assessment",
 "Take-home exercise","Onsite interview loop","Reference check","Background check","Offer approval",
 "Offer extended","Offer accepted","Pre-boarding","Rejection and feedback","Candidate withdrawal"]
ROLES = ["engineering","sales","support","marketing","finance","product","design","security"]
WELL = ["Stress","Burnout","Sleep","Movement","Nutrition","Financial wellbeing","Caregiving","Grief",
 "Return to work","Time zones","Meeting load","Focus time","Ergonomics","Eye strain","Loneliness",
 "Parent support","Neurodiversity","Chronic conditions","Mental health days","Time off planning"]

def h(*a): return int(hashlib.md5("|".join(map(str,a)).encode()).hexdigest(),16)
def pick(opts,*k): return opts[h(*k)%len(opts)]
def slug(s): return "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-").replace("--","-")

def doc(title, family, tags, body, related):
    rel = "\n".join(f"- [{t}]({p})" for t,p in related)
    return (f"---\ntitle: {title}\nsynthetic: true\nfamily: {family}\ntags: [{', '.join(tags)}]\n---\n\n"
            f"# {title}\n\n> Synthetic training document. Not official Sourcegraph policy. "
            f"Figures marked as policy facts match the real handbook.\n\n{body.strip()}\n\n## Related\n{rel}\n")

REL = {"pto":("Time off","benefits-pay-perks/benefits-perks/time-off/index.md"),
 "parental":("Parental leave","benefits-pay-perks/benefits-perks/parental-leave.md"),
 "loa":("Leave of absence","benefits-pay-perks/benefits-perks/leave-of-absence.md"),
 "benefits":("Benefits and perks","benefits-pay-perks/benefits-perks/benefits.md"),
 "exp":("Expenses","benefits-pay-perks/pay-expenses/expenses/index.md"),
 "spend":("Spending company money","benefits-pay-perks/benefits-perks/spending-company-money.md"),
 "mh":("Mental health","benefits-pay-perks/benefits-perks/mental-health/index.md"),
 "hol":("Holidays","company-info-and-process/working-at-sourcegraph/holidays.md"),
 "ld":("Learning and development","departments/people-talent/learning-and-development.md"),
 "oe":("Outside employment","departments/people-talent/outside-employment.md"),
 "leave":("Leaving Sourcegraph","departments/people-talent/people-ops/process/leaving.md"),
 "peo":("PEO","departments/people-talent/people-ops/tools/peo.md")}
def rel(*k): return [REL[x] for x in k]

def region_docs():
    for reg in REGIONS:
        us = reg=="United States"
        prem = (f"For US-based full-time employees, Sourcegraph covers {R['premium_pct']}% of premiums for medical, dental, and vision insurance for employees and dependents."
                if us else f"Medical coverage in {reg} follows the country supplement maintained by the People Team; the {R['premium_pct']}% premium rule is stated for US-based full-time employees only.")
        yield (f"{reg} leave guide","region",[reg.lower(),"leave"],
         f"""## Paid parental leave in {reg}
Eligible teammates in {reg} receive 100% of salary for up to {R['parental_weeks']} weeks per birth, and up to {R['parental_weeks']} weeks for one or multiple adoption or placement events per {R['parental_window_months']} months. Where a country or state benefit exists, Sourcegraph covers the difference.

## How to start
Complete the leave questionnaire. Your Leave Case Manager will reach out in {R['leave_case_mgr_days']} business days with a leave plan. You may take the weeks consecutively or non-consecutively within {R['parental_window_months']} months of your first leave date.""", rel("parental","loa"))
        yield (f"{reg} benefits summary","region",[reg.lower(),"benefits"],
         f"""## Benefits in {reg}
{prem}

All teammates, wherever they are located, can use the **${R['wellness_stipend_usd']} monthly wellness stipend** (paid in local currency equivalent) and the **${R['education_usd']} per year** education and professional development budget.

Questions specific to {reg} go to the People Team, who confirm local entitlements before any commitment is made to a teammate.""", rel("benefits","mh"))
        yield (f"{reg} expense reimbursement","region",[reg.lower(),"expenses"],
         f"""## Submitting expenses from {reg}
Expenses are submitted through Airbase, reached via Okta. Itemized receipts are required for any reimbursement request over ${R['receipt_threshold_usd']}. Expect reimbursement in {R['reimb_days']} days. Reimbursements are paid in the currency of your payroll location.

Software or subscriptions that would cost more than ${R['software_approval_usd']} per year need approval before purchase.""", rel("exp","spend"))
        yield (f"{reg} holidays and time off","region",[reg.lower(),"time-off"],
         f"""## Time off in {reg}
Teammates in {reg} follow the company time-off policy and the published holiday calendar. Request time off in advance, tell your manager, and mark it on your team calendar. Local public holidays that differ from the company calendar are confirmed with the People Team.

Leave longer than two weeks is handled as a leave of absence; your manager and the Leave Case Manager coordinate coverage.""", rel("pto","hol","loa"))
        yield (f"Working remotely from {reg}","region",[reg.lower(),"remote"],
         f"""## Remote work in {reg}
Working from {reg} as a regular location requires that payroll, tax and employer-of-record arrangements exist for that country. Do not begin working from a new country for an extended period until the People Team confirms. Short trips are fine for most teammates but still require manager awareness.

Equipment is shipped within {S['equipment_days']} business days of approval. Home office items follow the expense rules: receipts over ${R['receipt_threshold_usd']}, Airbase submission.""", rel("exp","peo"))

def function_docs():
    for fn in FUNCTIONS:
        c = S['checkin_days']
        yield (f"{fn} onboarding: first week","onboarding",[fn.lower(),"onboarding"],
         f"""## Week one for {fn}
- Day 1: laptop, Okta and identity setup; meet your manager and your onboarding buddy.
- Complete required trainings and review the {fn} handbook section.
- Schedule intro chats with two teammates outside {fn}.
- Your buddy is assigned for {S['buddy_weeks']} weeks.""", rel("ld",))
        yield (f"{fn} onboarding: {c[0]}-{c[1]}-{c[2]} day plan","onboarding",[fn.lower(),"onboarding","milestones"],
         f"""## Milestones for {fn}
Formal check-ins happen at day {c[0]}, day {c[1]} and day {c[2]}.
- **Day {c[0]}**: environment ready, first deliverable scoped for {fn}.
- **Day {c[1]}**: first deliverable shipped; feedback collected from two collaborators.
- **Day {c[2]}**: ownership of a recurring {fn} responsibility; written reflection on onboarding.""", rel("ld",))
        yield (f"{fn} onboarding buddy guide","onboarding",[fn.lower(),"buddy"],
         f"""## Being a buddy for a new {fn} teammate
Buddies are paired for {S['buddy_weeks']} weeks. Meet at least weekly, answer questions about tools and norms, and introduce the new teammate to people across {fn}. Buddies do not replace the manager and are not responsible for performance feedback.""", rel("ld",))
        yield (f"{fn} manager onboarding checklist","onboarding",[fn.lower(),"manager"],
         f"""## Manager checklist for a new {fn} hire
- Before day 1: confirm equipment is ordered (shipped within {S['equipment_days']} business days), accounts requested, buddy picked.
- Week 1: hold a kickoff, share goals for the {S['checkin_days'][0]}-day check-in.
- Hold check-ins at day {S['checkin_days'][0]}, {S['checkin_days'][1]} and {S['checkin_days'][2]} and record notes.""", rel("ld",))

def expense_docs():
    for cat in EXPENSE_CATS:
        extra = pick(["Keep the receipt even for small purchases.","Ask your manager before buying if unsure.",
          "Use the team budget line where one exists.","Bundle recurring items to reduce requests."],cat)
        yield (f"Expense guide: {cat}","expenses",[slug(cat),"expenses"],
         f"""## {cat}
Submit through Airbase. Itemized receipts are required for any request over ${R['receipt_threshold_usd']}; reimbursement typically takes {R['reimb_days']} days.

A software or subscription purchase expected to cost more than ${R['software_approval_usd']} per year needs prior approval. Learning-related items such as courses and conferences draw on the ${R['education_usd']} per year professional development budget. Wellness items draw on the ${R['wellness_stipend_usd']} monthly stipend.

{extra}""", rel("exp","spend"))

def leave_docs():
    for reg in REGIONS:
        for key,label in LEAVE_SCEN:
            yield (f"Parental leave for {label} in {reg}","leave",[key,reg.lower(),"parental"],
             f"""## Scenario: {label} ({reg})
Eligible teammates receive 100% of salary for up to {R['parental_weeks']} weeks, within {R['parental_window_months']} months of the qualifying event. State or country benefits are supplemented so the total reaches full pay.

Notify your manager and People Team early; the Leave Case Manager contacts you within {R['leave_case_mgr_days']} business days of your completed questionnaire. For {key} cases, share the supporting date (birth, adoption or placement) when requested.""", rel("parental","loa"))

def manager_docs():
    for sit in SITUATIONS:
        for variant in ["first steps","what to avoid","documentation","escalation","follow-up"]:
            yield (f"Manager guide: when {sit} - {variant}","manager",[slug(sit),slug(variant)],
             f"""## {variant.capitalize()} when {sit}
Respond to the teammate within {S['manager_response_hours']} hours. Keep the conversation confidential, listen first, and involve the People Team when policy, leave, pay or legal questions arise. Do not promise outcomes that depend on policy you cannot confirm.

For leave over two weeks, direct the teammate to the leave process. For learning requests, the ${R['education_usd']} per year budget applies. Outside work requires review under the outside employment policy.""", rel("loa","oe","ld"))

def learning_docs():
    for sk in SKILLS:
        for kind in ["course plan","budget guide","manager approval notes","sharing what you learned"]:
            yield (f"Learning track: {sk} - {kind}","learning",[slug(sk),"learning"],
             f"""## {sk}: {kind}
Each teammate has ${R['education_usd']} per year for education and professional development (conferences and courses). A {sk} course, book or conference counts against this budget. Submit via Airbase with receipts for anything over ${R['receipt_threshold_usd']}.

Agree learning goals with your manager at the {S['checkin_days'][1]}-day check-in and share what you learned with the team afterwards.""", rel("ld","exp"))

def offboarding_docs():
    for fn in FUNCTIONS:
        for kind in ["checklist","manager handoff"]:
            yield (f"Offboarding {kind}: {fn}","offboarding",[fn.lower(),"offboarding"],
             f"""## {fn} offboarding: {kind}
- Access is removed on the {S['access_removal']}.
- Return equipment; open expense reports should be submitted before departure (receipts over ${R['receipt_threshold_usd']}).
- The exit survey is sent and should be completed within {S['exit_survey_days']} days.
- Hand over {fn} responsibilities and documentation to named owners.""", rel("leave","exp"))

def wellness_docs():
    for t in WELL:
        for k in ["resources","manager tips","how to use your stipend","when to ask for more help"]:
            yield (f"Wellbeing: {t} - {k}","wellness",[slug(t),"wellness"],
             f"""## {t}: {k}
Sourcegraph offers a ${R['wellness_stipend_usd']} monthly wellness stipend to use at your discretion, plus mental health support described in the benefits pages. Managers should check in regularly and point teammates to the People Team. For longer absences, see the leave of absence process.""", rel("benefits","mh","loa"))

def recruiting_docs():
    for st in STAGES:
        for role in ROLES:
            yield (f"Recruiting: {st} ({role})","recruiting",[slug(st),role],
             f"""## {st} for {role} roles
Interviewers submit a scorecard within {S['scorecard_hours']} hours. Candidates receive a decision or next step promptly. Offers require approval before being extended, and candidates have {S['offer_response_days']} business days to respond. Benefits discussed with candidates must match the benefits page; premiums at 100% apply to US-based full-time employees.""", rel("benefits","peo"))

def faq_docs():
    qs = [("How many weeks of paid parental leave?",f"Up to {R['parental_weeks']} weeks at 100% of salary per birth or placement, within {R['parental_window_months']} months.","parental"),
          ("What is the wellness stipend?",f"${R['wellness_stipend_usd']} per month, usable at your discretion.","benefits"),
          ("What is the education budget?",f"${R['education_usd']} per year for conferences and courses.","ld"),
          ("When do I need a receipt?",f"For any reimbursement request over ${R['receipt_threshold_usd']}.","exp"),
          ("How long do reimbursements take?",f"Expect {R['reimb_days']} days via Airbase.","exp"),
          ("When is approval needed for software?",f"For software or a subscription costing more than ${R['software_approval_usd']} per year.","spend"),
          ("Are insurance premiums covered?",f"Yes, {R['premium_pct']}% of medical, dental and vision premiums for US-based full-time employees and dependents.","benefits"),
          ("Who contacts me after I request leave?",f"Your Leave Case Manager, within {R['leave_case_mgr_days']} business days of the questionnaire.","loa")]
    ctx = ["new hires","managers","remote teammates","people returning from leave","recruiters","finance partners","executives","contractors converting to employees"]
    for i,(q,a,r) in enumerate(qs):
        for c in ctx:
            yield (f"FAQ for {c}: {q}","faq",[slug(c),"faq"],
             f"## {q}\n\nAsked often by {c}.\n\n**Answer:** {a}\n\nIf your situation is unusual, ask the People Team before acting.", rel(r))

FAMS = [region_docs,function_docs,expense_docs,leave_docs,manager_docs,learning_docs,
        offboarding_docs,wellness_docs,recruiting_docs,faq_docs]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="sourcegraph_files")
    ap.add_argument("--count", type=int, default=1000); a = ap.parse_args()
    out = Path(a.root)/"synthetic"
    if out.exists(): shutil.rmtree(out)
    docs, seen = [], set()
    # round-robin so truncation keeps every family represented
    gens = [list(f()) for f in FAMS]
    i = 0
    while len(docs) < a.count and any(gens):
        g = gens[i % len(gens)]
        if g:
            t,fam,tags,body,r = g.pop(0)
            if t not in seen:
                seen.add(t); docs.append((t,fam,tags,body,r))
        i += 1
    for t,fam,tags,body,r in docs:
        p = out/fam/f"{slug(t)}.md"; p.parent.mkdir(parents=True, exist_ok=True)
        # relative links point back into the real corpus
        r = [(n, os.path.relpath(Path(a.root)/path, p.parent)) for n,path in r]
        p.write_text(doc(t,fam,tags,body,r), encoding="utf-8")
    print(f"wrote {len(docs)} synthetic docs to {out}")
if __name__ == "__main__": main()
