import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "evaluation_set.json"

TOPIC_TEMPLATES = [
    {
        "theme": "Health benefits eligibility",
        "questions": [
            "Are US-based full-time teammates eligible for health benefits on their date of hire?",
            "Do dependents qualify for Sourcegraph health benefits?",
            "What is the deadline for US new hires to enroll in insurance after joining?",
            "Can a teammate waive health coverage during the initial enrollment window?",
            "When can a teammate change benefits outside of open enrollment?",
        ],
        "gold": "Sourcegraph provides health benefits for US-based full-time employees and their eligible dependents. Benefits are effective on the date of hire, and US-based new hires have 30 days from their start date to enroll. Coverage can be waived if a teammate declines it, and plan changes outside open enrollment are allowed only after a qualifying life event.",
        "sources": ["benefits-pay-perks/benefits-perks/benefits.md"],
    },
    {
        "theme": "Open enrollment",
        "questions": [
            "What is open enrollment for benefits?",
            "When did the 2024 US benefits open enrollment take place?",
            "How do teammates review or update their current benefits selections?",
            "Are plan changes effective immediately or only on a fixed date?",
            "Can employees make insurance changes outside open enrollment for any reason?",
        ],
        "gold": "Open enrollment is the annual period for reviewing and changing benefit elections, adding or removing dependents, and reviewing costs. Sourcegraph's 2024 open enrollment ran from 2023-11-20 to 2023-12-01, and changes became effective as of 2024-01-01. Outside of open enrollment, changes are only permitted for a qualifying life event.",
        "sources": ["benefits-pay-perks/benefits-perks/benefits.md"],
    },
    {
        "theme": "Travel and insurance",
        "questions": [
            "Does the health benefit cover teammates while traveling internationally for fun?",
            "What coverage exists when a teammate travels for work on behalf of Sourcegraph?",
            "Are international leisure trips covered by the regular US health plans?",
            "What should a teammate do when traveling for vacation and wanting insurance?",
            "Who provides the corporate travel insurance for business travel?",
        ],
        "gold": "The regular health benefits do not cover international leisure travel. For business travel on behalf of Sourcegraph, the company has a corporate policy providing travel assistance and medical coverage. For vacation travel, teammates are advised to purchase separate travel insurance, such as GeoBlue.",
        "sources": ["benefits-pay-perks/benefits-perks/benefits.md"],
    },
    {
        "theme": "Remote work",
        "questions": [
            "Is Sourcegraph a remote-first company?",
            "Does a teammate need permission to step away during the workday?",
            "How does Sourcegraph support team cohesion in an all-remote culture?",
            "Is there an expectation to work weekends or after hours?",
            "What tools help remote coworkers build informal connections?",
        ],
        "gold": "Sourcegraph is an all-remote company and teammates are encouraged to work remotely. There is no requirement to ask permission for a brief step away, and the company tries to limit synchronous meetings while allowing flexibility across time zones. There is no expectation of weekend responsiveness except for rare situations with explicit notice, and the company supports remote cohesion through async communication, 1:1s, and watercooler routines.",
        "sources": ["company-info-and-process/remote/index.md"],
    },
    {
        "theme": "Holidays and PTO",
        "questions": [
            "How does Sourcegraph handle local holidays across different countries?",
            "Do teammates have to list each country holiday individually?",
            "Are company meetings rescheduled around certain US holidays?",
            "Does the company observe a holiday blackout period in late December?",
            "How should teammates communicate their holiday dates?",
        ],
        "gold": "Sourcegraph respects local holidays and does not require teammates to enumerate holidays by country because holidays are covered as part of the unlimited PTO policy. Teammates should add their local holidays to their calendar and notify their manager and teammates in advance. Certain company meetings and onboarding sessions are rescheduled if they fall on selected observed holidays, and there is a company blackout between December 23 and January 1.",
        "sources": ["company-info-and-process/working-at-sourcegraph/holidays.md"],
    },
    {
        "theme": "Respectful workplace",
        "questions": [
            "What is the purpose of the respectful workplace policy?",
            "Who does the respectful workplace policy apply to?",
            "What kind of behavior is prohibited?",
            "Does the policy cover interactions with external vendors and partners?",
            "What should a teammate do if they feel disrespected at work?",
        ],
        "gold": "The respectful workplace policy exists to ensure that everyone at Sourcegraph and those who interact with teammates feels welcomed and included. It applies to all teammates globally, including contractors, and also covers interactions with third parties such as vendors, applicants, and partners. The policy prohibits harassment, discrimination, and bullying and treats allegations seriously.",
        "sources": ["company-info-and-process/respectful-workplace-policy.md"],
    },
    {
        "theme": "Leave and absence",
        "questions": [
            "What kind of leave guidance is documented in the policy corpus?",
            "Does Sourcegraph offer short-term and long-term disability?",
            "Where can a teammate find medical leave guidance?",
            "Can leave decisions be made solely by an AI assistant?",
            "What should a teammate do if they need formal leave support?",
        ],
        "gold": "The documents note that Sourcegraph does not offer short-term or long-term disability at the time of writing and direct teammates to the medical leave of absence policy. Leave and accommodation decisions require a human review, not a machine authorization, and the assistant should avoid acting as the sole decision-maker.",
        "sources": ["benefits-pay-perks/benefits-perks/leave-of-absence.md", "benefits-pay-perks/benefits-perks/benefits.md"],
    },
    {
        "theme": "Equality and inclusion",
        "questions": [
            "What is Sourcegraph's general approach to equal opportunity?",
            "Does the company provide paid time off for a range of reasons?",
            "What values are emphasized around workplace equality?",
            "Does Sourcegraph commit to an inclusive environment?",
            "How should the company treat LGBTQIA+ teammates?",
        ],
        "gold": "Sourcegraph states it is committed to equal opportunity and an environment free from harassment and discrimination. The policy documents also call out paid time off arrangements that account for well-being, religious observance, life events, and holidays, and the company emphasizes inclusion for LGBTQIA+ teammates and employees broadly.",
        "sources": ["company-info-and-process/equality-of-opportunity.md", "company-info-and-process/queer.md"],
    },
    {
        "theme": "Company communication norms",
        "questions": [
            "What communication style does Sourcegraph prefer for remote teams?",
            "Why is written communication especially important at an all-remote company?",
            "What is the best practice for async communication?",
            "How can teammates reduce friction when writing messages across time zones?",
            "What should writers do to make async communication less cold?",
        ],
        "gold": "Sourcegraph emphasizes async, written communication because the company is all-remote and distributed across time zones. Written communication is effective when writers anticipate questions and lead with positivity, and the company explicitly encourages thoughtful, written documentation to replace spontaneous in-person cues.",
        "sources": ["company-info-and-process/communication/asynchronous-communication.md", "company-info-and-process/remote/index.md"],
    },
    {
        "theme": "Onboarding and setup",
        "questions": [
            "What should a new teammate do when they have not received a Rippling invite?",
            "Where do new teammates get their payroll and tax information?",
            "What systems are used for job onboarding and device access?",
            "Can a teammate use work devices for personal reasons?",
            "What is the policy for privacy on company devices?",
        ],
        "gold": "New teammates use Rippling to access pay stubs, tax forms, and benefits information, and they should contact the People Team if the invitation is missing. Company devices are subject to privacy expectations, and personal usage is fine only when it does not interfere with work or violate policy. Sourcegraph also relies on systems such as Okta and device management for secure onboarding.",
        "sources": ["benefits-pay-perks/benefits-perks/benefits.md", "departments/tech-ops/process/team_device_usage_privacy.md"],
    },
    {
        "theme": "Security and vendor reviews",
        "questions": [
            "What is the general process for reviewing a new vendor in security?",
            "Who requests vendor reviews?",
            "What kinds of vendor data issues are considered?",
            "Does the security policy require a formal checklist?",
            "Can a teammate approve a vendor without a review?",
        ],
        "gold": "Vendor reviews are requested by teammates when a new vendor is being onboarded, and they cover aspects such as data processing, data protection, security reporting, and the type of data being shared. The review is intended to assess security risk before a vendor is approved, and the system should not authorize vendor access without a human review.",
        "sources": ["departments/security/security-support-rotation.md"],
    },
    {
        "theme": "Data privacy and acceptable use",
        "questions": [
            "What is the general privacy expectation for company devices?",
            "Is personal use of company equipment allowed?",
            "What does the acceptable use policy cover?",
            "Can employees use company devices for unrelated personal tasks?",
            "When would personal device use become prohibited?",
        ],
        "gold": "The policy documents say that privacy is a core principle at Sourcegraph and that acceptable use must not interfere with work or violate laws. Personal use of company devices is allowed only if it does not disrupt work, create policy violations, or create operational or legal risk.",
        "sources": ["company-info-and-process/policies/acceptable-use-policy.md", "departments/tech-ops/process/team_device_usage_privacy.md"],
    },
]


def build_entries() -> list[dict]:
    entries = []
    for index, topic in enumerate(TOPIC_TEMPLATES, start=1):
        for pos, question in enumerate(topic["questions"], start=1):
            entries.append({
                "id": f"Q{index:02d}-{pos:02d}",
                "theme": topic["theme"],
                "question": question,
                "gold_answer": topic["gold"],
                "expected_sources": topic["sources"],
            })
    return entries


def main() -> None:
    dataset = build_entries()
    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(dataset)} evaluation entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
