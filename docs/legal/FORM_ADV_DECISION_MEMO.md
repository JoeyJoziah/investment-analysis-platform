> **STATUS: DRAFT — NOT LEGAL ADVICE. Requires review by qualified counsel before any use.**
>
> **HUMAN-BLOCKING: This memo does NOT make the registration decision. The decision is reserved for [OWNER: Devin McGrath] and qualified legal counsel. No action should be taken based solely on this document.**

---

# Form ADV Decision Memo
## Does the Investment Analysis Platform Trigger Investment Adviser Registration?

**Prepared for:** [COMPANY NAME] — Internal Use Only
**Date:** 2026-06-23
**Version:** 1.0-DRAFT
**Classification:** CONFIDENTIAL — Attorney-Client Privilege Intended

---

## Purpose of This Memo

This memo frames the question that must be answered before the platform launches to US retail users:

> **Does the Investment Analysis Platform's "recommendation" feature require the company to register as an Investment Adviser under the Investment Advisers Act of 1940 ("IAA" or "the Act") and/or applicable state law?**

This is a **threshold legal question**. The answer determines:

- Whether the platform's current design is permissible as-is;
- What disclaimers and structural changes may be necessary;
- Whether registration (and all of its obligations) is required.

This memo is a **structured decision framework** for counsel review. It presents:
1. The legal test (general, educational summary)
2. The platform's factual profile mapped against that test
3. Available options
4. A recommended path

**This memo does not provide legal advice. It does not constitute a legal opinion. It does not determine whether registration is required. That determination must be made by qualified securities counsel after a full facts-and-circumstances review.**

---

## 1. The Investment Advisers Act of 1940: The Three-Prong Test

> *The following is a general educational summary of US securities law, provided to frame the analysis. It is not a complete or definitive statement of the law. Laws and SEC interpretations change; counsel must verify current standards.*

Under Section 202(a)(11) of the Investment Advisers Act of 1940, an "investment adviser" is generally defined as a person or firm that:

1. **Provides advice or analysis** regarding securities (including recommending specific securities or categories of securities, or furnishing reports or analyses concerning securities);
2. **As part of a regular business** (i.e., is in the business of providing such advice, not merely as an incidental part of another business); and
3. **For compensation** (broadly interpreted to include direct fees, subscription fees, or any economic benefit received, directly or indirectly, in connection with providing the advice).

All three prongs must generally be satisfied for the full registration regime to apply. **However, meeting all three prongs does not automatically mean registration is the only path** — exceptions and exemptions may apply depending on the specific facts, the number of clients, their sophistication, the state of operation, and other factors.

### 1.1 Prong 1 — Advice About Securities

The first prong is met if the platform:
- Provides recommendations about whether to buy, hold, or sell specific securities;
- Furnishes analysis about the value of specific securities;
- Issues reports or analyses that a reasonable person would understand as telling them what to do with specific securities.

General market commentary, educational content about how markets work, or broad macroeconomic analysis generally does not meet this prong without more. But specific "BUY AAPL" or "SELL TSLA" outputs (even if algorithm-generated) are closer to the advisory end of the spectrum.

### 1.2 Prong 2 — In the Business of Providing Such Advice

Relevant factors include:
- Whether providing the advice is a central purpose of the enterprise (vs. purely incidental);
- Whether the firm holds itself out as providing investment advice;
- The frequency with which advice is provided;
- Whether the advice is the basis on which users are attracted to and retained by the service.

### 1.3 Prong 3 — For Compensation

Compensation includes:
- Direct subscription fees paid by users;
- Indirect compensation (e.g., selling data derived from user activity);
- Any economic benefit received in exchange for advice, even if the advice is bundled with other services.

### 1.4 Exemptions and Exclusions

Even if all three prongs are met, certain exemptions and exclusions may apply, including (among others):

- **Publisher exemption (§202(a)(11)(D))**: Bona fide publishers of "financial publications of general and regular circulation" may be excluded — but the SEC and courts have applied this narrowly; interactive platforms with personalized outputs generally do not qualify.
- **SEC-registered investment adviser exemption thresholds**: Smaller advisers (fewer than a threshold number of clients, below an AUM threshold) may register at the state level rather than with the SEC.
- **Impersonal investment advice exception**: Under some older guidance, advice given in general, non-personalized publications to an unspecified public may not trigger registration — but this is increasingly narrow territory with algorithmic personalization.

**[PLACEHOLDER — COUNSEL REVIEW]:** Counsel must analyze current SEC no-action letters, enforcement actions, and recent guidance on algorithmic/robo-adviser registration requirements. The SEC's framework for robo-advisers (2017 guidance and subsequent updates) is directly relevant.

---

## 2. Platform Factual Profile — Mapped Against the Three-Prong Test

The following represents our current understanding of the platform's features. **Counsel should verify these facts against the current codebase and product specification before rendering an opinion.**

| Factor | Current Platform State | Registration Implication |
|---|---|---|
| **Specific security recommendations** | Yes — ML models generate BUY/SELL/HOLD outputs for individual tickers | Pushes toward "advice about securities" (Prong 1) |
| **Personalization** | Partially — outputs are currently computed without individualized suitability screening (no risk tolerance questionnaire, no KYC) | Cuts both ways: lack of personalization may support "impersonal" argument, but also means no suitability safeguards |
| **Compensation** | Subscription fee model (~$50/mo infrastructure target; user pricing TBD) | If fees are charged, Prong 3 is likely met |
| **In the business of providing advice** | The recommendation feature is a central, advertised feature of the platform | Pushes toward Prong 2 |
| **Number of clients** | Pre-launch; client count unknown | Relevant to state vs. SEC registration threshold |
| **Retail users targeted** | Yes — "retail investment-analysis web platform" | No institutional/sophisticated investor carve-out applies |
| **SEC registration** | Not currently registered | Registration gap identified in SEC audit (0% — CRITICAL) |
| **State registration** | Not currently registered in any state | State-level obligations in addition to federal |
| **Disclaimer language** | Implemented at API level; partially rendered in UI; TERMS_OF_SERVICE.md has "not investment advice" language | Disclaimers alone may not resolve the registration question |

### 2.1 Factors That Push Toward Registration Requirement

- The platform produces individual ticker-level BUY/SELL/HOLD outputs that a retail user could act on directly.
- The recommendation feature is marketed as a core capability.
- A subscription fee is anticipated, satisfying Prong 3.
- Users are retail individuals, not institutional or sophisticated investors.
- The SEC's own audit gap analysis (docs/SEC_REGULATORY_COMPLIANCE_AUDIT.md) flags adviser registration as a CRITICAL 0% item.

### 2.2 Factors That May Cut Against Registration Requirement (or Support an Exemption)

- The platform currently lacks individualized suitability analysis (no KYC, no risk-tolerance questionnaire), which may support an "impersonal advice" characterization — though this cuts against users as well.
- The "informational purposes only" disclaimers, if sufficiently prominent and user-acknowledged, may support a positioning argument.
- If the number of clients remains below the state registration threshold in the company's primary state, a lighter-touch state-level regime might apply.
- Restructuring the product to present outputs as screening results or research data (rather than recommendations) may reduce exposure under Prong 1.

---

## 3. Options

### Option A: Informational-Only Positioning with Robust Disclaimers (No Registration)

**Description**: Restructure the presentation of ML outputs so they are clearly framed as research screens, scores, or analytical data points — not "recommendations." Implement robust, user-acknowledged disclaimers at every touchpoint. Avoid language like "BUY," "SELL," "HOLD" in favor of neutral quantitative labels (e.g., score quartile, signal strength). Add a clear non-advisory disclaimer in the onboarding flow with explicit user acknowledgment.

**Potential advantages**:
- No registration cost, compliance infrastructure, or ongoing regulatory filings.
- Faster path to launch.
- Maintains flexibility in product design.

**Potential risks**:
- The SEC has historically looked at the substance of what a platform does, not just its labels. Relabeling "BUY" as "Strong Signal" does not automatically resolve the registration question if the substance is the same.
- If the SEC or a state regulator later determines registration was required, the company could face retroactive enforcement, disgorgement of fees, and reputational damage.
- This option requires sustained discipline in product language, marketing, and feature design; any slip back toward personalized advice could re-trigger the question.
- Does not provide the legal protection that registration and fiduciary compliance would afford.

**Prerequisites if pursuing Option A**:
- Counsel sign-off on specific product language and feature design;
- Comprehensive disclaimer architecture (onboarding acknowledgment + persistent UI warnings + ToS);
- User experience audit to ensure disclaimers are not buried;
- Periodic reassessment as the product evolves.

### Option B: Register as a Registered Investment Adviser (RIA)

**Description**: File Form ADV with the SEC (if AUM/client count meets federal thresholds) or with applicable state securities regulators. Operate under the full IAA fiduciary framework.

**Potential advantages**:
- Regulatory clarity; reduces legal risk of enforcement.
- Enables more explicit "recommendation" language and positioning.
- Builds long-term trust with users and potential institutional partners.

**Potential risks**:
- Significant compliance burden: ongoing Form ADV filings, compliance manual, annual updates, potential examination by SEC or state regulators.
- Fiduciary obligations: must act in clients' best interest, manage conflicts of interest, implement suitability procedures.
- May require employees to hold Series 65 (Investment Adviser Representative) license.
- Significantly higher operational and legal cost.
- May affect product design constraints (suitability screening, disclosure requirements for each recommendation).

**Prerequisites if pursuing Option B**:
- Retention of securities counsel experienced in RIA registration;
- Form ADV Part 1 and Part 2A (brochure) preparation;
- Compliance program buildout;
- Potential personnel licensing;
- Ongoing annual amendment filings.

### Option C: Restructure the Recommendation Feature

**Description**: Remove or fundamentally restructure the individual stock recommendation outputs so the platform qualifies as a purely analytical / research tool without advisory outputs. For example: provide only quantitative screening data, sector-level analysis, or raw factor scores without a composite directional signal.

**Potential advantages**:
- May cleanly avoid the first prong of the adviser test.
- Maintains informational positioning without the compliance burden of Option B.

**Potential risks**:
- May reduce the core value proposition of the product.
- May not be achievable if users interpret any analytical output as a "recommendation."
- Still requires careful counsel review of the specific feature design.

---

## 4. Recommended Path (Pending Counsel Confirmation)

> **IMPORTANT: The following represents a preliminary internal working hypothesis, not a legal conclusion. It is offered as a starting point for counsel review only. Devin McGrath and qualified securities counsel must make the final decision.**

**Preliminary recommended path: Option A — Informational-Only Positioning with Robust Disclaimers, subject to counsel review and sign-off.**

This path is recommended as the starting hypothesis because:

1. The platform is pre-launch with an unknown client base; registration thresholds and exemptions may apply.
2. The cost and operational overhead of immediate RIA registration (Option B) is disproportionate to the current scale of the business.
3. Option A is viable if — and only if — counsel confirms that the specific product language, feature design, and disclaimer architecture meet the threshold for non-advisory positioning.
4. Option A is reversible: if the product evolves toward more personalized advice, or if counsel's analysis changes, registration can be pursued later.

**However, Option A is NOT a safe default without counsel sign-off.** The disclaimers drafted in `TERMS_OF_SERVICE.md` and `PRIVACY_POLICY.md` are necessary but not sufficient on their own. They must be paired with product-level design decisions that counsel approves.

---

## 5. Decision Framework: Questions for Counsel

The following questions should be answered by qualified securities counsel before launch:

1. Based on the current feature set (BUY/SELL/HOLD outputs, subscription model, retail users), do all three prongs of the IAA test appear to be met?
2. Does the publisher exemption or any other exemption plausibly apply to the current product design?
3. What specific product language changes, if any, are necessary to support an informational-only positioning under Option A?
4. Is the disclaimer architecture in `TERMS_OF_SERVICE.md` sufficient, or must it be supplemented by structural product changes?
5. Are there any state-specific registration requirements in [STATE OF INCORPORATION] or [PRIMARY USER MARKET] that apply regardless of the federal analysis?
6. If Option A is pursued, what constitutes a material change to the product that would require re-consulting counsel?
7. Does the SEC's 2017 Guidance on Robo-Advisers (IM-2017-2) apply to this platform, and if so, what obligations does it create?

---

## 6. Next Steps — Human Decision Required

| Action | Owner | Deadline |
|---|---|---|
| Retain securities counsel with robo-adviser / fintech experience | Devin McGrath | Before launch |
| Share this memo and current product spec with counsel | Devin McGrath | Before launch |
| Obtain written counsel opinion on registration question | Counsel | Before launch |
| Update TERMS_OF_SERVICE.md Section 3.5 based on counsel's determination | Engineering / Legal | Before launch |
| Implement counsel-approved disclaimer architecture in UI | Engineering | Before launch |
| Confirm recurring check-in cadence if Option A pursued | Devin McGrath + Counsel | Before launch |

**This section requires a human decision. The platform must not launch to retail US users until the registration question has been resolved by counsel.**

---

## 7. Source Materials

The following project documents informed this memo and should be provided to counsel for review:

| Document | Location |
|---|---|
| SEC Regulatory Compliance Audit | `docs/SEC_REGULATORY_COMPLIANCE_AUDIT.md` |
| Compliance Action Plan | `docs/COMPLIANCE_ACTION_PLAN.md` |
| Compliance Summary | `docs/COMPLIANCE_SUMMARY.txt` |
| Compliance README | `docs/COMPLIANCE_README.md` |
| Terms of Service (Draft) | `docs/legal/TERMS_OF_SERVICE.md` |
| Privacy Policy (Draft) | `docs/legal/PRIVACY_POLICY.md` |
| Backend recommendation router | `backend/api/routers/recommendations.py` |
| Backend settings / API keys | `backend/config/settings.py` |

---

## 8. Limitations and Disclaimers

- This memo was prepared by non-lawyers for internal use only. It is not a legal opinion.
- This memo does not constitute the practice of law and does not create an attorney-client relationship.
- Securities law is complex, fact-specific, and subject to change. The analysis above may not reflect the most current SEC guidance, no-action letters, or enforcement priorities.
- State securities laws vary significantly and have not been analyzed in this memo.
- This memo addresses only the US federal Investment Advisers Act of 1940. It does not address non-US regulatory regimes (e.g., FCA in the UK, MiFID II in the EU, or similar).
- **Nothing in this memo should be relied upon as legal advice or as a basis for business decisions without independent review by qualified legal counsel.**

---

*This document is a draft for counsel review. It is not a legal opinion and does not constitute legal advice. [COMPANY NAME] and its operators make no representations that this document satisfies all applicable legal or regulatory requirements. Qualified securities counsel must review this document before any action is taken.*
