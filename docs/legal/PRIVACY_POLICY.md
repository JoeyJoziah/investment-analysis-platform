> **STATUS: DRAFT — NOT LEGAL ADVICE. Requires review by qualified counsel before any use.**

---

# Privacy Policy

**[COMPANY NAME]** — Investment Analysis Platform

**Effective Date:** [EFFECTIVE DATE]
**Last Updated:** [EFFECTIVE DATE]
**Version:** 1.0-DRAFT

---

## Table of Contents

1. [Introduction and Scope](#1-introduction-and-scope)
2. [Data We Collect](#2-data-we-collect)
3. [How We Use Your Data](#3-how-we-use-your-data)
4. [Third-Party Services and Data Sources](#4-third-party-services-and-data-sources)
5. [Cookies and Tracking Technologies](#5-cookies-and-tracking-technologies)
6. [Data Retention](#6-data-retention)
7. [Security](#7-security)
8. [Your Rights — GDPR and CCPA](#8-your-rights--gdpr-and-ccpa)
9. [Children's Privacy](#9-childrens-privacy)
10. [Changes to This Policy](#10-changes-to-this-policy)
11. [Contact Information](#11-contact-information)

---

## 1. Introduction and Scope

[COMPANY NAME] ("we," "us," or "our") operates the Investment Analysis Platform (the "Service"), a web application that provides ML-generated stock analysis, market data, and investment-research tools for informational purposes.

This Privacy Policy describes:

- What personal data we collect when you use the Service
- How we use, store, and protect that data
- Which third-party services may process data on our behalf
- Your rights with respect to your data under applicable privacy law, including the EU General Data Protection Regulation ("GDPR") and the California Consumer Privacy Act ("CCPA")

This Policy applies to all users of the Service, regardless of location. Where legal obligations differ by jurisdiction, we note those distinctions.

**[PLACEHOLDER — COUNSEL REVIEW]:** Confirm whether a separate EU/UK representative appointment is required under GDPR Art. 27 / UK GDPR.

---

## 2. Data We Collect

### 2.1 Account Data

When you register for an account, we collect:

- Full name
- Email address
- Password (stored as a salted hash; we never store plaintext passwords)
- Date of account creation
- Account preferences and settings

### 2.2 Usage and Interaction Data

As you use the Service, we automatically collect:

- Pages and features accessed, and the time and duration of access
- Stock tickers, watchlists, and portfolios you create or view
- Searches and market-data queries you execute
- ML-generated recommendations you view or interact with
- Clicks and navigation paths within the application
- Feature flags and experiment variants served to your account

### 2.3 Technical and Device Data

- IP address (used for geolocation at the country/state level; anonymized in logs after [RETENTION PERIOD — see Section 6])
- Browser type and version
- Operating system
- Referring URL
- Session identifiers and authentication tokens

### 2.4 Communication Data

If you contact our support team or submit feedback:

- Your name and email address
- The content of your message
- Any attachments you provide

### 2.5 Financial Research Data

The Service retrieves publicly available market data (stock prices, fundamentals, news) from third-party data providers on your behalf when you make queries. We log the query parameters (e.g., ticker symbol, date range) to operate the Service and to comply with regulatory audit requirements. We do not collect, store, or process your brokerage account credentials, trade orders, or actual investment holdings unless you explicitly provide them.

### 2.6 Consent Records

Where we rely on your consent for a particular processing activity, we record:

- The consent type and version of the notice presented
- The timestamp and IP address at the time of consent
- Your consent status (granted / withdrawn)

We retain consent records for 10 years to demonstrate compliance, as described in Section 6.

---

## 3. How We Use Your Data

We process your personal data for the following purposes and on the following legal bases:

| Purpose | Data Used | Legal Basis (GDPR) |
|---|---|---|
| Create and manage your account | Account data | Contract performance (Art. 6(1)(b)) |
| Deliver the Service (market data queries, ML recommendations) | Usage data, query data | Contract performance (Art. 6(1)(b)) |
| Authenticate sessions and prevent unauthorized access | Account data, technical data | Legitimate interests (Art. 6(1)(f)) |
| Maintain audit logs for regulatory compliance | Usage data, technical data | Legal obligation (Art. 6(1)(c)) |
| Detect and prevent fraud, abuse, and security incidents | Technical data, usage data | Legitimate interests (Art. 6(1)(f)) |
| Improve the Service and ML models (aggregated, de-identified) | Aggregated usage data | Legitimate interests (Art. 6(1)(f)) |
| Respond to support requests | Communication data | Contract performance / Legitimate interests |
| Send transactional email (account notices, security alerts) | Account data | Contract performance (Art. 6(1)(b)) |
| Send optional product updates and newsletters | Account data | Consent (Art. 6(1)(a)) — opt-in only |
| Comply with law, court orders, or regulatory demands | As required | Legal obligation (Art. 6(1)(c)) |

**We do not sell your personal data to third parties.** We do not use your personal data to train or fine-tune our ML models without your explicit, informed consent.

**[PLACEHOLDER — COUNSEL REVIEW]:** Confirm CCPA "sale" / "sharing" determination, particularly for any analytics or advertising-adjacent services.

---

## 4. Third-Party Services and Data Sources

The Service integrates with the following third-party services. Each operates under its own privacy policy and terms of service; we encourage you to review them.

### 4.1 Market Data Providers

The Service retrieves publicly available financial data from the following APIs. Your query parameters (ticker symbol, date range) may be transmitted to these providers when you request market data:

| Provider | Data Type | Privacy Reference |
|---|---|---|
| **Polygon.io** | Real-time and historical stock prices, trades, quotes | [https://polygon.io/privacy](https://polygon.io/privacy) |
| **Finnhub** | Fundamentals, earnings, news, analyst estimates | [https://finnhub.io/privacy](https://finnhub.io/privacy) |
| **Alpha Vantage** | Historical OHLCV data, technical indicators | [https://www.alphavantage.co/privacy/](https://www.alphavantage.co/privacy/) |
| **NewsAPI** | Financial news headlines and articles | [https://newsapi.org/privacy](https://newsapi.org/privacy) |
| **Marketaux** | Financial news and sentiment data | [https://www.marketaux.com/privacy](https://www.marketaux.com/privacy) |
| **Federal Reserve Economic Data (FRED)** | Macroeconomic indicators | [https://fred.stlouisfed.org/legal/](https://fred.stlouisfed.org/legal/) |
| **Financial Modeling Prep (FMP)** | Fundamental financial data, SEC filings | [https://financialmodelingprep.com/privacy](https://financialmodelingprep.com/privacy) |

**[PLACEHOLDER — COUNSEL REVIEW]:** Confirm whether API gateway or server-side proxying sufficiently prevents direct user-to-provider data sharing, and whether Data Processing Agreements are required with any of the above under GDPR.

### 4.2 Infrastructure and Hosting

- **Cloud infrastructure provider**: [PROVIDER — e.g., AWS, GCP] — hosts application servers and databases. Data may be stored in [DATA CENTER REGION — e.g., US-East]. Subject to appropriate safeguards (Standard Contractual Clauses or equivalent) for EU/UK data transfers.
- **Database**: PostgreSQL hosted on [PROVIDER]. Financial transaction and audit log data is stored here.

### 4.3 Authentication

- **[AUTH PROVIDER — e.g., internal JWT / Auth0 / Clerk]**: Manages user session tokens and password hashing.

### 4.4 Monitoring and Error Tracking

- **[ERROR MONITORING PROVIDER — e.g., Sentry]**: Receives stack traces and error events. We configure these tools to exclude personal data from error payloads where possible.

### 4.5 Analytics

- **[ANALYTICS PROVIDER — e.g., none / self-hosted / PostHog]**: [PLACEHOLDER — describe or state "We do not use third-party web analytics at this time."]

**[PLACEHOLDER — COUNSEL REVIEW]:** Complete the infrastructure table with actual vendors; execute DPAs before launch.

---

## 5. Cookies and Tracking Technologies

We use a limited set of cookies and browser storage mechanisms:

| Name / Type | Purpose | Duration | Required |
|---|---|---|---|
| `session_id` | Maintains your authenticated session | Session (expires on browser close) | Yes |
| `auth_token` | Secure HTTP-only authentication token | [e.g., 7 days, refreshed on activity] | Yes |
| `disclaimer_[type]_acknowledged` | Records that you have read and acknowledged the investment risk disclaimer | 1 year | Yes (for compliance record) |
| Analytics cookies (if any) | Aggregate, anonymized usage metrics | [DURATION] | No — consent required |

We do not use third-party advertising cookies or cross-site tracking pixels.

If you block essential session cookies, certain features of the Service (including login) will not function.

**[PLACEHOLDER]:** Update the cookie table to reflect actual cookie names and durations confirmed during technical implementation review.

---

## 6. Data Retention

We retain personal data for as long as necessary for the purposes described in this Policy, subject to the following minimum retention periods driven by US regulatory and compliance obligations:

| Data Category | Retention Period | Regulatory Basis |
|---|---|---|
| User account data | Until account deletion, plus 90 days for recovery | Operational |
| Portfolio and watchlist data | Until account deletion | Operational |
| Financial transaction / recommendation-view records | 7 years from creation (anonymized after account deletion) | SEC Rule 17a-4 / financial records obligations |
| Audit logs (access, authentication events) | 7 years | SEC Rule 17a-4 |
| Consent records | 10 years | GDPR / CCPA best practice |
| System operational logs | 1 year | Operational |
| Support communications | 3 years from last interaction | Operational / legal claims |
| Anonymized, aggregated analytics | Indefinite (no personal data) | Operational |

After the applicable retention period, data is either deleted or irreversibly anonymized. Anonymized data is no longer personal data and is not subject to deletion requests.

**[PLACEHOLDER — COUNSEL REVIEW]:** Confirm whether the platform's recommendation-view records qualify as "financial records" triggering SEC Rule 17a-4 or Exchange Act Section 17 — this depends on the outcome of the adviser registration analysis (see `FORM_ADV_DECISION_MEMO.md`).

---

## 7. Security

We implement technical and organizational measures appropriate to the risk level, including:

- **Encryption in transit**: All data transmitted between your browser and our servers is encrypted using TLS 1.2 or higher.
- **Encryption at rest**: Database data is encrypted at rest using [ENCRYPTION STANDARD — e.g., AES-256].
- **Password hashing**: Passwords are stored using [HASHING ALGORITHM — e.g., bcrypt with work factor 12].
- **Access controls**: Role-based access controls limit employee access to personal data to those with a legitimate business need.
- **Audit logging**: All access to sensitive data is logged with immutable, cryptographically signed audit records.
- **Vulnerability management**: Dependencies are regularly scanned; security patches are applied promptly.

No security measure is perfectly infallible. In the event of a data breach that is likely to result in a high risk to your rights and freedoms, we will notify you and, where required, the appropriate supervisory authority within the legally required timeframe (72 hours under GDPR).

---

## 8. Your Rights — GDPR and CCPA

### 8.1 Rights Under GDPR (EU/UK Users)

If you are located in the European Economic Area (EEA) or the United Kingdom, you have the following rights under the GDPR / UK GDPR:

| Right | Description | How to Exercise |
|---|---|---|
| **Access (Art. 15)** | Obtain a copy of all personal data we hold about you | Submit a Data Subject Access Request to [CONTACT EMAIL] |
| **Rectification (Art. 16)** | Correct inaccurate or incomplete personal data | Update via your account settings or contact us |
| **Erasure ("Right to be Forgotten") (Art. 17)** | Request deletion of your personal data, subject to overriding legal obligations (e.g., audit log retention) | Submit a deletion request to [CONTACT EMAIL] |
| **Data Portability (Art. 20)** | Receive your personal data in a structured, machine-readable format | Request an export via account settings or contact us |
| **Restriction of Processing (Art. 18)** | Request that we limit processing of your data in certain circumstances | Contact us at [CONTACT EMAIL] |
| **Object to Processing (Art. 21)** | Object to processing based on legitimate interests or for direct marketing | Contact us at [CONTACT EMAIL] |
| **Withdraw Consent (Art. 7(3))** | Withdraw any consent you have given; this does not affect the lawfulness of processing prior to withdrawal | Use account settings opt-outs or contact us |

The Service provides the following self-service endpoints to support these rights:
- **Export your data**: `GET /gdpr/users/me/data-export`
- **Export your data (JSON)**: `GET /gdpr/users/me/data-export/json`
- **Request account deletion**: `POST /gdpr/users/me/delete-request`

We will respond to rights requests within 30 days. We may extend this period by a further two months for complex or numerous requests, with notice.

You have the right to lodge a complaint with your national data protection supervisory authority if you believe we have not handled your data lawfully.

### 8.2 Rights Under CCPA (California Residents)

If you are a California resident, you have the following rights under the California Consumer Privacy Act:

- **Right to Know**: Request disclosure of the categories and specific pieces of personal information we have collected about you, the sources of collection, the business or commercial purpose for collection, and the categories of third parties with whom we share it.
- **Right to Delete**: Request deletion of your personal information, subject to certain exceptions (e.g., information required to complete a transaction or comply with a legal obligation).
- **Right to Correct**: Request correction of inaccurate personal information.
- **Right to Opt-Out of Sale or Sharing**: We do not sell or share personal information for cross-context behavioral advertising. If this changes, we will update this Policy and provide an opt-out mechanism.
- **Right to Non-Discrimination**: We will not discriminate against you for exercising your privacy rights.

To exercise your CCPA rights, contact us at [CONTACT EMAIL] with the subject "CCPA Privacy Request."

We will verify your identity before processing requests to protect your privacy.

**[PLACEHOLDER — COUNSEL REVIEW]:** Confirm whether the CCPA "sharing" definition applies to any third-party analytics or data-provider integrations described in Section 4.

---

## 9. Children's Privacy

The Service is intended for users who are 18 years of age or older. We do not knowingly collect personal data from anyone under the age of 18. If you believe we have inadvertently collected data from a minor, please contact us immediately at [CONTACT EMAIL] and we will take prompt steps to delete it.

---

## 10. Changes to This Policy

We may update this Privacy Policy from time to time. When we do:

- The "Last Updated" date at the top of this page will be revised.
- If the changes are material, we will notify you by email (at the address associated with your account) and/or by displaying a prominent notice within the Service, at least 30 days before the changes take effect.
- Your continued use of the Service after the effective date of revised terms constitutes your acceptance of those changes.

We encourage you to review this Policy periodically.

---

## 11. Contact Information

For privacy-related questions, data requests, or concerns, please contact us at:

**[COMPANY NAME]**
Privacy Inquiries
[CONTACT EMAIL]

If you are located in the EU/EEA and we are required to appoint an EU representative under GDPR Art. 27, their contact details will be published here.

---

*This document is a draft for counsel review. It is not a legal opinion and does not constitute legal advice. [COMPANY NAME] and its operators make no representations that this document satisfies all applicable legal requirements. Qualified legal counsel must review and approve this document before it is published or relied upon.*
