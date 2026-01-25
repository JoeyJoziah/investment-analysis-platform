---
description: Perform comprehensive deal underwriting analysis. Analyzes credit, collateral, cash flow, and structure. Invokes the deal-underwriter agent.
---

# Underwrite Command

This command invokes the **deal-underwriter** agent for comprehensive credit and deal analysis.

## What This Command Does

1. **Credit Analysis** - Evaluate borrower creditworthiness and repayment capacity
2. **Collateral Assessment** - Analyze security package and lien structure
3. **Cash Flow Analysis** - Project cash flows and debt service capacity
4. **Structure Review** - Evaluate terms, covenants, and risk mitigants
5. **Decision Recommendation** - Provide approve/decline/modify recommendation

## When to Use

Use `/underwrite` when:
- Evaluating a new loan or investment opportunity
- Reviewing credit files for renewals
- Assessing deal modifications or restructures
- Analyzing intercreditor arrangements
- Reviewing collateral coverage

## How It Works

The deal-underwriter agent will:

1. **Gather Information** - Request deal summary, financials, and terms
2. **Spread Financials** - Normalize and analyze historical financials
3. **Calculate Ratios** - DSCR, leverage, coverage metrics
4. **Score Credit** - Apply scoring matrix to borrower
5. **Assess Collateral** - Evaluate security package and advance rates
6. **Review Structure** - Analyze covenants, pricing, and terms
7. **Identify Risks** - Flag concerns and propose mitigants
8. **Provide Recommendation** - Clear approve/decline with rationale

## Example Usage

```
User: /underwrite

Agent: I'll help you underwrite this deal. Please provide:

1. **Deal Summary**
   - Borrower name and entity type
   - Loan amount requested
   - Purpose of funds
   - Proposed terms

2. **Financial Information**
   - 3 years historical financials (or path to files)
   - Current year-to-date
   - Projections if available

3. **Collateral Information**
   - Proposed security package
   - Asset values/appraisals
   - Existing liens

Once I have this information, I'll perform comprehensive underwriting analysis.
```

## Output Format

The underwriting analysis includes:

### Credit Summary
| Factor | Score | Weight | Notes |
|--------|-------|--------|-------|
| Management | X/5 | 15% | [Assessment] |
| Industry | X/5 | 15% | [Assessment] |
| Financials | X/5 | 25% | [Assessment] |
| Cash Flow | X/5 | 20% | [Assessment] |
| Collateral | X/5 | 15% | [Assessment] |
| Guarantor | X/5 | 10% | [Assessment] |
| **Total** | **X/5** | | |

### Financial Analysis
- Revenue and EBITDA trends
- Key ratios (DSCR, leverage, coverage)
- Cash flow quality assessment

### Collateral Analysis
- Asset values and advance rates
- Lien priority and perfection status
- Coverage calculations

### Risk Assessment
- Identified risks by category
- Proposed mitigants
- Residual risk level

### Recommendation
- **Decision**: Approve / Conditional Approval / Decline
- **Pricing**: Suggested rate and fees
- **Structure**: Recommended terms and covenants
- **Conditions**: Requirements before funding

## Integration with Other Commands

After underwriting:
- Use `/model` to build detailed financial projections
- Use `/analyze-structure` to evaluate intercreditor arrangements
- Use `/scenario` for sensitivity analysis

## Related Agents

This command invokes the `deal-underwriter` agent located at:
`.claude/agents/deal-underwriter.md`
