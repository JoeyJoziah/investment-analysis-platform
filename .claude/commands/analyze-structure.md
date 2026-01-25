---
description: Analyze deal structure, security packages, and intercreditor arrangements. Reviews UCC liens, collateral, covenants, and legal documentation.
---

# Analyze Structure Command

This command analyzes deal structure, security packages, and legal documentation.

## What This Command Does

1. **Security Package Review** - Evaluate collateral and lien structure
2. **UCC Analysis** - Review filings, priority, and perfection
3. **Intercreditor Review** - Analyze multi-lender arrangements
4. **Covenant Analysis** - Evaluate financial and operational covenants
5. **Documentation Review** - Identify gaps and issues in legal docs

## When to Use

Use `/analyze-structure` when:
- Evaluating security packages for new deals
- Reviewing intercreditor agreements
- Analyzing UCC lien priority
- Assessing covenant structures
- Identifying documentation issues

## How It Works

The analysis process includes:

1. **Gather Documents** - Request relevant agreements and filings
2. **Map Collateral** - Identify all assets in security package
3. **Review Liens** - Analyze UCC filings and priority
4. **Evaluate Covenants** - Assess restrictiveness and enforceability
5. **Check Intercreditor** - Review senior/junior relationships
6. **Identify Issues** - Flag gaps, conflicts, and concerns
7. **Recommend Changes** - Suggest structure improvements

## Example Usage

```
User: /analyze-structure for multi-lender cannabis deal

Agent: I'll analyze the deal structure. Please provide:

**Security Package:**
- List of collateral (A/R, inventory, equipment, RE, etc.)
- Current UCC search results
- Appraisals or asset schedules

**Lender Information:**
- Number of lenders and positions (senior, mezz, junior)
- Facility amounts and terms
- Existing intercreditor agreement (if any)

**Documentation:**
- Security agreement
- Intercreditor agreement
- UCC-1 filings
- Loan agreement excerpts (covenants)

I'll analyze priority, coverage, and any structural concerns.
```

## Analysis Output

### Collateral Summary
| Asset Type | Value | Advance Rate | Available |
|------------|-------|--------------|-----------|
| A/R | $X | 80% | $X |
| Inventory | $X | 50% | $X |
| Equipment | $X | 70% | $X |
| Real Estate | $X | 65% | $X |
| **Total** | **$X** | | **$X** |

### Lien Priority Analysis
```
Position 1 (Senior): [Lender Name]
- Filing Date: [Date]
- Collateral: All assets
- Perfection Status: ✓ Perfected

Position 2 (Mezz): [Lender Name]
- Filing Date: [Date]
- Collateral: All assets (subordinate)
- Perfection Status: ✓ Perfected
- Intercreditor: Subordination agreement dated [Date]

Issues Identified:
- [ ] Gap in filing (state X not covered)
- [ ] Collateral description inconsistency
- [ ] Missing control agreement for deposit accounts
```

### Intercreditor Analysis
| Provision | Senior Position | Junior Position | Risk Level |
|-----------|-----------------|-----------------|------------|
| Standstill | 180 days | 180 days | Low |
| Payment Blockage | All defaults | Payment default only | Medium |
| Purchase Option | Par + 1% | Par + 1% | Low |
| Cross-Default | Broad | Broad | High |

### Covenant Analysis
| Covenant | Threshold | Current | Headroom | Risk |
|----------|-----------|---------|----------|------|
| Min DSCR | 1.25x | 1.45x | 16% | Low |
| Max Leverage | 4.0x | 3.2x | 20% | Low |
| Min Liquidity | $500K | $800K | 60% | Low |

### Issues and Recommendations

**High Priority:**
1. [Issue]: Missing UCC filing in [State]
   - **Recommendation**: File immediately to perfect lien
   - **Risk if not addressed**: Loss of priority

**Medium Priority:**
2. [Issue]: Intercreditor silent on [topic]
   - **Recommendation**: Negotiate amendment
   - **Risk if not addressed**: Dispute in enforcement

**Low Priority:**
3. [Issue]: Minor covenant definition inconsistency
   - **Recommendation**: Clarify in next amendment
   - **Risk if not addressed**: Interpretation dispute

## Integration with Other Commands

Works well with:
- Use `/underwrite` for comprehensive credit analysis
- Use `/model` for financial projections
- Use `/scenario` to stress test covenant headroom

## Skills Used

This command leverages the `deal-structuring` skill located at:
`.claude/skills/deal-structuring/SKILL.md`
