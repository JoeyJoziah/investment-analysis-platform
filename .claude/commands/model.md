---
description: Build or analyze financial models including DCF, LBO, and scenario analysis. Invokes the financial-modeler agent for quantitative analysis.
---

# Model Command

This command invokes the **financial-modeler** agent for building and analyzing financial models.

## What This Command Does

1. **Model Building** - Create DCF, LBO, or operating models
2. **Valuation Analysis** - Calculate enterprise and equity value
3. **Returns Modeling** - Project IRR, multiples, and cash flows
4. **Scenario Analysis** - Build base, downside, and upside cases
5. **Sensitivity Tables** - Two-way sensitivity on key drivers

## When to Use

Use `/model` when:
- Building new financial models from scratch
- Analyzing investment returns
- Performing valuation analysis
- Creating scenario and sensitivity analysis
- Consolidating financial data

## How It Works

The financial-modeler agent will:

1. **Understand Requirements** - Clarify model type and outputs needed
2. **Gather Data** - Request financial inputs and assumptions
3. **Build Structure** - Create model architecture and formulas
4. **Calculate Outputs** - Valuation, returns, cash flows
5. **Run Scenarios** - Model multiple cases
6. **Generate Sensitivities** - Create two-way tables
7. **Document Assumptions** - List all inputs and sources

## Example Usage

```
User: /model DCF for cannabis retail company

Agent: I'll build a DCF valuation model. Please provide:

**Historical Financials (3 years):**
- Revenue
- COGS and Gross Margin
- Operating Expenses
- EBITDA

**Projection Assumptions:**
- Revenue growth rates (by year)
- Target EBITDA margins
- CapEx requirements
- Working capital assumptions

**Valuation Inputs:**
- Discount rate / WACC
- Terminal growth rate
- Comparable company multiples (for sanity check)

**280E Considerations:**
- Current tax treatment
- Effective tax rate

Let me know if you have data files to share or want to input assumptions manually.
```

## Model Types Supported

### DCF (Discounted Cash Flow)
- Revenue build-up
- EBITDA projections
- Free cash flow calculation
- Terminal value (Gordon growth or exit multiple)
- WACC derivation

### LBO (Leveraged Buyout)
- Sources and uses
- Debt schedule with tranches
- Cash flow to equity
- IRR and multiple calculation
- Sensitivity on leverage and exit multiple

### Operating Model
- Revenue drivers and build-up
- Cost structure analysis
- Working capital modeling
- CapEx and D&A schedules
- Cash flow forecasting

### Consolidation Model
- Multi-entity consolidation
- Intercompany eliminations
- Segment reporting
- Management reporting package

## Output Format

### Valuation Summary
```
Enterprise Value (DCF):    $XX,XXX,XXX
  PV of Projected FCF:     $X,XXX,XXX
  PV of Terminal Value:    $X,XXX,XXX

Implied Multiples:
  EV / Revenue:            X.Xx
  EV / EBITDA:             X.Xx

Equity Value:              $XX,XXX,XXX
  Enterprise Value:        $XX,XXX,XXX
  Less: Net Debt:          ($X,XXX,XXX)
```

### Returns Analysis
```
Investment Summary:
  Entry Equity:            $X,XXX,XXX
  Exit Equity:             $XX,XXX,XXX
  Holding Period:          X years

Returns:
  IRR:                     XX.X%
  MOIC:                    X.Xx
  Cash Yield:              X.X% p.a.
```

### Sensitivity Table
```
                 Exit Multiple
                 6.0x   7.0x   8.0x   9.0x   10.0x
Revenue   8%     15%    18%    22%    25%    28%
Growth   10%     18%    22%    25%    28%    31%
         12%     22%    25%    28%    31%    35%
         14%     25%    28%    32%    35%    38%
```

## Integration with Other Commands

Works well with:
- Use `/underwrite` first to understand the credit
- Use `/scenario` for detailed scenario analysis
- Use `/analyze-structure` for deal terms review

## Related Agents

This command invokes the `financial-modeler` agent located at:
`.claude/agents/financial-modeler.md`
