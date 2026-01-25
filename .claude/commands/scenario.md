---
description: Run scenario and sensitivity analysis on financial models. Creates base, downside, and upside cases with stress testing.
---

# Scenario Command

This command runs scenario and sensitivity analysis on financial projections.

## What This Command Does

1. **Scenario Modeling** - Create base, downside, severe, and upside cases
2. **Sensitivity Analysis** - Two-way tables on key drivers
3. **Stress Testing** - Test covenant and coverage thresholds
4. **Break-Even Analysis** - Find trigger points for key metrics
5. **Probability Weighting** - Calculate expected outcomes

## When to Use

Use `/scenario` when:
- Testing investment thesis under different assumptions
- Stress testing covenant headroom
- Analyzing downside protection
- Presenting range of outcomes to stakeholders
- Validating model robustness

## How It Works

The scenario analysis process:

1. **Define Base Case** - Establish baseline assumptions
2. **Identify Drivers** - Key variables that impact outcomes
3. **Set Ranges** - Define realistic ranges for each driver
4. **Build Scenarios** - Create discrete scenarios
5. **Run Sensitivities** - Generate two-way tables
6. **Calculate Probabilities** - Weight scenarios if applicable
7. **Present Results** - Summary with visualizations

## Example Usage

```
User: /scenario for LBO model

Agent: I'll run scenario analysis on the LBO model. Please confirm:

**Base Case Assumptions:**
- Revenue CAGR: 10%
- EBITDA margin: 25%
- Exit multiple: 8.0x
- Entry leverage: 4.0x

**Key Drivers to Analyze:**
1. Revenue growth (range: 5% - 15%)
2. EBITDA margin (range: 20% - 30%)
3. Exit multiple (range: 6.0x - 10.0x)

**Scenarios to Model:**
- Base: Management case
- Downside: Revenue -20%, margin -3%
- Severe: Revenue -30%, margin -5%
- Upside: Revenue +10%, margin +2%

Confirm these parameters and I'll generate the analysis.
```

## Scenario Structure

### Discrete Scenarios

| Scenario | Revenue Growth | EBITDA Margin | Exit Multiple | Probability |
|----------|---------------|---------------|---------------|-------------|
| Upside | 15% | 28% | 9.0x | 15% |
| Base | 10% | 25% | 8.0x | 50% |
| Downside | 5% | 22% | 7.0x | 25% |
| Severe | 0% | 18% | 6.0x | 10% |

### Scenario Outputs

```
                  Upside    Base    Downside   Severe
Revenue Y5        $XXM      $XXM    $XXM       $XXM
EBITDA Y5         $XXM      $XXM    $XXM       $XXM
Exit EV           $XXM      $XXM    $XXM       $XXM
Exit Equity       $XXM      $XXM    $XXM       $XXM
IRR               XX%       XX%     XX%        XX%
MOIC              X.Xx      X.Xx    X.Xx       X.Xx
DSCR (Min)        X.Xx      X.Xx    X.Xx       X.Xx
```

### Expected Value Calculation

```
Expected IRR = Σ (Probability × Scenario IRR)
             = 15% × 35% + 50% × 22% + 25% × 12% + 10% × -5%
             = 18.7%

Expected MOIC = Σ (Probability × Scenario MOIC)
              = 15% × 3.2x + 50% × 2.4x + 25% × 1.8x + 10% × 0.8x
              = 2.2x
```

## Sensitivity Analysis

### Two-Way Sensitivity Tables

```
IRR Sensitivity: Revenue Growth vs. Exit Multiple

              Exit Multiple
              6.0x   7.0x   8.0x   9.0x   10.0x
Revenue  5%   8%     12%    15%    18%    21%
Growth  8%   12%    16%    19%    22%    25%
       10%   15%    19%    22%    25%    28%
       12%   18%    22%    25%    28%    31%
       15%   22%    26%    29%    32%    35%
```

### Break-Even Analysis

```
Break-Even Points:

For 15% IRR target:
- Minimum revenue growth: 7%
- Minimum EBITDA margin: 21%
- Minimum exit multiple: 6.5x

For 1.25x DSCR covenant:
- Maximum revenue decline: -18%
- Maximum margin compression: 4%
- Break-even revenue: $X
```

## Stress Testing

### Covenant Stress Test

| Covenant | Threshold | Current | Stress Test | Headroom |
|----------|-----------|---------|-------------|----------|
| DSCR | 1.25x | 1.45x | Rev -15% | 6% cushion |
| Leverage | 4.0x | 3.2x | Rev -20% | Passes |
| Liquidity | $500K | $800K | Rev -25% | Fails at -22% |

### Cash Flow Stress Test

```
Stress Scenario: 6-Month Revenue Stop

Month 1: Cash $800K → $650K (Burn: $150K)
Month 2: Cash $650K → $500K (Burn: $150K)
Month 3: Cash $500K → $400K (Burn: $100K) *Cost cuts*
Month 4: Cash $400K → $320K (Burn: $80K)
Month 5: Cash $320K → $250K (Burn: $70K)
Month 6: Cash $250K → $190K (Burn: $60K)

Result: Company survives 6 months with cost cuts
Liquidity covenant breach: Month 3
Recommendation: Require $300K debt service reserve
```

## Output Format

### Scenario Summary Dashboard

```
╔════════════════════════════════════════════════════════╗
║             SCENARIO ANALYSIS SUMMARY                  ║
╠════════════════════════════════════════════════════════╣
║                                                        ║
║  IRR Range:        12% - 35%  (Base: 22%)             ║
║  MOIC Range:       1.8x - 3.2x (Base: 2.4x)           ║
║  Expected IRR:     18.7%                               ║
║  Expected MOIC:    2.2x                                ║
║                                                        ║
║  Downside Protection:                                  ║
║  • Positive IRR in Downside: ✓                        ║
║  • Capital preserved in Severe: ✗ (0.8x MOIC)         ║
║  • Covenant headroom: 16% revenue decline              ║
║                                                        ║
║  Key Sensitivities:                                    ║
║  1. Exit multiple: 300 bps IRR per 1.0x               ║
║  2. Revenue growth: 150 bps IRR per 1%                ║
║  3. EBITDA margin: 100 bps IRR per 1%                 ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
```

## Integration with Other Commands

Best used after:
- `/model` to build the base financial model
- `/underwrite` to understand the credit

Use before:
- Final investment decision
- Committee presentations

## Related Skills

This command leverages the `financial-modeling` skill located at:
`.claude/skills/financial-modeling/SKILL.md`
