# Portfolio Dashboard Design Specification

## Investment Analysis Platform - UI/Visualization Design

**Version:** 1.0
**Last Updated:** 2026-01-25
**Target:** React.js + Material-UI + Recharts/Plotly
**WCAG Compliance:** 2.1 AA

---

## 1. ASCII Wireframe - Desktop Layout (1200px+)

```
+-----------------------------------------------------------------------------------+
|  HEADER: Investment Dashboard                                    [Search] [Alerts] |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|  +------------------+  +------------------+  +------------------+  +--------------+ |
|  | PORTFOLIO VALUE  |  | TOTAL RETURN     |  | DAY P&L          |  | AI SENTIMENT | |
|  | $1,245,678       |  | +24.5%           |  | +$2,340 (+0.19%) |  | BULLISH      | |
|  | +$2,340 today    |  | $245,678         |  | vs S&P: +0.08%   |  | 78/100       | |
|  +------------------+  +------------------+  +------------------+  +--------------+ |
|                                                                                    |
|  +------------------------------------------------+  +----------------------------+ |
|  |  PERFORMANCE CHART                              |  | AI RECOMMENDATIONS         | |
|  |  [1D] [1W] [1M] [3M] [YTD] [1Y] [ALL]          |  |                            | |
|  |                                                 |  | +------------------------+ | |
|  |     ___________                                 |  | | AAPL  BUY   95% conf   | | |
|  |    /           \                                |  | | $185 -> $210 (+13.5%)  | | |
|  |   /             \___                            |  | +------------------------+ | |
|  |  /                  \____                       |  | +------------------------+ | |
|  | /                        \                      |  | | NVDA  BUY   92% conf   | | |
|  |/                          \_____                |  | | $875 -> $950 (+8.6%)   | | |
|  |                                                 |  | +------------------------+ | |
|  | [Volume bars below]                             |  | +------------------------+ | |
|  |  ||| |||| ||| || ||||| ||| ||                   |  | | GOOGL HOLD  88% conf   | | |
|  +------------------------------------------------+  | | $142 -> $145 (+2.1%)   | | |
|                                                       | +------------------------+ | |
|                                                       |        [View All ->]       | |
|                                                       +----------------------------+ |
|                                                                                    |
|  +------------------------------------------------+  +----------------------------+ |
|  |  HOLDINGS TABLE                                 |  | ALLOCATION                 | |
|  |  +--------+--------+-------+--------+-------+  |  |                            | |
|  |  | Symbol | Shares | Price | Value  | P&L   |  |  |    [====] Tech 35%        | |
|  |  +--------+--------+-------+--------+-------+  |  |    [===]  Health 22%      | |
|  |  | AAPL   | 150    |$185.4 |$27,810 | +12%  |  |  |    [==]   Finance 18%    | |
|  |  | MSFT   | 100    |$378.2 |$37,820 | +8%   |  |  |    [==]   Consumer 15%   | |
|  |  | GOOGL  | 75     |$141.8 |$10,635 | +5%   |  |  |    [=]    Energy 10%     | |
|  |  | ...    | ...    | ...   | ...    | ...   |  |  |                            | |
|  |  +--------+--------+-------+--------+-------+  |  |   [PIE CHART VISUAL]       | |
|  |            [Show All Holdings]                  |  |                            | |
|  +------------------------------------------------+  +----------------------------+ |
|                                                                                    |
|  +-------------------------------+  +--------------------------------------------+ |
|  |  MARKET OVERVIEW              |  | NEWS & ALERTS                              | |
|  |                               |  |                                            | |
|  | S&P 500  4,892  +0.45%       |  | * AAPL: Earnings beat expectations...     | |
|  | NASDAQ   15,628 +0.62%       |  | * Fed signals rate decision next week...  | |
|  | DOW      38,245 +0.28%       |  | * NVDA: New AI chip announcement...       | |
|  |                               |  | * Alert: MSFT hit target price           | |
|  | [Sector Heatmap]              |  |                           [View All ->]   | |
|  +-------------------------------+  +--------------------------------------------+ |
|                                                                                    |
+-----------------------------------------------------------------------------------+
```

---

## 2. Component Hierarchy & Specifications

### 2.1 Component Tree

```
PortfolioDashboard/
├── DashboardLayout/
│   ├── Header/
│   │   ├── PageTitle
│   │   ├── DateDisplay
│   │   ├── SearchButton
│   │   ├── NotificationsButton
│   │   └── RefreshButton
│   │
│   ├── MetricsRow/
│   │   ├── PortfolioValueCard
│   │   ├── TotalReturnCard
│   │   ├── DayPnLCard
│   │   └── AISentimentCard
│   │
│   ├── MainContent/
│   │   ├── PerformanceSection/
│   │   │   ├── PerformanceChart
│   │   │   ├── PeriodSelector
│   │   │   └── ChartControls
│   │   │
│   │   └── RecommendationsPanel/
│   │       ├── PanelHeader
│   │       ├── RecommendationCardList/
│   │       │   └── RecommendationCardCompact (x3)
│   │       └── ViewAllButton
│   │
│   ├── SecondaryContent/
│   │   ├── HoldingsSection/
│   │   │   ├── HoldingsTable
│   │   │   │   ├── TableHeader
│   │   │   │   ├── TableBody/
│   │   │   │   │   └── HoldingRow (xN)
│   │   │   │   └── TablePagination
│   │   │   └── QuickActions
│   │   │
│   │   └── AllocationSection/
│   │       ├── AllocationChart (Donut)
│   │       ├── AllocationLegend
│   │       └── DiversificationScore
│   │
│   └── FooterContent/
│       ├── MarketOverview/
│       │   ├── IndexCards
│       │   └── SectorHeatmap
│       │
│       └── NewsFeed/
│           ├── NewsCardList/
│           │   └── NewsCard (x5)
│           └── AlertsList
│
└── Modals/
    ├── SearchModal
    ├── NotificationsPanel
    └── StockDetailModal
```

### 2.2 Component Specifications

#### PortfolioValueCard
```typescript
interface PortfolioValueCardProps {
  totalValue: number;
  dayChange: number;
  dayChangePercent: number;
  isLoading?: boolean;
}

// Dimensions
// Desktop: 280px width, 120px height
// Tablet: 100% width, 100px height
// Mobile: 100% width, 90px height

// Content Layout:
// - Label (top): "Portfolio Value" - Caption text
// - Value (center): Currency formatted - H4 typography
// - Change (bottom): Arrow icon + value + percent - Body2 typography
```

#### PerformanceChart
```typescript
interface PerformanceChartProps {
  data: PerformancePoint[];
  period: '1D' | '1W' | '1M' | '3M' | 'YTD' | '1Y' | 'ALL';
  onPeriodChange: (period: string) => void;
  showVolume?: boolean;
  showBenchmark?: boolean;
  height?: number;
}

interface PerformancePoint {
  date: string;
  value: number;
  volume?: number;
  benchmark?: number;
}

// Dimensions
// Desktop: 100% width, 350px height (chart) + 50px (controls)
// Tablet: 100% width, 300px height
// Mobile: 100% width, 250px height

// Features:
// - Area chart with gradient fill
// - Optional volume bars (secondary axis)
// - Period selector (toggle buttons)
// - Tooltip on hover with date, value, change
// - Optional S&P 500 benchmark line
```

#### HoldingsTable
```typescript
interface HoldingsTableProps {
  positions: Position[];
  onSort: (column: string, direction: 'asc' | 'desc') => void;
  onRowClick: (ticker: string) => void;
  isLoading?: boolean;
  maxRows?: number;
}

// Columns:
// Symbol (sticky) | Company | Shares | Avg Cost | Price | Value | Day P&L | Total P&L | Actions

// Features:
// - Sortable columns
// - Color-coded P&L (green/red)
// - Sparkline trend in Symbol column
// - Quick action buttons (Trade, Alert, Details)
// - Virtualized for large lists (100+ positions)
```

#### AllocationChart
```typescript
interface AllocationChartProps {
  data: AllocationData[];
  type: 'sector' | 'asset' | 'geography';
  interactive?: boolean;
  showLabels?: boolean;
}

interface AllocationData {
  name: string;
  value: number;
  percentage: number;
  color: string;
}

// Dimensions
// Desktop: 280px x 280px
// Tablet/Mobile: 100% width, aspect-ratio 1:1

// Features:
// - Donut chart with center label
// - Hover highlights segment
// - Click to filter holdings
// - Legend with percentages
```

#### RecommendationCardCompact
```typescript
interface RecommendationCardCompactProps {
  ticker: string;
  companyName: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  currentPrice: number;
  targetPrice: number;
  potentialReturn: number;
  onClick: () => void;
}

// Dimensions
// Desktop: 280px width, 100px height
// Mobile: 100% width, 90px height

// Layout:
// [Avatar] [Ticker/Company] [Action Badge]
//          [Price -> Target] [Return %]
```

---

## 3. Design Tokens

### 3.1 Color System

```typescript
const financialColors = {
  // Gain/Loss Colors (WCAG AA Compliant)
  gain: {
    primary: '#2E7D32',    // Green 800 - main text
    light: '#4CAF50',      // Green 500 - icons, badges
    bg: '#E8F5E9',         // Green 50 - backgrounds
    border: '#A5D6A7',     // Green 200 - borders
  },
  loss: {
    primary: '#C62828',    // Red 800 - main text
    light: '#EF5350',      // Red 400 - icons, badges
    bg: '#FFEBEE',         // Red 50 - backgrounds
    border: '#EF9A9A',     // Red 200 - borders
  },
  neutral: {
    primary: '#616161',    // Grey 700
    light: '#9E9E9E',      // Grey 500
    bg: '#F5F5F5',         // Grey 100
  },

  // Recommendation Actions
  actions: {
    buy: '#2E7D32',        // Green
    sell: '#C62828',       // Red
    hold: '#F57C00',       // Orange 700
    strongBuy: '#1B5E20',  // Dark Green
    strongSell: '#B71C1C', // Dark Red
  },

  // Risk Levels
  risk: {
    low: '#2E7D32',
    medium: '#F57C00',
    high: '#C62828',
  },

  // Confidence Levels
  confidence: {
    high: '#2E7D32',       // 80-100%
    medium: '#F57C00',     // 50-79%
    low: '#C62828',        // 0-49%
  },

  // Sector Colors (for pie charts)
  sectors: [
    '#1976D2', // Technology - Blue
    '#388E3C', // Healthcare - Green
    '#7B1FA2', // Finance - Purple
    '#F57C00', // Consumer - Orange
    '#D32F2F', // Energy - Red
    '#0288D1', // Industrials - Light Blue
    '#C2185B', // Materials - Pink
    '#689F38', // Utilities - Light Green
    '#5D4037', // Real Estate - Brown
    '#455A64', // Communication - Blue Grey
  ],
};

// Dark Mode Variants
const financialColorsDark = {
  gain: {
    primary: '#66BB6A',    // Green 400
    light: '#81C784',      // Green 300
    bg: 'rgba(46, 125, 50, 0.15)',
    border: '#388E3C',
  },
  loss: {
    primary: '#EF5350',    // Red 400
    light: '#E57373',      // Red 300
    bg: 'rgba(198, 40, 40, 0.15)',
    border: '#D32F2F',
  },
  // ... similar pattern
};
```

### 3.2 Spacing Scale

```typescript
const spacing = {
  // Component internal padding
  cardPadding: {
    desktop: 24,    // 3 * 8px
    tablet: 20,     // 2.5 * 8px
    mobile: 16,     // 2 * 8px
  },

  // Grid gaps
  gridGap: {
    desktop: 24,
    tablet: 16,
    mobile: 12,
  },

  // Section margins
  sectionMargin: {
    desktop: 32,
    tablet: 24,
    mobile: 20,
  },

  // Element spacing
  elementGap: {
    xs: 4,
    sm: 8,
    md: 12,
    lg: 16,
    xl: 24,
  },
};
```

### 3.3 Typography Hierarchy

```typescript
const typography = {
  // Page title
  pageTitle: {
    fontSize: '2rem',       // 32px
    fontWeight: 600,
    lineHeight: 1.2,
    letterSpacing: '-0.01em',
  },

  // Card value (large numbers)
  metricValue: {
    fontSize: '1.75rem',    // 28px
    fontWeight: 700,
    lineHeight: 1.2,
    fontFamily: '"SF Mono", Monaco, monospace', // Monospace for numbers
  },

  // Section headers
  sectionTitle: {
    fontSize: '1.25rem',    // 20px
    fontWeight: 600,
    lineHeight: 1.4,
  },

  // Card labels
  label: {
    fontSize: '0.75rem',    // 12px
    fontWeight: 500,
    lineHeight: 1.5,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    color: 'text.secondary',
  },

  // Table headers
  tableHeader: {
    fontSize: '0.75rem',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },

  // Table body
  tableBody: {
    fontSize: '0.875rem',   // 14px
    fontWeight: 400,
    lineHeight: 1.5,
  },

  // Ticker symbols
  ticker: {
    fontSize: '0.875rem',
    fontWeight: 700,
    fontFamily: '"SF Mono", Monaco, monospace',
    letterSpacing: '0.02em',
  },

  // Price values
  price: {
    fontSize: '0.875rem',
    fontWeight: 500,
    fontFamily: '"SF Mono", Monaco, monospace',
  },

  // Percentage changes
  percentage: {
    fontSize: '0.875rem',
    fontWeight: 600,
    fontFamily: '"SF Mono", Monaco, monospace',
  },
};
```

### 3.4 Elevation & Shadows

```typescript
const elevation = {
  // Card base
  card: {
    light: '0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06)',
    dark: '0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.24)',
  },

  // Card hover
  cardHover: {
    light: '0 4px 12px rgba(0,0,0,0.1), 0 2px 4px rgba(0,0,0,0.06)',
    dark: '0 4px 12px rgba(0,0,0,0.5), 0 2px 4px rgba(0,0,0,0.3)',
  },

  // Floating elements (modals, popovers)
  floating: {
    light: '0 10px 40px rgba(0,0,0,0.12)',
    dark: '0 10px 40px rgba(0,0,0,0.6)',
  },

  // Pressed/active state
  pressed: {
    light: 'inset 0 2px 4px rgba(0,0,0,0.08)',
    dark: 'inset 0 2px 4px rgba(0,0,0,0.3)',
  },
};
```

---

## 4. Responsive Behavior

### 4.1 Breakpoint Strategy

```typescript
const breakpoints = {
  xs: 0,      // Mobile portrait
  sm: 600,    // Mobile landscape / small tablet
  md: 960,    // Tablet
  lg: 1280,   // Desktop
  xl: 1920,   // Large desktop
};
```

### 4.2 Layout Transformations

#### Desktop (1200px+)
```
+-----------------------------------------------------------+
| [MetricCard] [MetricCard] [MetricCard] [MetricCard]       |
+-----------------------------------------------------------+
| [PerformanceChart - 8 cols]   | [Recommendations - 4 cols] |
+-----------------------------------------------------------+
| [HoldingsTable - 8 cols]      | [AllocationChart - 4 cols] |
+-----------------------------------------------------------+
| [MarketOverview - 4 cols]     | [NewsFeed - 8 cols]        |
+-----------------------------------------------------------+
```

#### Tablet (768-1199px)
```
+-------------------------------------------+
| [MetricCard] [MetricCard]                 |
| [MetricCard] [MetricCard]                 |
+-------------------------------------------+
| [PerformanceChart - 12 cols]              |
+-------------------------------------------+
| [Recommendations - 6 cols] | [Allocation] |
+-------------------------------------------+
| [HoldingsTable - 12 cols]                 |
+-------------------------------------------+
| [MarketOverview] | [NewsFeed]             |
+-------------------------------------------+
```

#### Mobile (<768px)
```
+---------------------------+
| [MetricCard - swipeable]  |
+---------------------------+
| [PerformanceChart]        |
+---------------------------+
| [Recommendations]         |
+---------------------------+
| [Allocation]              |
+---------------------------+
| [HoldingsTable]           |
| (horizontal scroll)       |
+---------------------------+
| [MarketOverview]          |
+---------------------------+
| [NewsFeed]                |
+---------------------------+
```

### 4.3 Component Responsive Behavior

| Component | Desktop | Tablet | Mobile |
|-----------|---------|--------|--------|
| MetricCards | 4 in row | 2x2 grid | Horizontal scroll |
| PerformanceChart | 8 cols + sidebar | Full width | Full width, reduced height |
| HoldingsTable | Full table | Condensed columns | Horizontal scroll |
| AllocationChart | Sidebar | Below recommendations | Full width |
| RecommendationsPanel | Sidebar | Half width | Full width |
| MarketOverview | 4 cols | 6 cols | Full width |
| NewsFeed | 8 cols | 6 cols | Full width |

---

## 5. Interaction Patterns

### 5.1 Hover States

```typescript
const hoverEffects = {
  // Card hover
  card: {
    transform: 'translateY(-2px)',
    boxShadow: elevation.cardHover,
    transition: 'all 0.2s ease-out',
  },

  // Table row hover
  tableRow: {
    backgroundColor: 'rgba(0, 0, 0, 0.02)', // light mode
    // backgroundColor: 'rgba(255, 255, 255, 0.02)', // dark mode
  },

  // Interactive element focus
  focus: {
    outline: '2px solid primary.main',
    outlineOffset: '2px',
  },

  // Button hover
  button: {
    backgroundColor: 'rgba(25, 118, 210, 0.08)',
    transform: 'scale(1.02)',
  },
};
```

### 5.2 Click/Tap Behaviors

| Element | Action | Result |
|---------|--------|--------|
| Metric Card | Click | Navigate to detailed analytics |
| Chart Period Button | Click | Update chart timeframe |
| Recommendation Card | Click | Open stock detail modal |
| Holdings Row | Click | Navigate to stock page |
| Allocation Segment | Click | Filter holdings by sector |
| News Item | Click | Expand or open article |
| Sort Header | Click | Toggle sort direction |

### 5.3 Loading States

```typescript
const loadingStates = {
  // Skeleton for metric cards
  metricCardSkeleton: {
    height: 120,
    animation: 'pulse 1.5s ease-in-out infinite',
    borderRadius: 8,
  },

  // Chart loading
  chartLoading: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: 350,
    // Show CircularProgress
  },

  // Table skeleton
  tableRowSkeleton: {
    height: 52,
    animation: 'pulse 1.5s ease-in-out infinite',
    // Repeat for 5 rows
  },

  // Shimmer effect
  shimmer: `
    background: linear-gradient(
      90deg,
      transparent 0%,
      rgba(255,255,255,0.4) 50%,
      transparent 100%
    );
    animation: shimmer 2s infinite;
  `,
};
```

### 5.4 Error States

```typescript
const errorStates = {
  // Connection lost
  connectionError: {
    banner: true,
    message: 'Connection lost. Reconnecting...',
    color: 'warning',
    icon: 'WifiOff',
    action: 'Retry',
  },

  // Data fetch failed
  fetchError: {
    inline: true,
    message: 'Failed to load data',
    color: 'error',
    icon: 'ErrorOutline',
    action: 'Try Again',
  },

  // Stale data indicator
  staleData: {
    chip: true,
    message: 'Data may be outdated',
    color: 'warning',
    showTimestamp: true,
  },
};
```

---

## 6. Data Visualization Specifications

### 6.1 Portfolio Performance Line/Area Chart

```typescript
interface PerformanceChartConfig {
  type: 'area';

  // Main price line
  primaryLine: {
    color: theme.palette.primary.main,
    strokeWidth: 2,
    fill: 'url(#gradientArea)',
    fillOpacity: 0.15,
  };

  // Gradient definition
  gradient: {
    id: 'gradientArea',
    stops: [
      { offset: '0%', color: 'primary.main', opacity: 0.3 },
      { offset: '100%', color: 'primary.main', opacity: 0 },
    ],
  };

  // Volume bars (secondary)
  volumeBars: {
    color: theme.palette.grey[400],
    opacity: 0.5,
    yAxis: 'secondary',
  };

  // Benchmark line (optional)
  benchmarkLine: {
    color: theme.palette.grey[500],
    strokeWidth: 1,
    strokeDasharray: '4 4',
    label: 'S&P 500',
  };

  // Axes
  xAxis: {
    tickFormatter: (date) => format(date, 'MMM dd'),
    tickCount: 6,
    axisLine: false,
    tickLine: false,
  };

  yAxis: {
    tickFormatter: (value) => `$${formatNumber(value)}`,
    axisLine: false,
    tickLine: false,
    domain: ['auto', 'auto'],
  };

  // Tooltip
  tooltip: {
    background: theme.palette.background.paper,
    border: '1px solid',
    borderColor: theme.palette.divider,
    borderRadius: 8,
    padding: 12,
    content: ({ date, value, change }) => `
      <div>${format(date, 'MMM dd, yyyy')}</div>
      <div>Value: $${formatNumber(value)}</div>
      <div style="color: ${change >= 0 ? green : red}">
        ${change >= 0 ? '+' : ''}${change.toFixed(2)}%
      </div>
    `,
  };

  // Crosshair
  crosshair: {
    stroke: theme.palette.text.secondary,
    strokeDasharray: '3 3',
  };

  // Reference lines
  referenceLines: [
    { y: initialValue, label: 'Start', stroke: grey[400] },
  ];
}
```

### 6.2 Sector Allocation Donut Chart

```typescript
interface AllocationChartConfig {
  type: 'pie';
  innerRadius: '60%';
  outerRadius: '85%';
  paddingAngle: 2;

  // Center label
  centerLabel: {
    primary: 'Total',
    secondary: '$1,245,678',
    primaryStyle: typography.label,
    secondaryStyle: typography.metricValue,
  };

  // Segment styling
  segments: {
    stroke: theme.palette.background.paper,
    strokeWidth: 2,
    cornerRadius: 4,
  };

  // Hover effect
  activeSegment: {
    outerRadius: '90%',
    scale: 1.05,
    shadow: '0 4px 12px rgba(0,0,0,0.15)',
  };

  // Labels
  labels: {
    show: true,
    position: 'outside',
    formatter: ({ name, percentage }) => `${name}\n${percentage}%`,
  };

  // Legend
  legend: {
    position: 'bottom',
    layout: 'horizontal',
    align: 'center',
    iconType: 'circle',
    iconSize: 10,
  };

  // Colors
  colors: financialColors.sectors;
}
```

### 6.3 Holdings Sparkline

```typescript
interface SparklineConfig {
  type: 'line';
  width: 60;
  height: 24;
  data: number[]; // Last 30 days closing prices

  // Line styling
  line: {
    stroke: (trend) => trend >= 0 ? financialColors.gain.primary : financialColors.loss.primary,
    strokeWidth: 1.5,
    strokeLinecap: 'round',
  };

  // No axes, labels, or grid
  axes: false;
  grid: false;
  tooltip: false;

  // Optional: fill area
  fill: {
    show: false,
  };
}
```

### 6.4 Risk Gauge

```typescript
interface RiskGaugeConfig {
  type: 'gauge';
  min: 0;
  max: 100;
  value: number;

  // Arc styling
  arc: {
    width: 12,
    cornerRadius: 6,
    padAngle: 0.02,
  };

  // Color zones
  zones: [
    { from: 0, to: 33, color: financialColors.risk.low },
    { from: 33, to: 66, color: financialColors.risk.medium },
    { from: 66, to: 100, color: financialColors.risk.high },
  ];

  // Needle
  needle: {
    length: 0.8,
    width: 4,
    color: theme.palette.text.primary,
  };

  // Center label
  centerLabel: {
    value: value,
    label: getRiskLabel(value), // 'Low', 'Medium', 'High'
  };
}
```

---

## 7. Accessibility (WCAG 2.1 AA)

### 7.1 Color Contrast Requirements

| Element | Foreground | Background | Ratio | Requirement |
|---------|------------|------------|-------|-------------|
| Body text | #212121 | #FFFFFF | 16.1:1 | AA (4.5:1) |
| Gain text | #2E7D32 | #FFFFFF | 5.2:1 | AA (4.5:1) |
| Loss text | #C62828 | #FFFFFF | 5.9:1 | AA (4.5:1) |
| Disabled | #9E9E9E | #FFFFFF | 3.5:1 | AA Large (3:1) |
| Dark mode body | #E0E0E0 | #0A0E27 | 12.4:1 | AA (4.5:1) |
| Dark mode gain | #66BB6A | #0A0E27 | 7.8:1 | AA (4.5:1) |
| Dark mode loss | #EF5350 | #0A0E27 | 5.4:1 | AA (4.5:1) |

### 7.2 Color-Independent Indicators

Always pair color with additional indicators:

```typescript
// Gain/Loss indicators
<Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
  {change >= 0 ? (
    <TrendingUpIcon aria-hidden="true" />
  ) : (
    <TrendingDownIcon aria-hidden="true" />
  )}
  <Typography
    color={change >= 0 ? 'success.main' : 'error.main'}
    aria-label={`${change >= 0 ? 'Gain' : 'Loss'} of ${Math.abs(change)} percent`}
  >
    {change >= 0 ? '+' : ''}{change.toFixed(2)}%
  </Typography>
</Box>

// Recommendation action badges
<Chip
  label={action}
  icon={
    action === 'BUY' ? <AddIcon /> :
    action === 'SELL' ? <RemoveIcon /> :
    <PauseIcon />
  }
  sx={{ bgcolor: getActionColor(action) }}
/>
```

### 7.3 Keyboard Navigation

```typescript
const keyboardNavigation = {
  // Tab order
  tabIndex: {
    metricCards: 1,
    chartPeriodButtons: 2,
    recommendations: 3,
    holdingsTable: 4,
    allocationChart: 5,
    marketOverview: 6,
    newsFeed: 7,
  },

  // Focus trap for modals
  modalFocusTrap: true,

  // Skip links
  skipLinks: [
    { label: 'Skip to main content', target: '#main-content' },
    { label: 'Skip to recommendations', target: '#recommendations' },
    { label: 'Skip to holdings', target: '#holdings' },
  ],

  // Table keyboard navigation
  table: {
    ArrowDown: 'Move to next row',
    ArrowUp: 'Move to previous row',
    Enter: 'Activate row (open details)',
    Space: 'Select row',
  },

  // Chart keyboard interaction
  chart: {
    ArrowLeft: 'Previous data point',
    ArrowRight: 'Next data point',
    Home: 'First data point',
    End: 'Last data point',
  },
};
```

### 7.4 Screen Reader Announcements

```typescript
const ariaAnnouncements = {
  // Live regions for real-time updates
  priceUpdate: {
    role: 'status',
    'aria-live': 'polite',
    'aria-atomic': true,
    message: (ticker, price, change) =>
      `${ticker} price updated to $${price}, ${change >= 0 ? 'up' : 'down'} ${Math.abs(change)} percent`,
  },

  // Data loading
  loading: {
    'aria-busy': true,
    'aria-describedby': 'loading-message',
  },

  // Chart description
  chartDescription: {
    'aria-label': (period, startValue, endValue, change) =>
      `Portfolio performance chart for ${period}. Starting value $${startValue}, ending value $${endValue}, ${change >= 0 ? 'gain' : 'loss'} of ${Math.abs(change)} percent.`,
  },

  // Table summary
  tableSummary: {
    'aria-label': (count) =>
      `Holdings table with ${count} positions. Sortable by column headers.`,
  },
};
```

### 7.5 Focus Management

```typescript
const focusManagement = {
  // Visible focus indicator
  focusVisible: {
    outline: '2px solid',
    outlineColor: 'primary.main',
    outlineOffset: '2px',
    borderRadius: '4px',
  },

  // Focus restoration after modal close
  restoreFocus: true,

  // Auto-focus on page load
  initialFocus: '#portfolio-value-card',

  // Focus on error
  errorFocus: '#error-message',
};
```

---

## 8. Real-Time Updates

### 8.1 WebSocket Integration

```typescript
interface RealtimeConfig {
  // Connection settings
  connection: {
    url: 'wss://api.example.com/ws',
    reconnectInterval: 5000,
    maxReconnectAttempts: 10,
  };

  // Subscriptions
  subscriptions: [
    'portfolio.value',
    'portfolio.positions',
    'market.indices',
    'recommendations.updates',
  ];

  // Update frequency
  throttle: {
    priceUpdates: 1000,    // 1 second
    portfolioValue: 5000,  // 5 seconds
    recommendations: 30000, // 30 seconds
  };

  // Visual feedback
  updateIndicator: {
    flash: {
      duration: 500,
      color: 'rgba(25, 118, 210, 0.3)',
    },
    pulse: {
      animation: 'pulse 0.5s ease-out',
    },
  };
}
```

### 8.2 Update Animations

```typescript
const updateAnimations = {
  // Price flash (green/red)
  priceFlash: {
    keyframes: `
      @keyframes priceFlash {
        0% { background-color: transparent; }
        50% { background-color: var(--flash-color); }
        100% { background-color: transparent; }
      }
    `,
    duration: '500ms',
    timing: 'ease-out',
  },

  // Value counter animation
  valueCounter: {
    duration: 300,
    easing: 'easeOut',
    // Use framer-motion or react-spring
  },

  // Card slide-in for new recommendations
  slideIn: {
    initial: { opacity: 0, x: 20 },
    animate: { opacity: 1, x: 0 },
    transition: { duration: 0.3 },
  },
};
```

---

## 9. Performance Optimization

### 9.1 Component Optimization

```typescript
// Memoization patterns
const MemoizedMetricCard = memo(MetricCard, (prevProps, nextProps) => {
  return prevProps.value === nextProps.value &&
         prevProps.change === nextProps.change;
});

// Virtualization for large lists
const VirtualizedHoldingsTable = () => (
  <VirtualizedTable
    rowCount={positions.length}
    rowHeight={52}
    overscanCount={5}
    rowRenderer={({ index, style }) => (
      <HoldingRow position={positions[index]} style={style} />
    )}
  />
);

// Lazy loading for charts
const PerformanceChart = lazy(() => import('./PerformanceChart'));

// Debounced search
const debouncedSearch = useMemo(
  () => debounce((query) => dispatch(searchStocks(query)), 300),
  [dispatch]
);
```

### 9.2 Data Fetching Strategy

```typescript
const dataFetchingConfig = {
  // Initial load priorities
  priority: [
    'portfolioSummary',      // Critical - load first
    'recommendations',        // High - visible on load
    'positions',              // High - main table
    'performanceHistory',     // Medium - chart data
    'marketOverview',         // Low - below fold
    'news',                   // Low - below fold
  ],

  // Caching strategy
  cache: {
    portfolioSummary: { ttl: 60000, staleWhileRevalidate: true },
    recommendations: { ttl: 300000, staleWhileRevalidate: true },
    positions: { ttl: 60000, staleWhileRevalidate: true },
    performanceHistory: { ttl: 300000, staleWhileRevalidate: false },
    news: { ttl: 600000, staleWhileRevalidate: true },
  },

  // Prefetching
  prefetch: {
    onHover: ['stockDetails'],
    onVisible: ['marketOverview', 'news'],
  },
};
```

---

## 10. Implementation Checklist

### Phase 1: Core Layout & Metrics
- [ ] DashboardLayout component with responsive grid
- [ ] MetricCard component (4 variants)
- [ ] Loading skeletons for all cards
- [ ] Error boundary with retry

### Phase 2: Charts & Visualizations
- [ ] PerformanceChart with period selection
- [ ] AllocationChart (donut)
- [ ] Sparklines for holdings
- [ ] Volume overlay option

### Phase 3: Data Tables
- [ ] HoldingsTable with sorting
- [ ] Table virtualization for large lists
- [ ] Quick action buttons
- [ ] Responsive column hiding

### Phase 4: Recommendations & News
- [ ] RecommendationCardCompact
- [ ] RecommendationsPanel with scroll
- [ ] NewsFeed component
- [ ] Sentiment indicators

### Phase 5: Real-Time & Polish
- [ ] WebSocket integration
- [ ] Price update animations
- [ ] Connection status indicator
- [ ] Offline fallback

### Phase 6: Accessibility & Testing
- [ ] WCAG audit (axe-core)
- [ ] Keyboard navigation testing
- [ ] Screen reader testing
- [ ] Color contrast verification

---

## Appendix A: MUI Theme Extension

```typescript
// Add to theme/index.ts
declare module '@mui/material/styles' {
  interface Palette {
    financial: {
      gain: PaletteColor;
      loss: PaletteColor;
      neutral: PaletteColor;
    };
    actions: {
      buy: string;
      sell: string;
      hold: string;
    };
    risk: {
      low: string;
      medium: string;
      high: string;
    };
  }

  interface PaletteOptions {
    financial?: {
      gain?: PaletteColorOptions;
      loss?: PaletteColorOptions;
      neutral?: PaletteColorOptions;
    };
    actions?: {
      buy?: string;
      sell?: string;
      hold?: string;
    };
    risk?: {
      low?: string;
      medium?: string;
      high?: string;
    };
  }
}
```

---

## Appendix B: Component File Structure

```
frontend/web/src/
├── components/
│   ├── dashboard/
│   │   ├── DashboardLayout.tsx
│   │   ├── MetricCard.tsx
│   │   ├── MetricCardSkeleton.tsx
│   │   ├── PerformanceSection.tsx
│   │   └── index.ts
│   │
│   ├── charts/
│   │   ├── PerformanceChart.tsx
│   │   ├── AllocationChart.tsx
│   │   ├── Sparkline.tsx
│   │   ├── RiskGauge.tsx
│   │   └── index.ts
│   │
│   ├── tables/
│   │   ├── HoldingsTable.tsx
│   │   ├── HoldingRow.tsx
│   │   ├── VirtualizedTable.tsx
│   │   └── index.ts
│   │
│   ├── panels/
│   │   ├── RecommendationsPanel.tsx
│   │   ├── AllocationPanel.tsx
│   │   ├── MarketOverviewPanel.tsx
│   │   ├── NewsFeedPanel.tsx
│   │   └── index.ts
│   │
│   └── common/
│       ├── PriceChange.tsx
│       ├── ConfidenceBadge.tsx
│       ├── RiskBadge.tsx
│       ├── ActionBadge.tsx
│       └── index.ts
│
├── hooks/
│   ├── usePortfolioData.ts
│   ├── useRealTimePrices.ts
│   ├── useChartData.ts
│   └── useResponsiveLayout.ts
│
└── utils/
    ├── formatters.ts
    ├── chartHelpers.ts
    └── accessibilityHelpers.ts
```
