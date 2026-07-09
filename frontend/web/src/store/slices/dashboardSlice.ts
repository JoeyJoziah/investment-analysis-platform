import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { apiService } from '../../services/api.service';

interface MarketIndex {
  symbol: string;
  value: number;
  change: number;
  changePercent: number;
}

interface Sector {
  name: string;
  change: number;
  volume: number;
}

interface HeatmapEntry {
  symbol: string;
  sector: string;
  change: number;
  changePercent: number;
  marketCap: number;
}

interface Recommendation {
  id: string;
  ticker: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  targetPrice: number;
  currentPrice: number;
  rationale: string;
  createdAt: string;
}

interface PositionSummary {
  ticker: string;
  companyName: string;
  change: number;
  changePercent: number;
  currentPrice: number;
  marketValue: number;
}

interface AllocationEntry {
  label: string;
  value: number;
  percent: number;
}

interface NewsItem {
  id: string;
  title: string;
  summary: string;
  source: string;
  url: string;
  publishedAt: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  tickers: string[];
}

interface ApiUsageEntry {
  provider: string;
  calls: number;
  cost: number;
  date: string;
}

interface CostBreakdownEntry {
  category: string;
  amount: number;
  percent: number;
}

interface CostAlert {
  id: string;
  message: string;
  severity: 'info' | 'warning' | 'critical';
  triggeredAt: string;
}

interface CostTrendEntry {
  date: string;
  cost: number;
  projected: boolean;
}

interface MarketSentimentState {
  overall: string;
  score: number;
  breakdown: {
    positive: number;
    neutral: number;
    negative: number;
  };
}

interface CostMetricsState {
  currentMonthCost: number;
  projectedMonthCost: number;
  dailyAverage: number;
  monthlyBudget: number;
  apiUsage: ApiUsageEntry[];
  costBreakdown: CostBreakdownEntry[];
  alerts: CostAlert[];
  lastUpdated: string;
  costTrend: CostTrendEntry[];
  savingsMode: boolean;
  emergencyMode: boolean;
}

interface DashboardMarketOverview {
  indices: MarketIndex[];
  heatmap: HeatmapEntry[];
  sectors: Sector[];
}

interface DashboardPortfolioSummary {
  totalValue: number;
  totalCost: number;
  totalReturn: number;
  totalReturnPercent: number;
  dayChange: number;
  dayChangePercent: number;
  weekChange: number;
  monthChange: number;
  yearChange: number;
  activePositions: number;
  performanceHistory: Array<{ date: string; value: number }>;
  topGainers: PositionSummary[];
  topLosers: PositionSummary[];
  allocation: AllocationEntry[];
  riskMetrics: {
    sharpeRatio: number;
    beta: number;
    standardDeviation: number;
    maxDrawdown: number;
  };
  diversificationScore: number;
  cashBalance: number;
  marginUsed: number;
}

interface DashboardState {
  marketOverview: DashboardMarketOverview | null;
  topRecommendations: Recommendation[];
  portfolioSummary: DashboardPortfolioSummary | null;
  recentNews: NewsItem[];
  marketSentiment: MarketSentimentState | null;
  costMetrics: CostMetricsState | null;
  loading: boolean;
  error: string | null;
}

// Resolved payload for the aggregate dashboard endpoint. Reuses the nested
// interfaces above so the fulfilled reducer sees concrete field types.
interface DashboardDataPayload {
  marketOverview: DashboardMarketOverview | null;
  topRecommendations: Recommendation[];
  portfolioSummary: DashboardPortfolioSummary | null;
  recentNews: NewsItem[];
  marketSentiment: MarketSentimentState | null;
  costMetrics: CostMetricsState | null;
}

const initialState: DashboardState = {
  marketOverview: null,
  topRecommendations: [],
  portfolioSummary: null,
  recentNews: [],
  marketSentiment: null,
  costMetrics: null,
  loading: false,
  error: null,
};

// The backend wraps successful responses in an ApiResponse envelope
// ({ success, data: <payload> }). Axios exposes the body on response.data, so
// the real payload lives at response.data.data. Unwrap defensively so reducers
// receive the actual dashboard payload (e.g. .marketOverview, .portfolioSummary)
// rather than the envelope — otherwise those fields are undefined and the
// dashboard renders empty.
const unwrapData = <T = unknown>(body: unknown): T => {
  if (body && typeof body === 'object' && 'data' in body) {
    return (body as { data: T }).data;
  }
  return body as T;
};

export const fetchDashboardData = createAsyncThunk<DashboardDataPayload>(
  'dashboard/fetchData',
  async () => {
    const response = await apiService.get('/api/v1/dashboard');
    return unwrapData<DashboardDataPayload>(response.data);
  }
);

export const fetchMarketOverview = createAsyncThunk<DashboardMarketOverview>(
  'dashboard/fetchMarketOverview',
  async () => {
    const response = await apiService.get('/api/v1/market/overview');
    return unwrapData<DashboardMarketOverview>(response.data);
  }
);

export const fetchPortfolioSummary = createAsyncThunk<DashboardPortfolioSummary>(
  'dashboard/fetchPortfolioSummary',
  async () => {
    const response = await apiService.get('/api/v1/portfolio/summary');
    return unwrapData<DashboardPortfolioSummary>(response.data);
  }
);

export const fetchCostMetrics = createAsyncThunk<CostMetricsState>(
  'dashboard/fetchCostMetrics',
  async () => {
    const response = await apiService.get('/api/v1/admin/metrics');
    return unwrapData<CostMetricsState>(response.data);
  }
);

const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    updateMarketSentiment: (state, action: PayloadAction<MarketSentimentState>) => {
      state.marketSentiment = action.payload;
    },
    updateCostMetrics: (state, action: PayloadAction<CostMetricsState>) => {
      state.costMetrics = action.payload;
    },
    addNews: (state, action: PayloadAction<NewsItem>) => {
      state.recentNews.unshift(action.payload);
      if (state.recentNews.length > 20) {
        state.recentNews.pop();
      }
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch Dashboard Data
      .addCase(fetchDashboardData.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchDashboardData.fulfilled, (state, action) => {
        state.loading = false;
        state.marketOverview = action.payload.marketOverview;
        state.topRecommendations = action.payload.topRecommendations;
        state.portfolioSummary = action.payload.portfolioSummary;
        state.recentNews = action.payload.recentNews;
        state.marketSentiment = action.payload.marketSentiment;
        state.costMetrics = action.payload.costMetrics;
      })
      .addCase(fetchDashboardData.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch dashboard data';
      })
      // Fetch Market Overview
      .addCase(fetchMarketOverview.fulfilled, (state, action) => {
        state.marketOverview = action.payload;
      })
      // Fetch Portfolio Summary
      .addCase(fetchPortfolioSummary.fulfilled, (state, action) => {
        state.portfolioSummary = action.payload;
      })
      // Fetch Cost Metrics
      .addCase(fetchCostMetrics.fulfilled, (state, action) => {
        state.costMetrics = action.payload;
      });
  },
});

export const { updateMarketSentiment, updateCostMetrics, addNews, clearError } = dashboardSlice.actions;
export default dashboardSlice.reducer;