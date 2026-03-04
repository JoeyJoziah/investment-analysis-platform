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

interface DashboardState {
  marketOverview: {
    indices: MarketIndex[];
    heatmap: HeatmapEntry[];
    sectors: Sector[];
  } | null;
  topRecommendations: Recommendation[];
  portfolioSummary: {
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
  } | null;
  recentNews: NewsItem[];
  marketSentiment: MarketSentimentState | null;
  costMetrics: CostMetricsState | null;
  loading: boolean;
  error: string | null;
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

export const fetchDashboardData = createAsyncThunk(
  'dashboard/fetchData',
  async () => {
    const response = await apiService.get('/dashboard');
    return response.data;
  }
);

export const fetchMarketOverview = createAsyncThunk(
  'dashboard/fetchMarketOverview',
  async () => {
    const response = await apiService.get('/market/overview');
    return response.data;
  }
);

export const fetchPortfolioSummary = createAsyncThunk(
  'dashboard/fetchPortfolioSummary',
  async () => {
    const response = await apiService.get('/portfolio/summary');
    return response.data;
  }
);

export const fetchCostMetrics = createAsyncThunk(
  'dashboard/fetchCostMetrics',
  async () => {
    const response = await apiService.get('/admin/cost-metrics');
    return response.data;
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