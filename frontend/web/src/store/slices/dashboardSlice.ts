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

interface DashboardState {
  marketOverview: {
    indices: MarketIndex[];
    heatmap: any[];
    sectors: Sector[];
  } | null;
  topRecommendations: any[];
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
    topGainers: any[];
    topLosers: any[];
    allocation: any[];
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
  recentNews: any[];
  marketSentiment: {
    overall: string;
    score: number;
    breakdown: {
      positive: number;
      neutral: number;
      negative: number;
    };
  } | null;
  costMetrics: {
    currentMonthCost: number;
    projectedMonthCost: number;
    dailyAverage: number;
    monthlyBudget: number;
    apiUsage: any[];
    costBreakdown: any[];
    alerts: any[];
    lastUpdated: string;
    costTrend: any[];
    savingsMode: boolean;
    emergencyMode: boolean;
  } | null;
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
    updateMarketSentiment: (state, action: PayloadAction<any>) => {
      state.marketSentiment = action.payload;
    },
    updateCostMetrics: (state, action: PayloadAction<any>) => {
      state.costMetrics = action.payload;
    },
    addNews: (state, action: PayloadAction<any>) => {
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