import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { apiService } from '../../services/api.service';

export interface MarketIndex {
  symbol: string;
  name: string;
  value: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  previousClose: number;
  timestamp: string;
}

export interface MarketMover {
  ticker: string;
  companyName: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  sector: string;
}

export interface SectorPerformance {
  sector: string;
  changePercent: number;
  marketCap: number;
  volume: number;
  gainers: number;
  losers: number;
  topStock: {
    ticker: string;
    changePercent: number;
  };
}

export interface MarketNews {
  id: string;
  title: string;
  summary: string;
  url: string;
  source: string;
  publishedAt: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  relatedTickers: string[];
  image?: string;
}

export interface MarketBreadth {
  advancers: number;
  decliners: number;
  unchanged: number;
  newHighs: number;
  newLows: number;
  advanceDeclineRatio: number;
  upVolume: number;
  downVolume: number;
  totalVolume: number;
}

interface MarketState {
  indices: MarketIndex[];
  topGainers: MarketMover[];
  topLosers: MarketMover[];
  mostActive: MarketMover[];
  sectorPerformance: SectorPerformance[];
  marketNews: MarketNews[];
  marketBreadth: MarketBreadth | null;
  heatmapData: Array<{
    ticker: string;
    name: string;
    sector: string;
    changePercent: number;
    marketCap: number;
    volume: number;
  }>;
  economicCalendar: Array<{
    date: string;
    time: string;
    event: string;
    importance: 'high' | 'medium' | 'low';
    actual?: number;
    forecast?: number;
    previous?: number;
  }>;
  isLoading: boolean;
  error: string | null;
  lastUpdated: string | null;
}

const initialState: MarketState = {
  indices: [],
  topGainers: [],
  topLosers: [],
  mostActive: [],
  sectorPerformance: [],
  marketNews: [],
  marketBreadth: null,
  heatmapData: [],
  economicCalendar: [],
  isLoading: false,
  error: null,
  lastUpdated: null,
};

// Async thunks
export const fetchMarketOverview = createAsyncThunk(
  'market/fetchOverview',
  async () => {
    const response = await apiService.get('/market/overview');
    return response.data;
  }
);

export const fetchMarketIndices = createAsyncThunk(
  'market/fetchIndices',
  async () => {
    const response = await apiService.get('/market/indices');
    return response.data;
  }
);

export const fetchMarketMovers = createAsyncThunk(
  'market/fetchMovers',
  async () => {
    const response = await apiService.get('/market/movers');
    return response.data;
  }
);

export const fetchSectorPerformance = createAsyncThunk(
  'market/fetchSectors',
  async () => {
    const response = await apiService.get('/market/sectors');
    return response.data;
  }
);

export const fetchMarketNews = createAsyncThunk(
  'market/fetchNews',
  async (params?: { limit?: number; category?: string }) => {
    const response = await apiService.get('/market/news', { params });
    return response.data;
  }
);

export const fetchHeatmapData = createAsyncThunk(
  'market/fetchHeatmap',
  async (params?: { index?: string; sector?: string }) => {
    const response = await apiService.get('/market/heatmap', { params });
    return response.data;
  }
);

export const fetchEconomicCalendar = createAsyncThunk(
  'market/fetchCalendar',
  async () => {
    const response = await apiService.get('/market/calendar');
    return response.data;
  }
);

const marketSlice = createSlice({
  name: 'market',
  initialState,
  reducers: {
    updateMarketIndex: (state, action: PayloadAction<MarketIndex>) => {
      const index = state.indices.findIndex(i => i.symbol === action.payload.symbol);
      if (index !== -1) {
        state.indices[index] = action.payload;
      } else {
        state.indices.push(action.payload);
      }
    },
    updateMarketBreadth: (state, action: PayloadAction<MarketBreadth>) => {
      state.marketBreadth = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch market overview
      .addCase(fetchMarketOverview.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchMarketOverview.fulfilled, (state, action) => {
        state.isLoading = false;
        state.indices = action.payload.indices;
        state.topGainers = action.payload.topGainers;
        state.topLosers = action.payload.topLosers;
        state.mostActive = action.payload.mostActive;
        state.marketBreadth = action.payload.marketBreadth;
        state.lastUpdated = new Date().toISOString();
      })
      .addCase(fetchMarketOverview.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Failed to fetch market overview';
      })
      // Fetch indices
      .addCase(fetchMarketIndices.fulfilled, (state, action) => {
        state.indices = action.payload;
      })
      // Fetch movers
      .addCase(fetchMarketMovers.fulfilled, (state, action) => {
        state.topGainers = action.payload.gainers;
        state.topLosers = action.payload.losers;
        state.mostActive = action.payload.active;
      })
      // Fetch sectors
      .addCase(fetchSectorPerformance.fulfilled, (state, action) => {
        state.sectorPerformance = action.payload;
      })
      // Fetch news
      .addCase(fetchMarketNews.fulfilled, (state, action) => {
        state.marketNews = action.payload;
      })
      // Fetch heatmap
      .addCase(fetchHeatmapData.fulfilled, (state, action) => {
        state.heatmapData = action.payload;
      })
      // Fetch calendar
      .addCase(fetchEconomicCalendar.fulfilled, (state, action) => {
        state.economicCalendar = action.payload;
      });
  },
});

export const { updateMarketIndex, updateMarketBreadth, clearError } = marketSlice.actions;
export default marketSlice.reducer;