import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { apiService } from '../../services/api.service';

interface Recommendation {
  ticker: string;
  company_name: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  target_price: number;
  current_price: number;
  potential_return: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  reasoning: string;
  technical_score: number;
  fundamental_score: number;
  sentiment_score: number;
  ml_prediction: number;
  time_horizon: string;
  sector: string;
  market_cap: number;
  volume: number;
  pe_ratio: number;
  dividend_yield: number;
  created_at: string;
}

interface RecommendationsState {
  recommendations: Recommendation[];
  filteredRecommendations: Recommendation[];
  selectedRecommendation: Recommendation | null;
  filters: {
    action: string | null;
    riskLevel: string | null;
    sector: string | null;
    minConfidence: number;
    minReturn: number;
  };
  sortBy: 'confidence' | 'potential_return' | 'created_at';
  sortOrder: 'asc' | 'desc';
  pagination: {
    page: number;
    limit: number;
    total: number;
  };
  loading: boolean;
  error: string | null;
}

const initialState: RecommendationsState = {
  recommendations: [],
  filteredRecommendations: [],
  selectedRecommendation: null,
  filters: {
    action: null,
    riskLevel: null,
    sector: null,
    minConfidence: 0,
    minReturn: 0,
  },
  sortBy: 'confidence',
  sortOrder: 'desc',
  pagination: {
    page: 1,
    limit: 20,
    total: 0,
  },
  loading: false,
  error: null,
};

interface FetchRecommendationsParams {
  page?: number;
  limit?: number;
  action?: string;
  riskLevel?: string;
  sector?: string;
  minConfidence?: number;
  minReturn?: number;
  sortBy?: string;
  sortOrder?: string;
}

export const fetchRecommendations = createAsyncThunk(
  'recommendations/fetchRecommendations',
  async (params: FetchRecommendationsParams = {}) => {
    const queryParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        queryParams.append(key, value.toString());
      }
    });
    const response = await apiService.get(`/recommendations?${queryParams.toString()}`);
    return response.data;
  }
);

export const fetchRecommendationByTicker = createAsyncThunk(
  'recommendations/fetchByTicker',
  async (ticker: string) => {
    const response = await apiService.get(`/recommendations/${ticker}`);
    return response.data;
  }
);

export const generateRecommendation = createAsyncThunk(
  'recommendations/generate',
  async (ticker: string) => {
    const response = await apiService.post(`/recommendations/generate`, { ticker });
    return response.data;
  }
);

const applyFilters = (state: RecommendationsState) => {
  let filtered = [...state.recommendations];

  // Apply filters
  if (state.filters.action) {
    filtered = filtered.filter(r => r.action === state.filters.action);
  }
  if (state.filters.riskLevel) {
    filtered = filtered.filter(r => r.risk_level === state.filters.riskLevel);
  }
  if (state.filters.sector) {
    filtered = filtered.filter(r => r.sector === state.filters.sector);
  }
  if (state.filters.minConfidence > 0) {
    filtered = filtered.filter(r => r.confidence >= state.filters.minConfidence);
  }
  if (state.filters.minReturn !== 0) {
    filtered = filtered.filter(r => r.potential_return >= state.filters.minReturn);
  }

  // Apply sorting
  filtered.sort((a, b) => {
    let aValue: any = a[state.sortBy];
    let bValue: any = b[state.sortBy];

    if (state.sortBy === 'created_at') {
      aValue = new Date(aValue).getTime();
      bValue = new Date(bValue).getTime();
    }

    if (state.sortOrder === 'asc') {
      return aValue > bValue ? 1 : -1;
    } else {
      return aValue < bValue ? 1 : -1;
    }
  });

  state.filteredRecommendations = filtered;
};

const recommendationsSlice = createSlice({
  name: 'recommendations',
  initialState,
  reducers: {
    setFilters: (state, action: PayloadAction<Partial<RecommendationsState['filters']>>) => {
      state.filters = { ...state.filters, ...action.payload };
      applyFilters(state);
    },
    setSorting: (state, action: PayloadAction<{ sortBy?: string; sortOrder?: string }>) => {
      if (action.payload.sortBy) {
        state.sortBy = action.payload.sortBy as any;
      }
      if (action.payload.sortOrder) {
        state.sortOrder = action.payload.sortOrder as 'asc' | 'desc';
      }
      applyFilters(state);
    },
    setPage: (state, action: PayloadAction<number>) => {
      state.pagination.page = action.payload;
    },
    selectRecommendation: (state, action: PayloadAction<string>) => {
      state.selectedRecommendation = state.recommendations.find(
        r => r.ticker === action.payload
      ) || null;
    },
    clearSelectedRecommendation: (state) => {
      state.selectedRecommendation = null;
    },
    updateRecommendation: (state, action: PayloadAction<Recommendation>) => {
      const index = state.recommendations.findIndex(
        r => r.ticker === action.payload.ticker
      );
      if (index !== -1) {
        state.recommendations[index] = action.payload;
        applyFilters(state);
      }
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch Recommendations
      .addCase(fetchRecommendations.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchRecommendations.fulfilled, (state, action) => {
        state.loading = false;
        state.recommendations = action.payload.recommendations || action.payload;
        state.pagination.total = action.payload.total || state.recommendations.length;
        applyFilters(state);
      })
      .addCase(fetchRecommendations.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch recommendations';
      })
      // Fetch Recommendation by Ticker
      .addCase(fetchRecommendationByTicker.fulfilled, (state, action) => {
        state.selectedRecommendation = action.payload;
      })
      // Generate Recommendation
      .addCase(generateRecommendation.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(generateRecommendation.fulfilled, (state, action) => {
        state.loading = false;
        const newRecommendation = action.payload;
        const existingIndex = state.recommendations.findIndex(
          r => r.ticker === newRecommendation.ticker
        );
        if (existingIndex !== -1) {
          state.recommendations[existingIndex] = newRecommendation;
        } else {
          state.recommendations.unshift(newRecommendation);
        }
        applyFilters(state);
      })
      .addCase(generateRecommendation.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to generate recommendation';
      });
  },
});

export const {
  setFilters,
  setSorting,
  setPage,
  selectRecommendation,
  clearSelectedRecommendation,
  updateRecommendation,
  clearError,
} = recommendationsSlice.actions;

export default recommendationsSlice.reducer;