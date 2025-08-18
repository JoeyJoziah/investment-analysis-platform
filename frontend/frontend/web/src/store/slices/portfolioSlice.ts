import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { apiService } from '../../services/api.service';

export interface Position {
  id: string;
  ticker: string;
  companyName: string;
  quantity: number;
  averagePrice: number;
  currentPrice: number;
  marketValue: number;
  totalGain: number;
  totalGainPercent: number;
  dayGain: number;
  dayGainPercent: number;
  sector: string;
  lastUpdated: string;
}

export interface Transaction {
  id: string;
  ticker: string;
  type: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  totalAmount: number;
  date: string;
  notes?: string;
}

export interface PortfolioMetrics {
  totalValue: number;
  totalCost: number;
  totalGain: number;
  totalGainPercent: number;
  dayGain: number;
  dayGainPercent: number;
  cashBalance: number;
  buyingPower: number;
  diversification: {
    sector: Record<string, number>;
    asset: Record<string, number>;
  };
  performance: {
    daily: Array<{ date: string; value: number }>;
    monthly: Array<{ date: string; value: number }>;
    yearly: Array<{ date: string; value: number }>;
  };
  riskMetrics: {
    sharpeRatio: number;
    beta: number;
    alpha: number;
    standardDeviation: number;
    maxDrawdown: number;
  };
}

interface PortfolioState {
  positions: Position[];
  transactions: Transaction[];
  metrics: PortfolioMetrics | null;
  watchlist: string[];
  isLoading: boolean;
  error: string | null;
  lastUpdated: string | null;
}

const initialState: PortfolioState = {
  positions: [],
  transactions: [],
  metrics: null,
  watchlist: [],
  isLoading: false,
  error: null,
  lastUpdated: null,
};

// Async thunks
export const fetchPortfolio = createAsyncThunk(
  'portfolio/fetchPortfolio',
  async () => {
    const response = await apiService.get('/portfolio');
    return response.data;
  }
);

export const fetchTransactions = createAsyncThunk(
  'portfolio/fetchTransactions',
  async (params?: { limit?: number; offset?: number }) => {
    const response = await apiService.get('/portfolio/transactions', { params });
    return response.data;
  }
);

export const addTransaction = createAsyncThunk(
  'portfolio/addTransaction',
  async (transaction: Omit<Transaction, 'id'>) => {
    const response = await apiService.post('/portfolio/transactions', transaction);
    return response.data;
  }
);

export const deletePosition = createAsyncThunk(
  'portfolio/deletePosition',
  async (positionId: string) => {
    await apiService.delete(`/portfolio/positions/${positionId}`);
    return positionId;
  }
);

export const fetchWatchlist = createAsyncThunk(
  'portfolio/fetchWatchlist',
  async () => {
    const response = await apiService.get('/portfolio/watchlist');
    return response.data;
  }
);

export const addToWatchlist = createAsyncThunk(
  'portfolio/addToWatchlist',
  async (ticker: string) => {
    const response = await apiService.post('/portfolio/watchlist', { ticker });
    return response.data;
  }
);

export const removeFromWatchlist = createAsyncThunk(
  'portfolio/removeFromWatchlist',
  async (ticker: string) => {
    await apiService.delete(`/portfolio/watchlist/${ticker}`);
    return ticker;
  }
);

const portfolioSlice = createSlice({
  name: 'portfolio',
  initialState,
  reducers: {
    updatePosition: (state, action: PayloadAction<Position>) => {
      const index = state.positions.findIndex(p => p.id === action.payload.id);
      if (index !== -1) {
        state.positions[index] = action.payload;
      }
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch portfolio
      .addCase(fetchPortfolio.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchPortfolio.fulfilled, (state, action) => {
        state.isLoading = false;
        state.positions = action.payload.positions;
        state.metrics = action.payload.metrics;
        state.lastUpdated = new Date().toISOString();
      })
      .addCase(fetchPortfolio.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Failed to fetch portfolio';
      })
      // Fetch transactions
      .addCase(fetchTransactions.fulfilled, (state, action) => {
        state.transactions = action.payload;
      })
      // Add transaction
      .addCase(addTransaction.fulfilled, (state, action) => {
        state.transactions.unshift(action.payload);
      })
      // Delete position
      .addCase(deletePosition.fulfilled, (state, action) => {
        state.positions = state.positions.filter(p => p.id !== action.payload);
      })
      // Watchlist
      .addCase(fetchWatchlist.fulfilled, (state, action) => {
        state.watchlist = action.payload;
      })
      .addCase(addToWatchlist.fulfilled, (state, action) => {
        if (!state.watchlist.includes(action.payload)) {
          state.watchlist.push(action.payload);
        }
      })
      .addCase(removeFromWatchlist.fulfilled, (state, action) => {
        state.watchlist = state.watchlist.filter(ticker => ticker !== action.payload);
      });
  },
});

export const { updatePosition, clearError } = portfolioSlice.actions;
export default portfolioSlice.reducer;