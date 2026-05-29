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

// Watchlist types matching the new API
export interface WatchlistItem {
  id: number;
  watchlist_id: number;
  stock_id: number;
  added_at: string;
  target_price: number | null;
  notes: string | null;
  alert_enabled: boolean;
  symbol: string;
  company_name: string;
  current_price: number | null;
  price_change: number | null;
  price_change_percent: number | null;
}

export interface Watchlist {
  id: number;
  name: string;
  description: string | null;
  is_public: boolean;
  user_id: number;
  created_at: string;
  updated_at: string;
  items: WatchlistItem[];
  item_count: number;
}

interface PortfolioState {
  positions: Position[];
  transactions: Transaction[];
  metrics: PortfolioMetrics | null;
  watchlist: Watchlist | null;
  watchlistLoading: boolean;
  watchlistError: string | null;
  isLoading: boolean;
  error: string | null;
  lastUpdated: string | null;
}

const initialState: PortfolioState = {
  positions: [],
  transactions: [],
  metrics: null,
  watchlist: null,
  watchlistLoading: false,
  watchlistError: null,
  isLoading: false,
  error: null,
  lastUpdated: null,
};

// The backend wraps successful watchlist responses in an ApiResponse envelope
// ({ success, data: <payload> }). Axios exposes the body on response.data, so
// the real payload lives at response.data.data. Unwrap defensively so reducers
// receive the actual Watchlist/WatchlistItem (not the envelope) — otherwise
// state.watchlist.items is undefined and the list silently renders empty.
const unwrapData = <T = unknown>(body: unknown): T => {
  if (body && typeof body === 'object' && 'data' in body) {
    return (body as { data: T }).data;
  }
  return body as T;
};

// Resolved payload for the portfolio endpoint, consumed by the fulfilled reducer.
export interface PortfolioPayload {
  positions: Position[];
  metrics: PortfolioMetrics;
}

// Async thunks
export const fetchPortfolio = createAsyncThunk<PortfolioPayload>(
  'portfolio/fetchPortfolio',
  async () => {
    const response = await apiService.get<PortfolioPayload>('/api/v1/portfolio');
    return response.data;
  }
);

export const fetchTransactions = createAsyncThunk<
  Transaction[],
  { limit?: number; offset?: number } | undefined
>(
  'portfolio/fetchTransactions',
  async (params?: { limit?: number; offset?: number }) => {
    const response = await apiService.get<Transaction[]>('/api/v1/portfolio/transactions', {
      params,
    });
    return response.data;
  }
);

export const addTransaction = createAsyncThunk<Transaction, Omit<Transaction, 'id'>>(
  'portfolio/addTransaction',
  async (transaction: Omit<Transaction, 'id'>) => {
    const response = await apiService.post<Transaction>(
      '/api/v1/portfolio/transactions',
      transaction
    );
    return response.data;
  }
);

export const deletePosition = createAsyncThunk(
  'portfolio/deletePosition',
  async (positionId: string) => {
    await apiService.delete(`/api/v1/portfolio/positions/${positionId}`);
    return positionId;
  }
);

// Watchlist async thunks - using the new Watchlist API
export const fetchWatchlist = createAsyncThunk(
  'portfolio/fetchWatchlist',
  async (_, { rejectWithValue }) => {
    try {
      const response = await apiService.get('/api/v1/watchlists/default');
      return unwrapData<Watchlist>(response.data);
    } catch (error: unknown) {
      const axiosError = error as { response?: { data?: { detail?: string } } };
      return rejectWithValue(axiosError.response?.data?.detail ?? 'Failed to fetch watchlist');
    }
  }
);

export const addToWatchlist = createAsyncThunk(
  'portfolio/addToWatchlist',
  async (
    { symbol, targetPrice, notes }: { symbol: string; targetPrice?: number; notes?: string },
    { rejectWithValue }
  ) => {
    try {
      const response = await apiService.post(`/api/v1/watchlists/default/symbols/${symbol}`, {
        target_price: targetPrice,
        notes,
      });
      return unwrapData<WatchlistItem>(response.data);
    } catch (error: unknown) {
      const axiosError = error as { response?: { data?: { detail?: string } } };
      return rejectWithValue(axiosError.response?.data?.detail ?? 'Failed to add to watchlist');
    }
  }
);

export const removeFromWatchlist = createAsyncThunk(
  'portfolio/removeFromWatchlist',
  async (symbol: string, { rejectWithValue }) => {
    try {
      await apiService.delete(`/api/v1/watchlists/default/symbols/${symbol}`);
      return symbol;
    } catch (error: unknown) {
      const axiosError = error as { response?: { data?: { detail?: string } } };
      return rejectWithValue(axiosError.response?.data?.detail ?? 'Failed to remove from watchlist');
    }
  }
);

export const updateWatchlistItem = createAsyncThunk(
  'portfolio/updateWatchlistItem',
  async (
    {
      watchlistId,
      itemId,
      updates,
    }: {
      watchlistId: number;
      itemId: number;
      updates: { target_price?: number | null; notes?: string | null; alert_enabled?: boolean };
    },
    { rejectWithValue }
  ) => {
    try {
      const response = await apiService.put(
        `/api/v1/watchlists/${watchlistId}/items/${itemId}`,
        updates
      );
      return unwrapData<WatchlistItem>(response.data);
    } catch (error: unknown) {
      const axiosError = error as { response?: { data?: { detail?: string } } };
      return rejectWithValue(axiosError.response?.data?.detail ?? 'Failed to update watchlist item');
    }
  }
);

// Fetch all user watchlists
export const fetchAllWatchlists = createAsyncThunk(
  'portfolio/fetchAllWatchlists',
  async (_, { rejectWithValue }) => {
    try {
      const response = await apiService.get('/api/v1/watchlists');
      return unwrapData(response.data);
    } catch (error: unknown) {
      const axiosError = error as { response?: { data?: { detail?: string } } };
      return rejectWithValue(axiosError.response?.data?.detail ?? 'Failed to fetch watchlists');
    }
  }
);

// Create a new watchlist
export const createWatchlist = createAsyncThunk(
  'portfolio/createWatchlist',
  async (
    { name, description, isPublic }: { name: string; description?: string; isPublic?: boolean },
    { rejectWithValue }
  ) => {
    try {
      const response = await apiService.post('/api/v1/watchlists', {
        name,
        description,
        is_public: isPublic,
      });
      return unwrapData<Watchlist>(response.data);
    } catch (error: unknown) {
      const axiosError = error as { response?: { data?: { detail?: string } } };
      return rejectWithValue(axiosError.response?.data?.detail ?? 'Failed to create watchlist');
    }
  }
);

// Delete a watchlist
export const deleteWatchlist = createAsyncThunk(
  'portfolio/deleteWatchlist',
  async (watchlistId: number, { rejectWithValue }) => {
    try {
      await apiService.delete(`/api/v1/watchlists/${watchlistId}`);
      return watchlistId;
    } catch (error: unknown) {
      const axiosError = error as { response?: { data?: { detail?: string } } };
      return rejectWithValue(axiosError.response?.data?.detail ?? 'Failed to delete watchlist');
    }
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
    clearWatchlistError: (state) => {
      state.watchlistError = null;
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
      // Watchlist - fetch
      .addCase(fetchWatchlist.pending, (state) => {
        state.watchlistLoading = true;
        state.watchlistError = null;
      })
      .addCase(fetchWatchlist.fulfilled, (state, action) => {
        state.watchlistLoading = false;
        state.watchlist = action.payload;
      })
      .addCase(fetchWatchlist.rejected, (state, action) => {
        state.watchlistLoading = false;
        state.watchlistError = action.payload as string;
      })
      // Watchlist - add item
      .addCase(addToWatchlist.pending, (state) => {
        state.watchlistLoading = true;
        state.watchlistError = null;
      })
      .addCase(addToWatchlist.fulfilled, (state, action) => {
        state.watchlistLoading = false;
        // Add the new item to the watchlist
        if (state.watchlist) {
          state.watchlist.items.push(action.payload);
          state.watchlist.item_count = state.watchlist.items.length;
        }
      })
      .addCase(addToWatchlist.rejected, (state, action) => {
        state.watchlistLoading = false;
        state.watchlistError = action.payload as string;
      })
      // Watchlist - remove item
      .addCase(removeFromWatchlist.pending, (state) => {
        state.watchlistLoading = true;
        state.watchlistError = null;
      })
      .addCase(removeFromWatchlist.fulfilled, (state, action) => {
        state.watchlistLoading = false;
        if (state.watchlist) {
          state.watchlist.items = state.watchlist.items.filter(
            (item) => item.symbol !== action.payload
          );
          state.watchlist.item_count = state.watchlist.items.length;
        }
      })
      .addCase(removeFromWatchlist.rejected, (state, action) => {
        state.watchlistLoading = false;
        state.watchlistError = action.payload as string;
      })
      // Watchlist - update item
      .addCase(updateWatchlistItem.pending, (state) => {
        state.watchlistLoading = true;
        state.watchlistError = null;
      })
      .addCase(updateWatchlistItem.fulfilled, (state, action) => {
        state.watchlistLoading = false;
        if (state.watchlist) {
          const index = state.watchlist.items.findIndex(
            (item) => item.id === action.payload.id
          );
          if (index !== -1) {
            state.watchlist.items[index] = action.payload;
          }
        }
      })
      .addCase(updateWatchlistItem.rejected, (state, action) => {
        state.watchlistLoading = false;
        state.watchlistError = action.payload as string;
      });
  },
});

export const { updatePosition, clearError, clearWatchlistError } = portfolioSlice.actions;
export default portfolioSlice.reducer;