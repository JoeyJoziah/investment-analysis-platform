import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { apiService } from '../../services/api.service';

export interface StockQuote {
  ticker: string;
  companyName: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  avgVolume: number;
  marketCap: number;
  peRatio: number;
  week52High: number;
  week52Low: number;
  dividendYield: number;
  beta: number;
  eps: number;
  open: number;
  high: number;
  low: number;
  previousClose: number;
  timestamp: string;
}

export interface StockChart {
  ticker: string;
  interval: '1d' | '1w' | '1m' | '3m' | '6m' | '1y' | '5y' | 'max';
  data: Array<{
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }>;
}

export interface TechnicalIndicators {
  rsi: number;
  macd: {
    macd: number;
    signal: number;
    histogram: number;
  };
  sma: {
    sma20: number;
    sma50: number;
    sma200: number;
  };
  ema: {
    ema12: number;
    ema26: number;
  };
  bollingerBands: {
    upper: number;
    middle: number;
    lower: number;
  };
  stochastic: {
    k: number;
    d: number;
  };
  atr: number;
  adx: number;
  obv: number;
  volumeProfile: Array<{
    price: number;
    volume: number;
  }>;
  signals: {
    trend: 'bullish' | 'bearish' | 'neutral';
    momentum: 'strong' | 'moderate' | 'weak';
    volatility: 'high' | 'medium' | 'low';
    recommendation: 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell';
  };
}

export interface FundamentalData {
  revenue: number;
  revenueGrowth: number;
  earnings: number;
  earningsGrowth: number;
  profitMargin: number;
  operatingMargin: number;
  roe: number;
  roa: number;
  debtToEquity: number;
  currentRatio: number;
  quickRatio: number;
  freeCashFlow: number;
  bookValue: number;
  priceToBook: number;
  priceToSales: number;
  pegRatio: number;
  forwardPE: number;
  dividendRate: number;
  payoutRatio: number;
  insiderOwnership: number;
  institutionalOwnership: number;
  shortInterest: number;
  analystRating: {
    consensus: string;
    targetPrice: number;
    strongBuy: number;
    buy: number;
    hold: number;
    sell: number;
    strongSell: number;
  };
}

export interface StockNews {
  id: string;
  title: string;
  summary: string;
  url: string;
  source: string;
  publishedAt: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  relevanceScore: number;
}

export interface OptionsChain {
  ticker: string;
  expirationDates: string[];
  calls: Array<{
    strike: number;
    bid: number;
    ask: number;
    volume: number;
    openInterest: number;
    impliedVolatility: number;
    inTheMoney: boolean;
  }>;
  puts: Array<{
    strike: number;
    bid: number;
    ask: number;
    volume: number;
    openInterest: number;
    impliedVolatility: number;
    inTheMoney: boolean;
  }>;
}

interface StockState {
  selectedTicker: string | null;
  quote: StockQuote | null;
  chartData: StockChart | null;
  technicalIndicators: TechnicalIndicators | null;
  fundamentalData: FundamentalData | null;
  news: StockNews[];
  optionsChain: OptionsChain | null;
  similarStocks: Array<{
    ticker: string;
    name: string;
    correlation: number;
    changePercent: number;
  }>;
  searchResults: Array<{
    ticker: string;
    name: string;
    exchange: string;
    type: string;
  }>;
  isLoading: boolean;
  error: string | null;
}

const initialState: StockState = {
  selectedTicker: null,
  quote: null,
  chartData: null,
  technicalIndicators: null,
  fundamentalData: null,
  news: [],
  optionsChain: null,
  similarStocks: [],
  searchResults: [],
  isLoading: false,
  error: null,
};

// Async thunks
export const fetchStockData = createAsyncThunk(
  'stock/fetchData',
  async (ticker: string) => {
    const [quote, technical, fundamental, news] = await Promise.all([
      apiService.get(`/stocks/${ticker}/quote`),
      apiService.get(`/stocks/${ticker}/technical`),
      apiService.get(`/stocks/${ticker}/fundamental`),
      apiService.get(`/stocks/${ticker}/news`),
    ]);
    
    return {
      ticker,
      quote: quote.data,
      technical: technical.data,
      fundamental: fundamental.data,
      news: news.data,
    };
  }
);

export const fetchStockChart = createAsyncThunk(
  'stock/fetchChart',
  async ({ ticker, interval }: { ticker: string; interval: string }) => {
    const response = await apiService.get(`/stocks/${ticker}/chart`, {
      params: { interval },
    });
    return response.data;
  }
);

export const fetchOptionsChain = createAsyncThunk(
  'stock/fetchOptions',
  async (ticker: string) => {
    const response = await apiService.get(`/stocks/${ticker}/options`);
    return response.data;
  }
);

export const searchStocks = createAsyncThunk(
  'stock/search',
  async (query: string) => {
    const response = await apiService.get('/stocks/search', {
      params: { q: query },
    });
    return response.data;
  }
);

export const fetchSimilarStocks = createAsyncThunk(
  'stock/fetchSimilar',
  async (ticker: string) => {
    const response = await apiService.get(`/stocks/${ticker}/similar`);
    return response.data;
  }
);

const stockSlice = createSlice({
  name: 'stock',
  initialState,
  reducers: {
    selectStock: (state, action: PayloadAction<string>) => {
      state.selectedTicker = action.payload;
    },
    updateQuote: (state, action: PayloadAction<Partial<StockQuote>>) => {
      if (state.quote) {
        state.quote = { ...state.quote, ...action.payload };
      }
    },
    clearSearchResults: (state) => {
      state.searchResults = [];
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch stock data
      .addCase(fetchStockData.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchStockData.fulfilled, (state, action) => {
        state.isLoading = false;
        state.selectedTicker = action.payload.ticker;
        state.quote = action.payload.quote;
        state.technicalIndicators = action.payload.technical;
        state.fundamentalData = action.payload.fundamental;
        state.news = action.payload.news;
      })
      .addCase(fetchStockData.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Failed to fetch stock data';
      })
      // Fetch chart
      .addCase(fetchStockChart.fulfilled, (state, action) => {
        state.chartData = action.payload;
      })
      // Fetch options
      .addCase(fetchOptionsChain.fulfilled, (state, action) => {
        state.optionsChain = action.payload;
      })
      // Search stocks
      .addCase(searchStocks.fulfilled, (state, action) => {
        state.searchResults = action.payload;
      })
      // Fetch similar stocks
      .addCase(fetchSimilarStocks.fulfilled, (state, action) => {
        state.similarStocks = action.payload;
      });
  },
});

export const { selectStock, updateQuote, clearSearchResults, clearError } = stockSlice.actions;
export default stockSlice.reducer;