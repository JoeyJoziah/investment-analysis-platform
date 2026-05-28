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

// The backend wraps successful responses in an ApiResponse envelope
// ({ success, data: <payload> }). Axios exposes the body on response.data, so
// the real payload lives at response.data.data. Unwrap defensively so reducers
// receive the actual payload rather than the envelope — otherwise the consumed
// fields are undefined and the page renders empty.
const unwrapData = <T = unknown>(body: unknown): T => {
  if (body && typeof body === 'object' && 'data' in body) {
    return (body as { data: T }).data;
  }
  return body as T;
};

// Async thunks
export const fetchStockData = createAsyncThunk(
  'stock/fetchData',
  async (ticker: string) => {
    // Use allSettled so a missing/optional sub-endpoint (technical, fundamental and
    // news are not all implemented yet -> 404) does NOT discard the working quote.
    // The quote is the primary data; the rest are best-effort.
    const [quoteR, technicalR, fundamentalR, newsR] = await Promise.allSettled([
      apiService.get(`/api/v1/stocks/${ticker}/quote`),
      apiService.get(`/api/v1/stocks/${ticker}/technical`),
      apiService.get(`/api/v1/stocks/${ticker}/fundamental`),
      apiService.get(`/api/v1/stocks/${ticker}/news`),
    ]);

    // r.value is the axios response, so r.value.data is the ApiResponse ENVELOPE.
    // Unwrap one more level via unwrapData so callers receive the real payload
    // (r.value.data.data) for quote/technical/fundamental/news.
    const dataOf = (r: PromiseSettledResult<{ data: unknown }>): unknown =>
      r.status === 'fulfilled' ? unwrapData(r.value.data) : null;

    const rawQuote = dataOf(quoteR);
    if (!rawQuote) {
      // Only fail the whole load if even the quote is unavailable.
      throw new Error(`No quote data available for ${ticker}`);
    }

    // The backend returns snake_case quote fields; map them to the camelCase StockQuote
    // shape the UI renders so values display and undefined fields never crash *.toFixed().
    const q = rawQuote as Record<string, unknown>;
    const num = (v: unknown): number => (typeof v === 'number' ? v : Number(v) || 0);
    const str = (v: unknown): string => (typeof v === 'string' ? v : '');
    const quote = {
      ticker: str(q.symbol ?? q.ticker) || ticker,
      companyName: str(q.company_name ?? q.companyName ?? q.name),
      price: num(q.price),
      change: num(q.change),
      changePercent: num(q.change_percent ?? q.changePercent),
      volume: num(q.volume),
      avgVolume: num(q.avg_volume ?? q.avgVolume),
      marketCap: num(q.market_cap ?? q.marketCap),
      peRatio: num(q.pe_ratio ?? q.peRatio),
      week52High: num(q.fifty_two_week_high ?? q.week52High),
      week52Low: num(q.fifty_two_week_low ?? q.week52Low),
      dividendYield: num(q.dividend_yield ?? q.dividendYield),
      beta: num(q.beta),
      eps: num(q.eps),
      open: num(q.open),
      high: num(q.high),
      low: num(q.low),
      previousClose: num(q.previous_close ?? q.previousClose),
      timestamp: str(q.timestamp),
    };

    return {
      ticker,
      quote,
      technical: dataOf(technicalR),
      fundamental: dataOf(fundamentalR),
      news: dataOf(newsR),
    };
  }
);

export const fetchStockChart = createAsyncThunk(
  'stock/fetchChart',
  async ({ ticker, interval }: { ticker: string; interval: string }) => {
    const response = await apiService.get(`/api/v1/stocks/${ticker}/chart`, {
      params: { interval },
    });
    return unwrapData(response.data);
  }
);

export const fetchOptionsChain = createAsyncThunk(
  'stock/fetchOptions',
  async (ticker: string) => {
    const response = await apiService.get(`/api/v1/stocks/${ticker}/options`);
    return unwrapData(response.data);
  }
);

export const searchStocks = createAsyncThunk(
  'stock/search',
  async (query: string) => {
    const response = await apiService.get('/api/v1/stocks/search', {
      params: { q: query },
    });
    return unwrapData(response.data);
  }
);

export const fetchSimilarStocks = createAsyncThunk(
  'stock/fetchSimilar',
  async (ticker: string) => {
    const response = await apiService.get(`/api/v1/stocks/${ticker}/similar`);
    return unwrapData(response.data);
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