import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Analysis from './Analysis';
import { renderWithProviders, mergeWithDefaults } from '../test-utils';

// Mock all stock slice thunks to prevent state corruption from async resolutions.
// The thunks fire on mount; without mocking, the mock API returns { data: {} }
// which causes the reducer to overwrite technicalIndicators with undefined.
vi.mock('../store/slices/stockSlice', async () => {
  const actual = await vi.importActual<typeof import('../store/slices/stockSlice')>('../store/slices/stockSlice');
  return {
    ...actual,
    fetchStockData: vi.fn(() => ({ type: 'stock/fetchData/noop' })),
    fetchStockChart: vi.fn(() => ({ type: 'stock/fetchChart/noop' })),
    fetchOptionsChain: vi.fn(() => ({ type: 'stock/fetchOptions/noop' })),
    fetchSimilarStocks: vi.fn(() => ({ type: 'stock/fetchSimilar/noop' })),
  };
});

// Mock the API service
vi.mock('../services/api.service', () => ({
  apiService: {
    get: vi.fn().mockImplementation((url: string) => {
      if (url.includes('/recommendations')) {
        return Promise.resolve({ data: { recommendations: [], total: 0 } });
      }
      return Promise.resolve({ data: {} });
    }),
    post: vi.fn().mockResolvedValue({ data: {} }),
  },
}));

// Mock chart components (they rely on canvas/SVG which jsdom does not fully support)
vi.mock('../components/charts/StockChart', () => ({
  default: () => <div data-testid="stock-chart">Stock Chart</div>,
}));

vi.mock('recharts', async (importOriginal) => {
  const actual = await importOriginal<typeof import('recharts')>();
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container">{children}</div>
    ),
    RadarChart: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="radar-chart">{children}</div>
    ),
    PolarGrid: () => null,
    PolarAngleAxis: () => null,
    PolarRadiusAxis: () => null,
    Radar: () => null,
  };
});

// Mock react-router-dom useParams
const mockUseParams = vi.fn();
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useParams: () => mockUseParams(),
    useNavigate: () => mockNavigate,
  };
});

const mockQuote = {
  ticker: 'AAPL',
  companyName: 'Apple Inc.',
  price: 175.50,
  change: 2.30,
  changePercent: 1.33,
  volume: 52000000,
  avgVolume: 48000000,
  marketCap: 2800000000000,
  peRatio: 28.5,
  week52High: 199.62,
  week52Low: 124.17,
  dividendYield: 0.0055,
  beta: 1.28,
  eps: 6.16,
  open: 173.20,
  high: 176.10,
  low: 172.80,
  previousClose: 173.20,
  timestamp: '2024-01-24T16:00:00Z',
};

const mockTechnicalIndicators = {
  rsi: 55.2,
  macd: { macd: 1.5, signal: 1.2, histogram: 0.3 },
  sma: { sma20: 172.0, sma50: 168.0, sma200: 160.0 },
  ema: { ema12: 173.5, ema26: 170.0 },
  bollingerBands: { upper: 180.0, middle: 172.0, lower: 164.0 },
  stochastic: { k: 65, d: 60 },
  atr: 3.5,
  adx: 25.0,
  obv: 1000000,
  volumeProfile: [],
  signals: {
    trend: 'bullish' as const,
    momentum: 'strong' as const,
    volatility: 'medium' as const,
    recommendation: 'buy' as const,
  },
};

const mockFundamentalData = {
  revenue: 383000000000,
  revenueGrowth: 2.1,
  earnings: 97000000000,
  earningsGrowth: 5.3,
  profitMargin: 25.3,
  operatingMargin: 29.8,
  roe: 150.1,
  roa: 28.3,
  debtToEquity: 1.73,
  currentRatio: 0.99,
  quickRatio: 0.94,
  freeCashFlow: 111000000000,
  bookValue: 4.25,
  priceToBook: 41.3,
  priceToSales: 7.3,
  pegRatio: 2.1,
  forwardPE: 25.5,
  dividendRate: 0.96,
  payoutRatio: 15.5,
  insiderOwnership: 0.07,
  institutionalOwnership: 60.5,
  shortInterest: 0.7,
  analystRating: {
    consensus: 'Buy',
    targetPrice: 200,
    strongBuy: 15,
    buy: 20,
    hold: 8,
    sell: 2,
    strongSell: 0,
  },
};

describe('Analysis', () => {
  beforeEach(() => {
    mockNavigate.mockClear();
  });

  describe('without ticker param', () => {
    beforeEach(() => {
      mockUseParams.mockReturnValue({ ticker: undefined });
    });

    it('renders empty state with search prompt', () => {
      renderWithProviders(<Analysis />, { preloadedState: mergeWithDefaults({}) });

      expect(screen.getByText(/enter a stock ticker above to begin analysis/i)).toBeInTheDocument();
    });

    it('renders ticker search input', () => {
      renderWithProviders(<Analysis />, { preloadedState: mergeWithDefaults({}) });

      expect(screen.getByLabelText(/ticker symbol/i)).toBeInTheDocument();
    });

    it('renders analyze button', () => {
      renderWithProviders(<Analysis />, { preloadedState: mergeWithDefaults({}) });

      expect(screen.getByRole('button', { name: /analyze/i })).toBeInTheDocument();
    });

    it('disables analyze button when search is empty', () => {
      renderWithProviders(<Analysis />, { preloadedState: mergeWithDefaults({}) });

      expect(screen.getByRole('button', { name: /analyze/i })).toBeDisabled();
    });

    it('navigates on search submit', async () => {
      const user = userEvent.setup();
      renderWithProviders(<Analysis />, { preloadedState: mergeWithDefaults({}) });

      const input = screen.getByLabelText(/ticker symbol/i);
      await user.type(input, 'AAPL');

      const analyzeButton = screen.getByRole('button', { name: /analyze/i });
      await user.click(analyzeButton);

      expect(mockNavigate).toHaveBeenCalledWith('/analysis/AAPL');
    });
  });

  describe('with ticker param and loading', () => {
    beforeEach(() => {
      mockUseParams.mockReturnValue({ ticker: 'AAPL' });
    });

    it('shows loading indicator when data is loading', () => {
      const state = mergeWithDefaults({
        stock: {
          selectedTicker: 'AAPL',
          quote: null,
          chartData: null,
          technicalIndicators: null,
          fundamentalData: null,
          news: [],
          optionsChain: null,
          similarStocks: [],
          searchResults: [],
          isLoading: true,
          error: null,
        },
      });

      renderWithProviders(<Analysis />, { preloadedState: state });

      expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });

    it('shows error alert on error', () => {
      const state = mergeWithDefaults({
        stock: {
          selectedTicker: 'AAPL',
          quote: null,
          chartData: null,
          technicalIndicators: null,
          fundamentalData: null,
          news: [],
          optionsChain: null,
          similarStocks: [],
          searchResults: [],
          isLoading: false,
          error: 'Failed to load stock data',
        },
      });

      renderWithProviders(<Analysis />, { preloadedState: state });

      expect(screen.getByRole('alert')).toBeInTheDocument();
      expect(screen.getByText('Failed to load stock data')).toBeInTheDocument();
    });
  });

  describe('with quote loaded', () => {
    const stateWithQuote = mergeWithDefaults({
      stock: {
        selectedTicker: 'AAPL',
        quote: mockQuote,
        chartData: null,
        technicalIndicators: mockTechnicalIndicators,
        fundamentalData: mockFundamentalData,
        news: [],
        optionsChain: null,
        similarStocks: [],
        searchResults: [],
        isLoading: false,
        error: null,
      },
    });

    beforeEach(() => {
      mockUseParams.mockReturnValue({ ticker: 'AAPL' });
    });

    it('renders ticker symbol', () => {
      renderWithProviders(<Analysis />, { preloadedState: stateWithQuote });

      expect(screen.getByRole('heading', { name: /AAPL/i })).toBeInTheDocument();
    });

    it('renders company name', () => {
      renderWithProviders(<Analysis />, { preloadedState: stateWithQuote });

      expect(screen.getByText('Apple Inc.')).toBeInTheDocument();
    });

    it('renders stock price', () => {
      renderWithProviders(<Analysis />, { preloadedState: stateWithQuote });

      expect(screen.getByText('$175.50')).toBeInTheDocument();
    });

    it('renders key statistics section', () => {
      renderWithProviders(<Analysis />, { preloadedState: stateWithQuote });

      expect(screen.getByText('Key Statistics')).toBeInTheDocument();
      expect(screen.getByText('Market Cap')).toBeInTheDocument();
      expect(screen.getByText('P/E Ratio')).toBeInTheDocument();
    });

    it('renders the refresh button', () => {
      renderWithProviders(<Analysis />, { preloadedState: stateWithQuote });

      expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument();
    });

    it('renders all analysis tabs', () => {
      renderWithProviders(<Analysis />, { preloadedState: stateWithQuote });

      expect(screen.getByRole('tab', { name: /chart/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /technical/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /fundamental/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /news/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /options/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /similar/i })).toBeInTheDocument();
    });

    it('shows chart tab as default', () => {
      renderWithProviders(<Analysis />, { preloadedState: stateWithQuote });

      expect(screen.getByRole('tab', { name: /chart/i })).toHaveAttribute('aria-selected', 'true');
    });

    it('switches to technical tab', async () => {
      const user = userEvent.setup();
      renderWithProviders(<Analysis />, { preloadedState: stateWithQuote });

      const technicalTab = screen.getByRole('tab', { name: /technical/i });
      await user.click(technicalTab);

      expect(technicalTab).toHaveAttribute('aria-selected', 'true');

      await waitFor(() => {
        expect(screen.getByText('Technical Indicators')).toBeInTheDocument();
      });
    });

    it('switches to fundamental tab', async () => {
      const user = userEvent.setup();
      renderWithProviders(<Analysis />, { preloadedState: stateWithQuote });

      const fundamentalTab = screen.getByRole('tab', { name: /fundamental/i });
      await user.click(fundamentalTab);

      expect(fundamentalTab).toHaveAttribute('aria-selected', 'true');

      await waitFor(() => {
        expect(screen.getByText('Financial Performance')).toBeInTheDocument();
      });
      expect(screen.getByText('Valuation Metrics')).toBeInTheDocument();
      expect(screen.getByText('Financial Health')).toBeInTheDocument();
    });

    it('switches to news tab', async () => {
      const user = userEvent.setup();
      renderWithProviders(<Analysis />, { preloadedState: stateWithQuote });

      const newsTab = screen.getByRole('tab', { name: /news/i });
      await user.click(newsTab);

      expect(newsTab).toHaveAttribute('aria-selected', 'true');
    });

    it('displays technical trading signals', async () => {
      const user = userEvent.setup();
      renderWithProviders(<Analysis />, { preloadedState: stateWithQuote });

      const technicalTab = screen.getByRole('tab', { name: /technical/i });
      await user.click(technicalTab);

      await waitFor(() => {
        expect(screen.getByText('Trading Signals')).toBeInTheDocument();
      });
      expect(screen.getByText('Trend')).toBeInTheDocument();
      expect(screen.getByText('Momentum')).toBeInTheDocument();
      expect(screen.getByText('Volatility')).toBeInTheDocument();
      expect(screen.getByText('Recommendation')).toBeInTheDocument();
    });

    it('displays analyst ratings in fundamental tab', async () => {
      const user = userEvent.setup();
      renderWithProviders(<Analysis />, { preloadedState: stateWithQuote });

      const fundamentalTab = screen.getByRole('tab', { name: /fundamental/i });
      await user.click(fundamentalTab);

      await waitFor(() => {
        expect(screen.getByText('Analyst Ratings')).toBeInTheDocument();
      });
      expect(screen.getByText('Consensus Rating')).toBeInTheDocument();
    });
  });
});
