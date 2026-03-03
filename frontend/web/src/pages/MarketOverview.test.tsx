import { describe, it, expect, vi } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import MarketOverview from './MarketOverview';
import { renderWithProviders, mergeWithDefaults } from '../test-utils';

// Mock recharts to avoid rendering issues in jsdom
vi.mock('recharts', () => ({
  BarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Cell: () => null,
  AreaChart: () => null,
  Area: () => null,
  LineChart: () => null,
  Line: () => null,
  PieChart: () => null,
  Pie: () => null,
  Legend: () => null,
  Treemap: () => null,
}));

// Mock MarketHeatmap
vi.mock('../components/charts/MarketHeatmap', () => ({
  default: () => <div data-testid="market-heatmap">Market Heatmap</div>,
}));

// Mock all market slice thunks to prevent state corruption from async resolutions.
// The thunks fire on mount; without mocking, the mock API returns { data: {} }
// which causes the reducer to overwrite arrays with undefined.
vi.mock('../store/slices/marketSlice', async () => {
  const actual = await vi.importActual<typeof import('../store/slices/marketSlice')>('../store/slices/marketSlice');
  return {
    ...actual,
    fetchMarketOverview: vi.fn(() => ({ type: 'market/fetchOverview/noop' })),
    fetchSectorPerformance: vi.fn(() => ({ type: 'market/fetchSectors/noop' })),
    fetchMarketNews: vi.fn(() => ({ type: 'market/fetchNews/noop' })),
    fetchHeatmapData: vi.fn(() => ({ type: 'market/fetchHeatmap/noop' })),
    fetchEconomicCalendar: vi.fn(() => ({ type: 'market/fetchCalendar/noop' })),
  };
});

// Mock API service as fallback
vi.mock('../services/api.service', () => ({
  apiService: {
    get: vi.fn().mockResolvedValue({ data: {} }),
    post: vi.fn().mockResolvedValue({ data: {} }),
  },
}));

describe('MarketOverview', () => {
  const defaultState = mergeWithDefaults({
    market: {
      indices: [
        {
          symbol: 'SPY',
          name: 'S&P 500',
          value: 4800,
          change: 12.5,
          changePercent: 0.26,
          volume: 3500000000,
          high: 4810,
          low: 4785,
          previousClose: 4787.5,
          timestamp: '2026-03-03T16:00:00Z',
        },
        {
          symbol: 'QQQ',
          name: 'Nasdaq 100',
          value: 16500,
          change: -25.3,
          changePercent: -0.15,
          volume: 2100000000,
          high: 16550,
          low: 16450,
          previousClose: 16525.3,
          timestamp: '2026-03-03T16:00:00Z',
        },
      ],
      topGainers: [
        {
          ticker: 'NVDA',
          companyName: 'NVIDIA Corporation',
          price: 820.5,
          change: 45.2,
          changePercent: 5.83,
          volume: 50000000,
          marketCap: 2000000000000,
          sector: 'Technology',
        },
      ],
      topLosers: [
        {
          ticker: 'META',
          companyName: 'Meta Platforms Inc.',
          price: 480.3,
          change: -12.5,
          changePercent: -2.54,
          volume: 25000000,
          marketCap: 1200000000000,
          sector: 'Technology',
        },
      ],
      mostActive: [
        {
          ticker: 'TSLA',
          companyName: 'Tesla Inc.',
          price: 195.8,
          change: 3.2,
          changePercent: 1.66,
          volume: 120000000,
          marketCap: 600000000000,
          sector: 'Consumer Discretionary',
        },
      ],
      sectorPerformance: [
        {
          sector: 'Technology',
          changePercent: 1.25,
          marketCap: 15000000000000,
          volume: 5000000000,
          gainers: 150,
          losers: 80,
          topStock: { ticker: 'NVDA', changePercent: 5.83 },
        },
      ],
      marketNews: [
        {
          id: 'news-1',
          title: 'Market Rally Continues',
          summary: 'Stocks rose broadly on strong earnings reports.',
          url: 'https://example.com/news/1',
          source: 'Reuters',
          publishedAt: '2026-03-03T12:00:00Z',
          sentiment: 'positive' as const,
          relatedTickers: ['AAPL', 'MSFT'],
          image: 'https://example.com/image.jpg',
        },
      ],
      marketBreadth: {
        advancers: 320,
        decliners: 180,
        unchanged: 50,
        newHighs: 45,
        newLows: 12,
        advanceDeclineRatio: 1.78,
        upVolume: 2500000000,
        downVolume: 1200000000,
        totalVolume: 3700000000,
      },
      heatmapData: [],
      economicCalendar: [],
      isLoading: false,
      error: null,
      lastUpdated: null,
    },
  });

  describe('rendering', () => {
    it('renders the page title', () => {
      renderWithProviders(<MarketOverview />, { preloadedState: defaultState });
      expect(screen.getByText('Market Overview')).toBeInTheDocument();
    });

    it('renders the refresh button', () => {
      renderWithProviders(<MarketOverview />, { preloadedState: defaultState });
      expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument();
    });

    it('renders market index cards', () => {
      renderWithProviders(<MarketOverview />, { preloadedState: defaultState });
      expect(screen.getByText('S&P 500')).toBeInTheDocument();
      expect(screen.getByText('Nasdaq 100')).toBeInTheDocument();
    });

    it('renders index values', () => {
      renderWithProviders(<MarketOverview />, { preloadedState: defaultState });
      expect(screen.getByText('4,800')).toBeInTheDocument();
      expect(screen.getByText('16,500')).toBeInTheDocument();
    });

    it('renders market breadth section', () => {
      renderWithProviders(<MarketOverview />, { preloadedState: defaultState });
      expect(screen.getByText('Market Breadth')).toBeInTheDocument();
      expect(screen.getByText('320')).toBeInTheDocument();
      expect(screen.getByText('180')).toBeInTheDocument();
    });

    it('renders tabs', () => {
      renderWithProviders(<MarketOverview />, { preloadedState: defaultState });
      expect(screen.getByRole('tab', { name: /movers/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /sectors/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /heat map/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /news/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /economic calendar/i })).toBeInTheDocument();
    });
  });

  describe('movers tab (default)', () => {
    it('renders top gainers table', () => {
      renderWithProviders(<MarketOverview />, { preloadedState: defaultState });
      expect(screen.getByText('Top Gainers')).toBeInTheDocument();
      expect(screen.getByText('NVDA')).toBeInTheDocument();
    });

    it('renders top losers table', () => {
      renderWithProviders(<MarketOverview />, { preloadedState: defaultState });
      expect(screen.getByText('Top Losers')).toBeInTheDocument();
      expect(screen.getByText('META')).toBeInTheDocument();
    });

    it('renders most active table', () => {
      renderWithProviders(<MarketOverview />, { preloadedState: defaultState });
      expect(screen.getByText('Most Active')).toBeInTheDocument();
      expect(screen.getByText('TSLA')).toBeInTheDocument();
    });
  });

  describe('tab switching', () => {
    it('switches to sectors tab', async () => {
      const user = userEvent.setup();
      renderWithProviders(<MarketOverview />, { preloadedState: defaultState });

      await user.click(screen.getByRole('tab', { name: /sectors/i }));
      expect(screen.getByText('Sector Performance')).toBeInTheDocument();
      expect(screen.getByText('Technology')).toBeInTheDocument();
    });

    it('switches to news tab', async () => {
      const user = userEvent.setup();
      renderWithProviders(<MarketOverview />, { preloadedState: defaultState });

      await user.click(screen.getByRole('tab', { name: /news/i }));
      expect(screen.getByText('Market News')).toBeInTheDocument();
      expect(screen.getByText('Market Rally Continues')).toBeInTheDocument();
    });

    it('switches to heat map tab', async () => {
      const user = userEvent.setup();
      renderWithProviders(<MarketOverview />, { preloadedState: defaultState });

      await user.click(screen.getByRole('tab', { name: /heat map/i }));
      expect(screen.getByText('Market Heat Map')).toBeInTheDocument();
    });

    it('switches to economic calendar tab', async () => {
      const user = userEvent.setup();
      renderWithProviders(<MarketOverview />, { preloadedState: defaultState });

      await user.click(screen.getByRole('tab', { name: /economic calendar/i }));
      // Tab label + panel heading both say "Economic Calendar"
      expect(screen.getAllByText('Economic Calendar').length).toBeGreaterThanOrEqual(2);
    });
  });

  describe('loading state', () => {
    it('shows loading indicator when loading', () => {
      const loadingState = mergeWithDefaults({
        market: {
          ...defaultState.market,
          isLoading: true,
        },
      });
      renderWithProviders(<MarketOverview />, { preloadedState: loadingState });
      expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });
  });

  describe('empty state', () => {
    it('shows empty state when no indices', () => {
      const emptyState = mergeWithDefaults({
        market: {
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
        },
      });
      renderWithProviders(<MarketOverview />, { preloadedState: emptyState });
      expect(screen.getByText('No market index data available')).toBeInTheDocument();
    });

    it('shows empty state when no movers', () => {
      const emptyMovers = mergeWithDefaults({
        market: {
          ...defaultState.market,
          topGainers: [],
          topLosers: [],
          mostActive: [],
        },
      });
      renderWithProviders(<MarketOverview />, { preloadedState: emptyMovers });
      expect(screen.getByText('No market mover data available')).toBeInTheDocument();
    });
  });
});
