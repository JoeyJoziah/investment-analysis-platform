import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Dashboard from './Dashboard';
import { renderWithProviders, mockPortfolioSummary, mockCostMetrics, mergeWithDefaults } from '../test-utils';

// Mock the child components to simplify testing
vi.mock('../components/charts/StockChart', () => ({
  default: () => <div data-testid="stock-chart">Stock Chart</div>,
}));

vi.mock('../components/cards/RecommendationCard', () => ({
  default: ({ recommendation }: { recommendation: { ticker: string } }) => (
    <div data-testid="recommendation-card">{recommendation.ticker}</div>
  ),
}));

vi.mock('../components/charts/MarketHeatmap', () => ({
  default: () => <div data-testid="market-heatmap">Market Heatmap</div>,
}));

vi.mock('../components/cards/PortfolioSummary', () => ({
  default: () => <div data-testid="portfolio-summary">Portfolio Summary</div>,
}));

vi.mock('../components/cards/NewsCard', () => ({
  default: ({ news }: { news: { title: string } }) => (
    <div data-testid="news-card">{news.title}</div>
  ),
}));

vi.mock('../components/monitoring/CostMonitor', () => ({
  default: () => <div data-testid="cost-monitor">Cost Monitor</div>,
}));

// Mock the API service
vi.mock('../services/api.service', () => ({
  apiService: {
    get: vi.fn().mockImplementation((url: string) => {
      if (url.includes('/recommendations')) {
        return Promise.resolve({ data: { recommendations: [], total: 0 } });
      }
      return Promise.resolve({ data: {} });
    }),
  },
}));

describe('Dashboard', () => {
  const defaultState = mergeWithDefaults({
    dashboard: {
      marketOverview: {
        indices: [
          { symbol: 'SPY', value: 480, change: 1.2, changePercent: 0.25 },
          { symbol: 'QQQ', value: 400, change: -0.8, changePercent: -0.20 },
        ],
        heatmap: [],
        sectors: [
          { name: 'Technology', change: 1.5, volume: 1000000 },
          { name: 'Healthcare', change: -0.5, volume: 500000 },
        ],
      },
      topRecommendations: [
        { ticker: 'AAPL', name: 'Apple Inc.', score: 85 },
        { ticker: 'MSFT', name: 'Microsoft Corp.', score: 82 },
      ],
      portfolioSummary: mockPortfolioSummary,
      recentNews: [
        { title: 'Market Update', content: 'Market is up' },
        { title: 'Tech News', content: 'Tech stocks rally' },
      ],
      marketSentiment: {
        overall: 'Bullish',
        score: 75,
        breakdown: { positive: 60, neutral: 30, negative: 10 },
      },
      costMetrics: mockCostMetrics,
      loading: false,
      error: null,
    },
  });

  it('renders the dashboard title', () => {
    renderWithProviders(<Dashboard />, { preloadedState: defaultState });

    expect(screen.getByRole('heading', { name: /investment dashboard/i })).toBeInTheDocument();
  });

  it('renders the current date', () => {
    renderWithProviders(<Dashboard />, { preloadedState: defaultState });

    // Check that some date is rendered
    const dateRegex = /\w+, \w+ \d+, \d{4}/i;
    expect(screen.getByText(dateRegex)).toBeInTheDocument();
  });

  it('renders key metric cards', () => {
    renderWithProviders(<Dashboard />, { preloadedState: defaultState });

    expect(screen.getByText('Portfolio Value')).toBeInTheDocument();
    expect(screen.getByText('Total Return')).toBeInTheDocument();
    expect(screen.getByText('Market Sentiment')).toBeInTheDocument();
    expect(screen.getByText('Active Positions')).toBeInTheDocument();
  });

  it('displays portfolio value correctly', () => {
    renderWithProviders(<Dashboard />, { preloadedState: defaultState });

    expect(screen.getByText('$150,000')).toBeInTheDocument();
  });

  it('displays market sentiment', () => {
    renderWithProviders(<Dashboard />, { preloadedState: defaultState });

    expect(screen.getByText('Bullish')).toBeInTheDocument();
  });

  it('displays active positions count', () => {
    renderWithProviders(<Dashboard />, { preloadedState: defaultState });

    expect(screen.getByText('12')).toBeInTheDocument();
  });

  it('renders market overview section', () => {
    renderWithProviders(<Dashboard />, { preloadedState: defaultState });

    expect(screen.getByText('Market Overview')).toBeInTheDocument();
    expect(screen.getByTestId('market-heatmap')).toBeInTheDocument();
  });

  it('renders top recommendations section', () => {
    renderWithProviders(<Dashboard />, { preloadedState: defaultState });

    expect(screen.getByText('Top Recommendations')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /view all/i })).toBeInTheDocument();
  });

  it('renders recommendation cards', () => {
    renderWithProviders(<Dashboard />, { preloadedState: defaultState });

    const recommendationCards = screen.getAllByTestId('recommendation-card');
    expect(recommendationCards.length).toBeGreaterThan(0);
  });

  it('renders portfolio performance section', () => {
    renderWithProviders(<Dashboard />, { preloadedState: defaultState });

    expect(screen.getByText('Portfolio Performance')).toBeInTheDocument();
    expect(screen.getByTestId('stock-chart')).toBeInTheDocument();
  });

  it('renders portfolio summary component', () => {
    renderWithProviders(<Dashboard />, { preloadedState: defaultState });

    expect(screen.getByTestId('portfolio-summary')).toBeInTheDocument();
  });

  it('renders market news section', () => {
    renderWithProviders(<Dashboard />, { preloadedState: defaultState });

    expect(screen.getByText('Market News & Insights')).toBeInTheDocument();
  });

  it('renders news cards', () => {
    renderWithProviders(<Dashboard />, { preloadedState: defaultState });

    const newsCards = screen.getAllByTestId('news-card');
    expect(newsCards.length).toBeGreaterThan(0);
  });

  it('renders cost monitor component', () => {
    renderWithProviders(<Dashboard />, { preloadedState: defaultState });

    expect(screen.getByTestId('cost-monitor')).toBeInTheDocument();
  });

  it('renders sector performance section', () => {
    renderWithProviders(<Dashboard />, { preloadedState: defaultState });

    expect(screen.getByText('Sector Performance')).toBeInTheDocument();
    expect(screen.getByText('Technology')).toBeInTheDocument();
    expect(screen.getByText('Healthcare')).toBeInTheDocument();
  });

  it('displays index chips in market overview', () => {
    renderWithProviders(<Dashboard />, { preloadedState: defaultState });

    // Check for the index symbol in a chip
    expect(screen.getByText(/SPY:/)).toBeInTheDocument();
  });

  describe('loading state', () => {
    it('shows loading indicator when loading', () => {
      const loadingState = mergeWithDefaults({
        dashboard: {
          ...defaultState.dashboard,
          loading: true,
        },
      });

      renderWithProviders(<Dashboard />, { preloadedState: loadingState });

      expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('shows error alert when there is an error', async () => {
      // Mock API to reject so error state persists
      const { apiService } = await import('../services/api.service');
      vi.mocked(apiService.get).mockRejectedValueOnce(new Error('Failed to load dashboard data'));

      const errorState = mergeWithDefaults({
        dashboard: {
          ...defaultState.dashboard,
          error: 'Failed to load dashboard data',
          loading: false,
        },
      });

      renderWithProviders(<Dashboard />, { preloadedState: errorState });

      // Wait for error state to be displayed
      await waitFor(() => {
        expect(screen.getByRole('alert')).toBeInTheDocument();
      });
    });
  });

  describe('refresh functionality', () => {
    it('has a refresh button', () => {
      renderWithProviders(<Dashboard />, { preloadedState: defaultState });

      expect(screen.getByRole('button', { name: /refresh data/i })).toBeInTheDocument();
    });

    it('has a notifications button', () => {
      renderWithProviders(<Dashboard />, { preloadedState: defaultState });

      expect(screen.getByRole('button', { name: /view notifications/i })).toBeInTheDocument();
    });
  });

  describe('empty data handling', () => {
    it('handles missing portfolio summary gracefully', () => {
      const stateWithoutPortfolio = mergeWithDefaults({
        dashboard: {
          ...defaultState.dashboard,
          portfolioSummary: null,
        },
      });

      renderWithProviders(<Dashboard />, { preloadedState: stateWithoutPortfolio });

      expect(screen.getByText('Portfolio Value')).toBeInTheDocument();
      expect(screen.getByText('$0')).toBeInTheDocument();
    });

    it('handles missing market sentiment gracefully', () => {
      const stateWithoutSentiment = mergeWithDefaults({
        dashboard: {
          ...defaultState.dashboard,
          marketSentiment: null,
        },
      });

      renderWithProviders(<Dashboard />, { preloadedState: stateWithoutSentiment });

      expect(screen.getByText('Market Sentiment')).toBeInTheDocument();
      expect(screen.getByText('Neutral')).toBeInTheDocument();
    });
  });
});
