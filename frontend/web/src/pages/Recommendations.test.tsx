import { describe, it, expect, vi } from 'vitest';
import { screen } from '@testing-library/react';
import Recommendations from './Recommendations';
import { renderWithProviders, mergeWithDefaults } from '../test-utils';

// Mock the thunks to prevent them from overwriting preloaded state.
const noopThunk = () => () => Promise.resolve();

vi.mock('../store/slices/recommendationsSlice', async () => {
  const actual = await vi.importActual('../store/slices/recommendationsSlice');
  return {
    ...actual,
    fetchRecommendations: vi.fn(() => noopThunk()),
  };
});

vi.mock('../store/slices/portfolioSlice', async () => {
  const actual = await vi.importActual('../store/slices/portfolioSlice');
  return {
    ...actual,
    addToWatchlist: vi.fn(() => noopThunk()),
    removeFromWatchlist: vi.fn(() => noopThunk()),
  };
});

vi.mock('../services/api.service', () => ({
  apiService: {
    get: vi.fn().mockResolvedValue({ data: {} }),
    post: vi.fn().mockResolvedValue({ data: {} }),
    delete: vi.fn().mockResolvedValue({ data: {} }),
  },
}));

describe('Recommendations', () => {
  const emptyState = mergeWithDefaults({
    recommendations: {
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
      sortBy: 'confidence' as const,
      sortOrder: 'desc' as const,
      pagination: { page: 1, limit: 20, total: 0 },
      loading: false,
      error: null,
    },
    portfolio: {
      positions: [],
      transactions: [],
      metrics: null,
      watchlist: { id: 1, name: 'Default', description: null, is_public: false, user_id: 1, items: [] },
      watchlistLoading: false,
      watchlistError: null,
      isLoading: false,
      error: null,
      lastUpdated: null,
    },
  });

  const populatedState = mergeWithDefaults({
    recommendations: {
      recommendations: [
        {
          id: '1',
          ticker: 'AAPL',
          companyName: 'Apple Inc.',
          sector: 'Technology',
          price: 175.50,
          targetPrice: 200.00,
          recommendation: 'BUY',
          confidence: 85,
          signals: { technical: 80, fundamental: 90, sentiment: 75, ml_prediction: 88 },
          reasons: ['Strong earnings growth', 'Positive analyst sentiment'],
          risk: 'LOW',
          timeHorizon: 'MEDIUM',
          expectedReturn: 14.0,
          lastUpdated: '2024-01-24',
        },
        {
          id: '2',
          ticker: 'TSLA',
          companyName: 'Tesla Inc.',
          sector: 'Automotive',
          price: 220.00,
          targetPrice: 180.00,
          recommendation: 'SELL',
          confidence: 70,
          signals: { technical: 40, fundamental: 50, sentiment: 35, ml_prediction: 45 },
          reasons: ['Declining margins', 'Increased competition'],
          risk: 'HIGH',
          timeHorizon: 'SHORT',
          expectedReturn: -18.2,
          lastUpdated: '2024-01-24',
        },
      ],
      filteredRecommendations: [],
      selectedRecommendation: null,
      filters: {
        action: null,
        riskLevel: null,
        sector: null,
        minConfidence: 0,
        minReturn: 0,
      },
      sortBy: 'confidence' as const,
      sortOrder: 'desc' as const,
      pagination: { page: 1, limit: 20, total: 2 },
      loading: false,
      error: null,
    },
    portfolio: {
      positions: [],
      transactions: [],
      metrics: null,
      watchlist: { id: 1, name: 'Default', description: null, is_public: false, user_id: 1, items: [] },
      watchlistLoading: false,
      watchlistError: null,
      isLoading: false,
      error: null,
      lastUpdated: null,
    },
  });

  it('renders the recommendations page title', () => {
    renderWithProviders(<Recommendations />, { preloadedState: emptyState });

    expect(screen.getByText('AI Recommendations')).toBeInTheDocument();
  });

  it('renders refresh button', () => {
    renderWithProviders(<Recommendations />, { preloadedState: emptyState });

    expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument();
  });

  it('renders view mode toggle buttons', () => {
    renderWithProviders(<Recommendations />, { preloadedState: emptyState });

    const toggleButtons = screen.getAllByRole('button').filter(
      (btn) => btn.getAttribute('value') === 'grid' || btn.getAttribute('value') === 'list'
    );
    expect(toggleButtons.length).toBe(2);
  });

  it('renders filter controls', () => {
    renderWithProviders(<Recommendations />, { preloadedState: emptyState });

    expect(screen.getByPlaceholderText(/search ticker or company/i)).toBeInTheDocument();
  });

  it('renders empty state when no recommendations', () => {
    renderWithProviders(<Recommendations />, { preloadedState: emptyState });

    expect(screen.getByText('No recommendations available yet')).toBeInTheDocument();
  });

  it('renders recommendation count', () => {
    renderWithProviders(<Recommendations />, { preloadedState: populatedState });

    expect(screen.getByText(/showing 2 of 2 recommendations/i)).toBeInTheDocument();
  });

  it('renders recommendation cards in grid view', () => {
    renderWithProviders(<Recommendations />, { preloadedState: populatedState });

    expect(screen.getByText('AAPL')).toBeInTheDocument();
    expect(screen.getByText('Apple Inc.')).toBeInTheDocument();
    expect(screen.getByText('TSLA')).toBeInTheDocument();
    expect(screen.getByText('Tesla Inc.')).toBeInTheDocument();
  });

  it('renders confidence scores', () => {
    renderWithProviders(<Recommendations />, { preloadedState: populatedState });

    expect(screen.getByText('85%')).toBeInTheDocument();
    expect(screen.getByText('70%')).toBeInTheDocument();
  });

  it('renders recommendation chips', () => {
    renderWithProviders(<Recommendations />, { preloadedState: populatedState });

    expect(screen.getByText('BUY')).toBeInTheDocument();
    expect(screen.getByText('SELL')).toBeInTheDocument();
  });

  it('renders sector chips', () => {
    renderWithProviders(<Recommendations />, { preloadedState: populatedState });

    expect(screen.getByText('Technology')).toBeInTheDocument();
    expect(screen.getByText('Automotive')).toBeInTheDocument();
  });

  it('shows loading indicator when loading', () => {
    const loadingState = mergeWithDefaults({
      recommendations: {
        ...emptyState.recommendations,
        loading: true,
      },
    });

    renderWithProviders(<Recommendations />, { preloadedState: loadingState });

    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  it('renders minimum confidence slider', () => {
    renderWithProviders(<Recommendations />, { preloadedState: emptyState });

    expect(screen.getByText(/minimum confidence/i)).toBeInTheDocument();
  });

  it('renders view analysis buttons', () => {
    renderWithProviders(<Recommendations />, { preloadedState: populatedState });

    const viewButtons = screen.getAllByRole('button', { name: /view analysis/i });
    expect(viewButtons.length).toBe(2);
  });
});
