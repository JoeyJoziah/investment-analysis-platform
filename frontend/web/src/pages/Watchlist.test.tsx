import { describe, it, expect, vi } from 'vitest';
import { screen } from '@testing-library/react';
import Watchlist from './Watchlist';
import { renderWithProviders, mergeWithDefaults } from '../test-utils';

// Mock the thunks to prevent them from overwriting preloaded state.
// Return a thunk that dispatches a no-op action type the reducer ignores.
const noopThunk = () => () => Promise.resolve();

vi.mock('../store/slices/portfolioSlice', async () => {
  const actual = await vi.importActual('../store/slices/portfolioSlice');
  return {
    ...actual,
    fetchWatchlist: vi.fn(() => noopThunk()),
    addToWatchlist: vi.fn(() => noopThunk()),
    removeFromWatchlist: vi.fn(() => noopThunk()),
    updateWatchlistItem: vi.fn(() => noopThunk()),
  };
});

vi.mock('../services/api.service', () => ({
  apiService: {
    get: vi.fn().mockResolvedValue({ data: {} }),
    post: vi.fn().mockResolvedValue({ data: {} }),
    put: vi.fn().mockResolvedValue({ data: {} }),
    delete: vi.fn().mockResolvedValue({ data: {} }),
  },
}));

describe('Watchlist', () => {
  const emptyWatchlistState = mergeWithDefaults({
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

  const populatedWatchlistState = mergeWithDefaults({
    portfolio: {
      positions: [],
      transactions: [],
      metrics: null,
      watchlist: {
        id: 1,
        name: 'Default',
        description: null,
        is_public: false,
        user_id: 1,
        items: [
          {
            id: 1,
            watchlist_id: 1,
            stock_id: 1,
            added_at: '2024-01-01',
            target_price: 200,
            notes: 'Test note',
            alert_enabled: true,
            symbol: 'AAPL',
            company_name: 'Apple Inc.',
            current_price: 175.50,
            price_change: 2.30,
            price_change_percent: 1.33,
          },
          {
            id: 2,
            watchlist_id: 1,
            stock_id: 2,
            added_at: '2024-01-01',
            target_price: null,
            notes: null,
            alert_enabled: false,
            symbol: 'MSFT',
            company_name: 'Microsoft Corp.',
            current_price: 380.20,
            price_change: -1.50,
            price_change_percent: -0.39,
          },
        ],
      },
      watchlistLoading: false,
      watchlistError: null,
      isLoading: false,
      error: null,
      lastUpdated: null,
    },
  });

  it('renders the watchlist page title', () => {
    renderWithProviders(<Watchlist />, { preloadedState: emptyWatchlistState });

    expect(screen.getByText('Watchlist')).toBeInTheDocument();
  });

  it('renders add stock button', () => {
    renderWithProviders(<Watchlist />, { preloadedState: emptyWatchlistState });

    expect(screen.getByRole('button', { name: /add stock/i })).toBeInTheDocument();
  });

  it('renders refresh button', () => {
    renderWithProviders(<Watchlist />, { preloadedState: emptyWatchlistState });

    expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument();
  });

  it('renders search field', () => {
    renderWithProviders(<Watchlist />, { preloadedState: emptyWatchlistState });

    expect(screen.getByPlaceholderText(/search stocks/i)).toBeInTheDocument();
  });

  it('renders summary cards', () => {
    renderWithProviders(<Watchlist />, { preloadedState: emptyWatchlistState });

    expect(screen.getByText('Total Stocks')).toBeInTheDocument();
    expect(screen.getByText('Gainers')).toBeInTheDocument();
    expect(screen.getByText('Losers')).toBeInTheDocument();
    expect(screen.getByText('Alerts Active')).toBeInTheDocument();
  });

  it('renders empty state when watchlist is empty', () => {
    renderWithProviders(<Watchlist />, { preloadedState: emptyWatchlistState });

    expect(screen.getByText('Your watchlist is empty')).toBeInTheDocument();
  });

  it('renders table headers when items exist', () => {
    renderWithProviders(<Watchlist />, { preloadedState: populatedWatchlistState });

    expect(screen.getByText('Symbol')).toBeInTheDocument();
    expect(screen.getByText('Company')).toBeInTheDocument();
    expect(screen.getByText('Price')).toBeInTheDocument();
    expect(screen.getByText('Change')).toBeInTheDocument();
    expect(screen.getByText('Target Price')).toBeInTheDocument();
  });

  it('renders stock symbols in the table', () => {
    renderWithProviders(<Watchlist />, { preloadedState: populatedWatchlistState });

    expect(screen.getByRole('button', { name: 'AAPL' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'MSFT' })).toBeInTheDocument();
  });

  it('renders company names in the table', () => {
    renderWithProviders(<Watchlist />, { preloadedState: populatedWatchlistState });

    expect(screen.getByText('Apple Inc.')).toBeInTheDocument();
    expect(screen.getByText('Microsoft Corp.')).toBeInTheDocument();
  });

  it('shows correct gainers count', () => {
    renderWithProviders(<Watchlist />, { preloadedState: populatedWatchlistState });

    // Gainers: 1 (AAPL positive), Losers: 1 (MSFT negative), Alerts: 1 (AAPL)
    const gainersLabel = screen.getByText('Gainers');
    const gainersCard = gainersLabel.closest('.MuiCardContent-root');
    expect(gainersCard).toHaveTextContent('1');
  });

  it('renders error alert when watchlist error exists', () => {
    const errorState = mergeWithDefaults({
      portfolio: {
        positions: [],
        transactions: [],
        metrics: null,
        watchlist: null,
        watchlistLoading: false,
        watchlistError: 'Failed to load watchlist',
        isLoading: false,
        error: null,
        lastUpdated: null,
      },
    });

    renderWithProviders(<Watchlist />, { preloadedState: errorState });

    // BUG-3 fix: watchlist errors now render a friendly empty-state notice
    // instead of the raw error string.
    expect(screen.getByText(/your watchlist isn.t available yet/i)).toBeInTheDocument();
  });
});
