import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock localStorage before importing slices
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => { store[key] = value; }),
    removeItem: vi.fn((key: string) => { delete store[key]; }),
    clear: vi.fn(() => { store = {}; }),
  };
})();
Object.defineProperty(globalThis, 'localStorage', { value: localStorageMock });

// Mock apiService before importing slices
vi.mock('../../services/api.service', () => ({
  apiService: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}));

vi.mock('../../config/api.config', () => ({
  apiConfig: {
    endpoints: {
      auth: { profile: '/auth/profile', login: '/auth/login', logout: '/auth/logout' },
    },
  },
}));

import appReducer, {
  setThemeMode,
  toggleTheme,
  toggleSidebar,
  setSidebarOpen,
  toggleSearch,
  setSearchOpen,
  addNotification,
  removeNotification,
  clearNotifications,
  setWebSocketConnected,
  initializeApp,
  login,
  logout,
} from './appSlice';

import dashboardReducer, {
  updateMarketSentiment,
  updateCostMetrics,
  addNews,
  clearError as dashboardClearError,
  fetchDashboardData,
} from './dashboardSlice';

import portfolioReducer, {
  updatePosition,
  clearError as portfolioClearError,
  clearWatchlistError,
  fetchPortfolio,
  fetchTransactions,
  addTransaction,
  deletePosition,
  fetchWatchlist,
  addToWatchlist,
  removeFromWatchlist,
} from './portfolioSlice';

import marketReducer, {
  updateMarketIndex,
  updateMarketBreadth,
  clearError as marketClearError,
  fetchMarketOverview,
} from './marketSlice';

import recommendationsReducer, {
  setFilters,
  setSorting,
  setPage,
  selectRecommendation,
  clearSelectedRecommendation,
  updateRecommendation,
  clearError as recClearError,
  fetchRecommendations,
  generateRecommendation,
} from './recommendationsSlice';

import stockReducer, {
  selectStock,
  updateQuote,
  clearSearchResults,
  clearError as stockClearError,
  fetchStockData,
  fetchStockChart,
  searchStocks,
} from './stockSlice';

// ---------------------------------------------------------------------------
// appSlice
// ---------------------------------------------------------------------------
describe('appSlice', () => {
  beforeEach(() => {
    localStorageMock.clear();
    vi.clearAllMocks();
  });

  it('returns initial state', () => {
    const state = appReducer(undefined, { type: 'unknown' });
    expect(state.isInitialized).toBe(false);
    expect(state.isAuthenticated).toBe(false);
    expect(state.user).toBeNull();
    expect(state.sidebarOpen).toBe(true);
    expect(state.searchOpen).toBe(false);
    expect(state.notifications).toEqual([]);
    expect(state.webSocketConnected).toBe(false);
  });

  it('setThemeMode updates theme and persists', () => {
    const state = appReducer(undefined, setThemeMode('light'));
    expect(state.themeMode).toBe('light');
    expect(localStorageMock.setItem).toHaveBeenCalledWith('themeMode', 'light');
  });

  it('toggleTheme switches between light and dark', () => {
    let state = appReducer(undefined, { type: 'unknown' });
    // Default is 'dark' (localStorage empty)
    state = appReducer(state, toggleTheme());
    expect(state.themeMode).toBe('light');
    state = appReducer(state, toggleTheme());
    expect(state.themeMode).toBe('dark');
  });

  it('toggleSidebar toggles state', () => {
    let state = appReducer(undefined, { type: 'unknown' });
    expect(state.sidebarOpen).toBe(true);
    state = appReducer(state, toggleSidebar());
    expect(state.sidebarOpen).toBe(false);
    state = appReducer(state, toggleSidebar());
    expect(state.sidebarOpen).toBe(true);
  });

  it('setSidebarOpen sets specific value', () => {
    const state = appReducer(undefined, setSidebarOpen(false));
    expect(state.sidebarOpen).toBe(false);
  });

  it('toggleSearch toggles state', () => {
    let state = appReducer(undefined, { type: 'unknown' });
    expect(state.searchOpen).toBe(false);
    state = appReducer(state, toggleSearch());
    expect(state.searchOpen).toBe(true);
  });

  it('setSearchOpen sets specific value', () => {
    const state = appReducer(undefined, setSearchOpen(true));
    expect(state.searchOpen).toBe(true);
  });

  it('addNotification adds and caps at 10', () => {
    let state = appReducer(undefined, { type: 'unknown' });
    for (let i = 0; i < 12; i++) {
      state = appReducer(state, addNotification({ type: 'info', message: `msg-${i}` }));
    }
    expect(state.notifications.length).toBe(10);
    // Oldest should have been shifted off
    expect(state.notifications[0].message).toBe('msg-2');
    expect(state.notifications[9].message).toBe('msg-11');
  });

  it('removeNotification removes by id', () => {
    let state = appReducer(undefined, { type: 'unknown' });
    state = appReducer(state, addNotification({ type: 'success', message: 'test' }));
    const id = state.notifications[0].id;
    state = appReducer(state, removeNotification(id));
    expect(state.notifications.length).toBe(0);
  });

  it('clearNotifications empties the array', () => {
    let state = appReducer(undefined, { type: 'unknown' });
    state = appReducer(state, addNotification({ type: 'error', message: 'err' }));
    state = appReducer(state, clearNotifications());
    expect(state.notifications).toEqual([]);
  });

  it('setWebSocketConnected sets value', () => {
    const state = appReducer(undefined, setWebSocketConnected(true));
    expect(state.webSocketConnected).toBe(true);
  });

  it('initializeApp.fulfilled sets auth state', () => {
    const user = { id: '1', email: 'a@b.com', name: 'Test' };
    const state = appReducer(undefined, {
      type: initializeApp.fulfilled.type,
      payload: { isAuthenticated: true, user },
    });
    expect(state.isInitialized).toBe(true);
    expect(state.isAuthenticated).toBe(true);
    expect(state.user).toEqual(user);
  });

  it('initializeApp.fulfilled applies user theme preference', () => {
    const state = appReducer(undefined, {
      type: initializeApp.fulfilled.type,
      payload: {
        isAuthenticated: true,
        user: { id: '1', email: 'a@b.com', name: 'T', preferences: { theme: 'light' as const } },
      },
    });
    expect(state.themeMode).toBe('light');
  });

  it('initializeApp.rejected sets initialized but not authenticated', () => {
    const state = appReducer(undefined, { type: initializeApp.rejected.type });
    expect(state.isInitialized).toBe(true);
    expect(state.isAuthenticated).toBe(false);
  });

  it('login.fulfilled sets user', () => {
    const user = { id: '2', email: 'x@y.com', name: 'User' };
    const state = appReducer(undefined, { type: login.fulfilled.type, payload: user });
    expect(state.isAuthenticated).toBe(true);
    expect(state.user).toEqual(user);
  });

  it('logout.fulfilled clears user', () => {
    let state = appReducer(undefined, {
      type: login.fulfilled.type,
      payload: { id: '1', email: 'a@b.com', name: 'T' },
    });
    state = appReducer(state, { type: logout.fulfilled.type });
    expect(state.isAuthenticated).toBe(false);
    expect(state.user).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// dashboardSlice
// ---------------------------------------------------------------------------
describe('dashboardSlice', () => {
  it('returns initial state', () => {
    const state = dashboardReducer(undefined, { type: 'unknown' });
    expect(state.marketOverview).toBeNull();
    expect(state.topRecommendations).toEqual([]);
    expect(state.loading).toBe(false);
    expect(state.error).toBeNull();
  });

  it('updateMarketSentiment sets sentiment', () => {
    const sentiment = { overall: 'Bullish', score: 80, breakdown: { positive: 60, neutral: 30, negative: 10 } };
    const state = dashboardReducer(undefined, updateMarketSentiment(sentiment));
    expect(state.marketSentiment).toEqual(sentiment);
  });

  it('updateCostMetrics sets metrics', () => {
    const metrics = { currentMonthCost: 42 };
    const state = dashboardReducer(undefined, updateCostMetrics(metrics));
    expect(state.costMetrics).toEqual(metrics);
  });

  it('addNews prepends and caps at 20', () => {
    let state = dashboardReducer(undefined, { type: 'unknown' });
    for (let i = 0; i < 22; i++) {
      state = dashboardReducer(state, addNews({ title: `news-${i}` }));
    }
    expect(state.recentNews.length).toBe(20);
    expect(state.recentNews[0].title).toBe('news-21');
  });

  it('clearError resets error', () => {
    let state = dashboardReducer(undefined, {
      type: fetchDashboardData.rejected.type,
      error: { message: 'fail' },
    });
    expect(state.error).toBe('fail');
    state = dashboardReducer(state, dashboardClearError());
    expect(state.error).toBeNull();
  });

  it('fetchDashboardData.pending sets loading', () => {
    const state = dashboardReducer(undefined, { type: fetchDashboardData.pending.type });
    expect(state.loading).toBe(true);
    expect(state.error).toBeNull();
  });

  it('fetchDashboardData.fulfilled populates state', () => {
    const payload = {
      marketOverview: { indices: [], heatmap: [], sectors: [] },
      topRecommendations: [{ ticker: 'AAPL' }],
      portfolioSummary: { totalValue: 100000 },
      recentNews: [{ title: 'News' }],
      marketSentiment: { overall: 'Neutral', score: 50, breakdown: { positive: 33, neutral: 34, negative: 33 } },
      costMetrics: { currentMonthCost: 10 },
    };
    const state = dashboardReducer(undefined, {
      type: fetchDashboardData.fulfilled.type,
      payload,
    });
    expect(state.loading).toBe(false);
    expect(state.marketOverview).toEqual(payload.marketOverview);
    expect(state.topRecommendations).toEqual(payload.topRecommendations);
    expect(state.portfolioSummary).toEqual(payload.portfolioSummary);
  });

  it('fetchDashboardData.rejected sets error', () => {
    const state = dashboardReducer(undefined, {
      type: fetchDashboardData.rejected.type,
      error: { message: 'Network error' },
    });
    expect(state.loading).toBe(false);
    expect(state.error).toBe('Network error');
  });
});

// ---------------------------------------------------------------------------
// portfolioSlice
// ---------------------------------------------------------------------------
describe('portfolioSlice', () => {
  const mockPosition = {
    id: '1', ticker: 'AAPL', companyName: 'Apple', quantity: 10,
    averagePrice: 150, currentPrice: 180, marketValue: 1800,
    totalGain: 300, totalGainPercent: 20, dayGain: 10,
    dayGainPercent: 0.56, sector: 'Tech', lastUpdated: '2024-01-01',
  };

  it('returns initial state', () => {
    const state = portfolioReducer(undefined, { type: 'unknown' });
    expect(state.positions).toEqual([]);
    expect(state.transactions).toEqual([]);
    expect(state.metrics).toBeNull();
    expect(state.isLoading).toBe(false);
    expect(state.error).toBeNull();
    expect(state.watchlist).toBeNull();
  });

  it('updatePosition updates existing position', () => {
    const initial = portfolioReducer(undefined, { type: 'unknown' });
    const withPos = { ...initial, positions: [mockPosition] };
    const updated = { ...mockPosition, currentPrice: 200 };
    const state = portfolioReducer(withPos, updatePosition(updated));
    expect(state.positions[0].currentPrice).toBe(200);
  });

  it('updatePosition ignores non-existent id', () => {
    const initial = portfolioReducer(undefined, { type: 'unknown' });
    const withPos = { ...initial, positions: [mockPosition] };
    const state = portfolioReducer(withPos, updatePosition({ ...mockPosition, id: '99' }));
    expect(state.positions.length).toBe(1);
    expect(state.positions[0].currentPrice).toBe(180);
  });

  it('clearError resets error', () => {
    let state = portfolioReducer(undefined, {
      type: fetchPortfolio.rejected.type,
      error: { message: 'err' },
    });
    state = portfolioReducer(state, portfolioClearError());
    expect(state.error).toBeNull();
  });

  it('clearWatchlistError resets watchlist error', () => {
    let state = portfolioReducer(undefined, { type: 'unknown' });
    state = { ...state, watchlistError: 'bad' };
    state = portfolioReducer(state, clearWatchlistError());
    expect(state.watchlistError).toBeNull();
  });

  it('fetchPortfolio.pending sets loading', () => {
    const state = portfolioReducer(undefined, { type: fetchPortfolio.pending.type });
    expect(state.isLoading).toBe(true);
    expect(state.error).toBeNull();
  });

  it('fetchPortfolio.fulfilled populates positions and metrics', () => {
    const payload = {
      positions: [mockPosition],
      metrics: { totalValue: 1800 },
    };
    const state = portfolioReducer(undefined, {
      type: fetchPortfolio.fulfilled.type,
      payload,
    });
    expect(state.isLoading).toBe(false);
    expect(state.positions).toEqual([mockPosition]);
    expect(state.metrics).toEqual({ totalValue: 1800 });
    expect(state.lastUpdated).toBeTruthy();
  });

  it('fetchPortfolio.rejected sets error', () => {
    const state = portfolioReducer(undefined, {
      type: fetchPortfolio.rejected.type,
      error: { message: 'API down' },
    });
    expect(state.isLoading).toBe(false);
    expect(state.error).toBe('API down');
  });

  it('fetchTransactions.fulfilled sets transactions', () => {
    const txns = [{ id: '1', ticker: 'AAPL', type: 'BUY', quantity: 10, price: 150, totalAmount: 1500, date: '2024-01-01' }];
    const state = portfolioReducer(undefined, {
      type: fetchTransactions.fulfilled.type,
      payload: txns,
    });
    expect(state.transactions).toEqual(txns);
  });

  it('addTransaction.fulfilled prepends transaction', () => {
    const existing = { id: '1', ticker: 'AAPL', type: 'BUY' as const, quantity: 10, price: 150, totalAmount: 1500, date: '2024-01-01' };
    const initial = { ...portfolioReducer(undefined, { type: 'unknown' }), transactions: [existing] };
    const newTx = { id: '2', ticker: 'MSFT', type: 'BUY' as const, quantity: 5, price: 300, totalAmount: 1500, date: '2024-01-02' };
    const state = portfolioReducer(initial, { type: addTransaction.fulfilled.type, payload: newTx });
    expect(state.transactions[0]).toEqual(newTx);
    expect(state.transactions.length).toBe(2);
  });

  it('deletePosition.fulfilled removes position', () => {
    const initial = { ...portfolioReducer(undefined, { type: 'unknown' }), positions: [mockPosition] };
    const state = portfolioReducer(initial, { type: deletePosition.fulfilled.type, payload: '1' });
    expect(state.positions.length).toBe(0);
  });

  it('fetchWatchlist lifecycle works', () => {
    let state = portfolioReducer(undefined, { type: fetchWatchlist.pending.type });
    expect(state.watchlistLoading).toBe(true);

    const watchlist = { id: 1, name: 'Default', items: [], item_count: 0 };
    state = portfolioReducer(state, { type: fetchWatchlist.fulfilled.type, payload: watchlist });
    expect(state.watchlistLoading).toBe(false);
    expect(state.watchlist).toEqual(watchlist);
  });

  it('fetchWatchlist.rejected sets error', () => {
    const state = portfolioReducer(undefined, {
      type: fetchWatchlist.rejected.type,
      payload: 'Not found',
    });
    expect(state.watchlistLoading).toBe(false);
    expect(state.watchlistError).toBe('Not found');
  });

  it('addToWatchlist.fulfilled adds item', () => {
    const watchlist = { id: 1, name: 'Default', items: [], item_count: 0,
      description: null, is_public: false, user_id: 1, created_at: '', updated_at: '' };
    const initial = { ...portfolioReducer(undefined, { type: 'unknown' }), watchlist };
    const newItem = { id: 10, symbol: 'AAPL', company_name: 'Apple' };

    const state = portfolioReducer(initial, { type: addToWatchlist.fulfilled.type, payload: newItem });
    expect(state.watchlist!.items.length).toBe(1);
    expect(state.watchlist!.item_count).toBe(1);
  });

  it('removeFromWatchlist.fulfilled removes item by symbol', () => {
    const item = { id: 10, symbol: 'AAPL', company_name: 'Apple', watchlist_id: 1,
      stock_id: 1, added_at: '', target_price: null, notes: null, alert_enabled: false,
      current_price: null, price_change: null, price_change_percent: null };
    const watchlist = { id: 1, name: 'Default', items: [item], item_count: 1,
      description: null, is_public: false, user_id: 1, created_at: '', updated_at: '' };
    const initial = { ...portfolioReducer(undefined, { type: 'unknown' }), watchlist };

    const state = portfolioReducer(initial, { type: removeFromWatchlist.fulfilled.type, payload: 'AAPL' });
    expect(state.watchlist!.items.length).toBe(0);
    expect(state.watchlist!.item_count).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// marketSlice
// ---------------------------------------------------------------------------
describe('marketSlice', () => {
  it('returns initial state', () => {
    const state = marketReducer(undefined, { type: 'unknown' });
    expect(state.indices).toEqual([]);
    expect(state.topGainers).toEqual([]);
    expect(state.isLoading).toBe(false);
    expect(state.error).toBeNull();
  });

  it('updateMarketIndex adds new index', () => {
    const idx = { symbol: 'SPY', name: 'S&P 500', value: 500, change: 1, changePercent: 0.2,
      volume: 1e6, high: 501, low: 499, previousClose: 499, timestamp: '2024-01-01' };
    const state = marketReducer(undefined, updateMarketIndex(idx));
    expect(state.indices.length).toBe(1);
    expect(state.indices[0].symbol).toBe('SPY');
  });

  it('updateMarketIndex updates existing index', () => {
    const idx = { symbol: 'SPY', name: 'S&P 500', value: 500, change: 1, changePercent: 0.2,
      volume: 1e6, high: 501, low: 499, previousClose: 499, timestamp: '2024-01-01' };
    let state = marketReducer(undefined, updateMarketIndex(idx));
    state = marketReducer(state, updateMarketIndex({ ...idx, value: 510 }));
    expect(state.indices.length).toBe(1);
    expect(state.indices[0].value).toBe(510);
  });

  it('updateMarketBreadth sets breadth data', () => {
    const breadth = { advancers: 300, decliners: 200, unchanged: 50, newHighs: 10,
      newLows: 5, advanceDeclineRatio: 1.5, upVolume: 1e9, downVolume: 5e8, totalVolume: 1.5e9 };
    const state = marketReducer(undefined, updateMarketBreadth(breadth));
    expect(state.marketBreadth).toEqual(breadth);
  });

  it('clearError resets error', () => {
    let state = marketReducer(undefined, {
      type: fetchMarketOverview.rejected.type,
      error: { message: 'timeout' },
    });
    state = marketReducer(state, marketClearError());
    expect(state.error).toBeNull();
  });

  it('fetchMarketOverview.pending sets loading', () => {
    const state = marketReducer(undefined, { type: fetchMarketOverview.pending.type });
    expect(state.isLoading).toBe(true);
  });

  it('fetchMarketOverview.fulfilled populates state', () => {
    const payload = {
      indices: [{ symbol: 'SPY' }],
      topGainers: [{ ticker: 'AAPL' }],
      topLosers: [{ ticker: 'MSFT' }],
      mostActive: [{ ticker: 'TSLA' }],
      marketBreadth: { advancers: 300 },
    };
    const state = marketReducer(undefined, {
      type: fetchMarketOverview.fulfilled.type,
      payload,
    });
    expect(state.isLoading).toBe(false);
    expect(state.indices).toEqual([{ symbol: 'SPY' }]);
    expect(state.topGainers).toEqual([{ ticker: 'AAPL' }]);
    expect(state.lastUpdated).toBeTruthy();
  });

  it('fetchMarketOverview.rejected sets error', () => {
    const state = marketReducer(undefined, {
      type: fetchMarketOverview.rejected.type,
      error: { message: 'Server error' },
    });
    expect(state.error).toBe('Server error');
  });
});

// ---------------------------------------------------------------------------
// recommendationsSlice
// ---------------------------------------------------------------------------
describe('recommendationsSlice', () => {
  const mockRec = {
    ticker: 'AAPL', company_name: 'Apple', action: 'BUY' as const,
    confidence: 85, target_price: 200, current_price: 180, potential_return: 11,
    risk_level: 'LOW' as const, reasoning: 'Strong fundamentals',
    technical_score: 80, fundamental_score: 90, sentiment_score: 75,
    ml_prediction: 0.85, time_horizon: '6M', sector: 'Technology',
    market_cap: 3e12, volume: 5e7, pe_ratio: 28, dividend_yield: 0.5,
    created_at: '2024-01-01T00:00:00Z',
  };

  it('returns initial state', () => {
    const state = recommendationsReducer(undefined, { type: 'unknown' });
    expect(state.recommendations).toEqual([]);
    expect(state.loading).toBe(false);
    expect(state.sortBy).toBe('confidence');
    expect(state.sortOrder).toBe('desc');
  });

  it('setFilters applies action filter', () => {
    const initial = recommendationsReducer(undefined, { type: 'unknown' });
    const withRecs = { ...initial, recommendations: [mockRec, { ...mockRec, ticker: 'MSFT', action: 'SELL' as const }] };
    const state = recommendationsReducer(withRecs, setFilters({ action: 'BUY' }));
    expect(state.filteredRecommendations.length).toBe(1);
    expect(state.filteredRecommendations[0].ticker).toBe('AAPL');
  });

  it('setFilters applies risk level filter', () => {
    const initial = recommendationsReducer(undefined, { type: 'unknown' });
    const withRecs = { ...initial, recommendations: [mockRec, { ...mockRec, ticker: 'MSFT', risk_level: 'HIGH' as const }] };
    const state = recommendationsReducer(withRecs, setFilters({ riskLevel: 'LOW' }));
    expect(state.filteredRecommendations.length).toBe(1);
  });

  it('setFilters applies minConfidence filter', () => {
    const initial = recommendationsReducer(undefined, { type: 'unknown' });
    const withRecs = { ...initial, recommendations: [mockRec, { ...mockRec, ticker: 'MSFT', confidence: 50 }] };
    const state = recommendationsReducer(withRecs, setFilters({ minConfidence: 80 }));
    expect(state.filteredRecommendations.length).toBe(1);
    expect(state.filteredRecommendations[0].ticker).toBe('AAPL');
  });

  it('setSorting changes sort order', () => {
    const initial = recommendationsReducer(undefined, { type: 'unknown' });
    const withRecs = {
      ...initial,
      recommendations: [mockRec, { ...mockRec, ticker: 'MSFT', confidence: 95 }],
    };
    const state = recommendationsReducer(withRecs, setSorting({ sortBy: 'confidence', sortOrder: 'asc' }));
    expect(state.filteredRecommendations[0].ticker).toBe('AAPL'); // 85 < 95
    expect(state.filteredRecommendations[1].ticker).toBe('MSFT');
  });

  it('setPage updates page', () => {
    const state = recommendationsReducer(undefined, setPage(3));
    expect(state.pagination.page).toBe(3);
  });

  it('selectRecommendation sets selected', () => {
    const initial = recommendationsReducer(undefined, { type: 'unknown' });
    const withRecs = { ...initial, recommendations: [mockRec] };
    const state = recommendationsReducer(withRecs, selectRecommendation('AAPL'));
    expect(state.selectedRecommendation).toEqual(mockRec);
  });

  it('selectRecommendation with unknown ticker sets null', () => {
    const state = recommendationsReducer(undefined, selectRecommendation('ZZZZ'));
    expect(state.selectedRecommendation).toBeNull();
  });

  it('clearSelectedRecommendation clears selection', () => {
    const initial = recommendationsReducer(undefined, { type: 'unknown' });
    const withSelected = { ...initial, selectedRecommendation: mockRec };
    const state = recommendationsReducer(withSelected, clearSelectedRecommendation());
    expect(state.selectedRecommendation).toBeNull();
  });

  it('updateRecommendation updates in-place and re-filters', () => {
    const initial = recommendationsReducer(undefined, { type: 'unknown' });
    const withRecs = { ...initial, recommendations: [mockRec] };
    const updated = { ...mockRec, confidence: 99 };
    const state = recommendationsReducer(withRecs, updateRecommendation(updated));
    expect(state.recommendations[0].confidence).toBe(99);
  });

  it('clearError resets error', () => {
    let state = recommendationsReducer(undefined, {
      type: fetchRecommendations.rejected.type,
      error: { message: 'err' },
    });
    state = recommendationsReducer(state, recClearError());
    expect(state.error).toBeNull();
  });

  it('fetchRecommendations.pending sets loading', () => {
    const state = recommendationsReducer(undefined, { type: fetchRecommendations.pending.type });
    expect(state.loading).toBe(true);
  });

  it('fetchRecommendations.fulfilled populates recommendations', () => {
    const state = recommendationsReducer(undefined, {
      type: fetchRecommendations.fulfilled.type,
      payload: { recommendations: [mockRec], total: 1 },
    });
    expect(state.loading).toBe(false);
    expect(state.recommendations.length).toBe(1);
    expect(state.pagination.total).toBe(1);
  });

  it('generateRecommendation.fulfilled adds new recommendation', () => {
    const state = recommendationsReducer(undefined, {
      type: generateRecommendation.fulfilled.type,
      payload: mockRec,
    });
    expect(state.loading).toBe(false);
    expect(state.recommendations[0]).toEqual(mockRec);
  });

  it('generateRecommendation.fulfilled updates existing', () => {
    const initial = recommendationsReducer(undefined, { type: 'unknown' });
    const withRecs = { ...initial, recommendations: [mockRec] };
    const updated = { ...mockRec, confidence: 95 };
    const state = recommendationsReducer(withRecs, {
      type: generateRecommendation.fulfilled.type,
      payload: updated,
    });
    expect(state.recommendations.length).toBe(1);
    expect(state.recommendations[0].confidence).toBe(95);
  });
});

// ---------------------------------------------------------------------------
// stockSlice
// ---------------------------------------------------------------------------
describe('stockSlice', () => {
  it('returns initial state', () => {
    const state = stockReducer(undefined, { type: 'unknown' });
    expect(state.selectedTicker).toBeNull();
    expect(state.quote).toBeNull();
    expect(state.chartData).toBeNull();
    expect(state.isLoading).toBe(false);
    expect(state.searchResults).toEqual([]);
  });

  it('selectStock sets ticker', () => {
    const state = stockReducer(undefined, selectStock('AAPL'));
    expect(state.selectedTicker).toBe('AAPL');
  });

  it('updateQuote merges with existing quote', () => {
    const quote = {
      ticker: 'AAPL', companyName: 'Apple', price: 180, change: 2,
      changePercent: 1.1, volume: 5e7, avgVolume: 4e7, marketCap: 3e12,
      peRatio: 28, week52High: 200, week52Low: 120, dividendYield: 0.5,
      beta: 1.2, eps: 6.5, open: 178, high: 182, low: 177,
      previousClose: 178, timestamp: '2024-01-01',
    };
    const initial = { ...stockReducer(undefined, { type: 'unknown' }), quote };
    const state = stockReducer(initial, updateQuote({ price: 185 }));
    expect(state.quote!.price).toBe(185);
    expect(state.quote!.companyName).toBe('Apple'); // preserved
  });

  it('updateQuote does nothing when no quote exists', () => {
    const state = stockReducer(undefined, updateQuote({ price: 185 }));
    expect(state.quote).toBeNull();
  });

  it('clearSearchResults empties results', () => {
    const initial = { ...stockReducer(undefined, { type: 'unknown' }), searchResults: [{ ticker: 'AAPL' }] };
    const state = stockReducer(initial as any, clearSearchResults());
    expect(state.searchResults).toEqual([]);
  });

  it('clearError resets error', () => {
    let state = stockReducer(undefined, {
      type: fetchStockData.rejected.type,
      error: { message: 'timeout' },
    });
    state = stockReducer(state, stockClearError());
    expect(state.error).toBeNull();
  });

  it('fetchStockData.pending sets loading', () => {
    const state = stockReducer(undefined, { type: fetchStockData.pending.type });
    expect(state.isLoading).toBe(true);
    expect(state.error).toBeNull();
  });

  it('fetchStockData.fulfilled populates all fields', () => {
    const payload = {
      ticker: 'AAPL',
      quote: { ticker: 'AAPL', price: 180 },
      technical: { rsi: 55 },
      fundamental: { revenue: 400e9 },
      news: [{ id: '1', title: 'Apple News' }],
    };
    const state = stockReducer(undefined, {
      type: fetchStockData.fulfilled.type,
      payload,
    });
    expect(state.isLoading).toBe(false);
    expect(state.selectedTicker).toBe('AAPL');
    expect(state.quote).toEqual({ ticker: 'AAPL', price: 180 });
    expect(state.technicalIndicators).toEqual({ rsi: 55 });
    expect(state.fundamentalData).toEqual({ revenue: 400e9 });
    expect(state.news.length).toBe(1);
  });

  it('fetchStockData.rejected sets error', () => {
    const state = stockReducer(undefined, {
      type: fetchStockData.rejected.type,
      error: { message: 'Not found' },
    });
    expect(state.error).toBe('Not found');
  });

  it('fetchStockChart.fulfilled sets chart data', () => {
    const chartData = { ticker: 'AAPL', interval: '1d', data: [{ date: '2024-01-01', close: 180 }] };
    const state = stockReducer(undefined, {
      type: fetchStockChart.fulfilled.type,
      payload: chartData,
    });
    expect(state.chartData).toEqual(chartData);
  });

  it('searchStocks.fulfilled sets results', () => {
    const results = [{ ticker: 'AAPL', name: 'Apple', exchange: 'NASDAQ', type: 'stock' }];
    const state = stockReducer(undefined, {
      type: searchStocks.fulfilled.type,
      payload: results,
    });
    expect(state.searchResults).toEqual(results);
  });
});
