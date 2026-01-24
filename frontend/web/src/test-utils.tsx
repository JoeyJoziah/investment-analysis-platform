import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore, PreloadedState } from '@reduxjs/toolkit';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { BrowserRouter } from 'react-router-dom';
import CssBaseline from '@mui/material/CssBaseline';

// Import reducers
import appReducer from './store/slices/appSlice';
import dashboardReducer from './store/slices/dashboardSlice';
import recommendationsReducer from './store/slices/recommendationsSlice';
import portfolioReducer from './store/slices/portfolioSlice';
import marketReducer from './store/slices/marketSlice';
import stockReducer from './store/slices/stockSlice';
import type { RootState } from './store';

// Create a default theme for tests
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#9c27b0',
    },
    success: {
      main: '#2e7d32',
    },
    error: {
      main: '#d32f2f',
    },
    warning: {
      main: '#ed6c02',
    },
    info: {
      main: '#0288d1',
    },
  },
});

// Create a test store with optional preloaded state
export function setupStore(preloadedState?: PreloadedState<RootState>) {
  return configureStore({
    reducer: {
      app: appReducer,
      dashboard: dashboardReducer,
      recommendations: recommendationsReducer,
      portfolio: portfolioReducer,
      market: marketReducer,
      stock: stockReducer,
    },
    preloadedState,
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware({
        serializableCheck: false,
      }),
  });
}

export type AppStore = ReturnType<typeof setupStore>;

interface ExtendedRenderOptions extends Omit<RenderOptions, 'queries'> {
  preloadedState?: PreloadedState<RootState>;
  store?: AppStore;
}

// Custom render function that wraps component with providers
export function renderWithProviders(
  ui: ReactElement,
  {
    preloadedState = {},
    store = setupStore(preloadedState),
    ...renderOptions
  }: ExtendedRenderOptions = {}
) {
  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <Provider store={store}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <BrowserRouter>{children}</BrowserRouter>
        </ThemeProvider>
      </Provider>
    );
  }

  return { store, ...render(ui, { wrapper: Wrapper, ...renderOptions }) };
}

// Mock data generators
export const mockPortfolioSummary = {
  totalValue: 150000,
  totalCost: 120000,
  totalReturn: 30000,
  totalReturnPercent: 25,
  dayChange: 1500,
  dayChangePercent: 1.01,
  weekChange: 3.5,
  monthChange: 5.2,
  yearChange: 25,
  activePositions: 12,
  performanceHistory: [
    { date: '2024-01-01', value: 100000 },
    { date: '2024-01-15', value: 110000 },
    { date: '2024-02-01', value: 130000 },
    { date: '2024-02-15', value: 150000 },
  ],
  topGainers: [
    { ticker: 'AAPL', name: 'Apple Inc.', change: 500, changePercent: 5.2, value: 10000 },
    { ticker: 'MSFT', name: 'Microsoft Corp.', change: 400, changePercent: 4.1, value: 10200 },
  ],
  topLosers: [
    { ticker: 'META', name: 'Meta Platforms', change: -200, changePercent: -2.1, value: 9300 },
    { ticker: 'NFLX', name: 'Netflix Inc.', change: -150, changePercent: -1.5, value: 9800 },
  ],
  allocation: [
    { sector: 'Technology', value: 35, percentage: 35 },
    { sector: 'Healthcare', value: 20, percentage: 20 },
    { sector: 'Finance', value: 15, percentage: 15 },
    { sector: 'Consumer', value: 15, percentage: 15 },
    { sector: 'Energy', value: 10, percentage: 10 },
    { sector: 'Other', value: 5, percentage: 5 },
  ],
  riskMetrics: {
    sharpeRatio: 1.45,
    beta: 1.12,
    standardDeviation: 15.3,
    maxDrawdown: -8.5,
  },
  diversificationScore: 78,
  cashBalance: 5000,
  marginUsed: 0,
};

export const mockCostMetrics = {
  currentMonthCost: 35.5,
  projectedMonthCost: 42,
  dailyAverage: 1.5,
  monthlyBudget: 50,
  apiUsage: [
    { provider: 'Alpha Vantage', used: 18, limit: 25, cost: 0, resetDate: '2024-02-01' },
    { provider: 'Finnhub', used: 45, limit: 60, cost: 0, resetDate: '2024-02-01' },
    { provider: 'Polygon.io', used: 3, limit: 5, cost: 0, resetDate: '2024-02-01' },
  ],
  costBreakdown: [
    { category: 'API Calls', amount: 15, percentage: 42 },
    { category: 'Compute', amount: 12, percentage: 34 },
    { category: 'Storage', amount: 5, percentage: 14 },
    { category: 'Other', amount: 3.5, percentage: 10 },
  ],
  alerts: [],
  lastUpdated: '2024-01-24T10:30:00Z',
  costTrend: [
    { date: 'Mon', cost: 1.2 },
    { date: 'Tue', cost: 1.5 },
    { date: 'Wed', cost: 1.8 },
    { date: 'Thu', cost: 1.3 },
    { date: 'Fri', cost: 1.6 },
  ],
  savingsMode: false,
  emergencyMode: false,
};

export const mockPosition = {
  id: '1',
  ticker: 'AAPL',
  companyName: 'Apple Inc.',
  quantity: 100,
  averagePrice: 150,
  currentPrice: 175,
  marketValue: 17500,
  totalGain: 2500,
  totalGainPercent: 16.67,
  dayGain: 125,
  dayGainPercent: 0.72,
  sector: 'Technology',
  lastUpdated: '2024-01-24T10:30:00Z',
};

export const mockTransaction = {
  id: '1',
  ticker: 'AAPL',
  type: 'BUY' as const,
  quantity: 100,
  price: 150,
  totalAmount: 15000,
  date: '2024-01-15T10:30:00Z',
  notes: 'Initial purchase',
};

export const mockPortfolioMetrics = {
  totalValue: 150000,
  totalCost: 120000,
  totalGain: 30000,
  totalGainPercent: 25,
  dayGain: 1500,
  dayGainPercent: 1.01,
  cashBalance: 5000,
  buyingPower: 5000,
  diversification: {
    sector: { Technology: 35, Healthcare: 20, Finance: 15 },
    asset: { Stocks: 80, ETFs: 15, Bonds: 5 },
  },
  performance: {
    daily: [{ date: '2024-01-24', value: 150000 }],
    monthly: [{ date: '2024-01', value: 145000 }],
    yearly: [{ date: '2024', value: 150000 }],
  },
  riskMetrics: {
    sharpeRatio: 1.45,
    beta: 1.12,
    alpha: 0.05,
    standardDeviation: 15.3,
    maxDrawdown: -8.5,
  },
};

// Default initial states for all slices (to ensure proper state structure)
export const defaultInitialState = {
  app: {
    user: null,
    isAuthenticated: false,
    theme: 'light',
    notifications: [],
    sidebarOpen: true,
    loading: false,
    error: null,
  },
  dashboard: {
    marketOverview: null,
    topRecommendations: [],
    portfolioSummary: null,
    recentNews: [],
    marketSentiment: null,
    costMetrics: null,
    loading: false,
    error: null,
  },
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
    pagination: {
      page: 1,
      limit: 20,
      total: 0,
    },
    loading: false,
    error: null,
  },
  portfolio: {
    positions: [],
    transactions: [],
    metrics: null,
    watchlist: [],
    isLoading: false,
    error: null,
    lastUpdated: null,
  },
  market: {
    overview: null,
    indices: [],
    sectors: [],
    watchlist: [],
    loading: false,
    error: null,
  },
  stock: {
    currentStock: null,
    priceHistory: [],
    fundamentals: null,
    technicals: null,
    news: [],
    loading: false,
    error: null,
  },
};

// Helper to merge states with defaults
export function mergeWithDefaults(partialState: Partial<typeof defaultInitialState>) {
  return {
    ...defaultInitialState,
    ...partialState,
    app: { ...defaultInitialState.app, ...(partialState.app || {}) },
    dashboard: { ...defaultInitialState.dashboard, ...(partialState.dashboard || {}) },
    recommendations: { ...defaultInitialState.recommendations, ...(partialState.recommendations || {}) },
    portfolio: { ...defaultInitialState.portfolio, ...(partialState.portfolio || {}) },
    market: { ...defaultInitialState.market, ...(partialState.market || {}) },
    stock: { ...defaultInitialState.stock, ...(partialState.stock || {}) },
  };
}

// Re-export everything from @testing-library/react
export * from '@testing-library/react';
