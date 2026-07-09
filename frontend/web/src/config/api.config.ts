/**
 * API Configuration
 * Centralized configuration for all API endpoints
 */

import { env } from '../utils/env';

// Get environment variables with fallbacks (Vite-compatible)
const API_URL = env.API_URL;
const WS_URL = env.WS_URL;

export const apiConfig = {
  baseURL: API_URL,
  wsURL: WS_URL,
  timeout: 30000, // 30 seconds
  
  // All endpoints use the canonical /api/v1/ prefix. The backend mounts every
  // router under /api/v1/ (see backend/api/main.py:333-348). There is no v2
  // migration planned; future major versions would mount at /api/v2/.
  endpoints: {
    // Authentication
    auth: {
      login: '/api/v1/auth/login',
      logout: '/api/v1/auth/logout',
      refresh: '/api/v1/auth/refresh',
      register: '/api/v1/auth/register',
      profile: '/api/v1/auth/me',
    },

    // Stocks
    stocks: {
      list: '/api/v1/stocks',
      detail: (ticker: string) => `/api/v1/stocks/${ticker}`,
      search: '/api/v1/stocks/search',
      trending: '/api/v1/stocks/trending',
    },

    // Market Data
    market: {
      overview: '/api/v1/market/overview',
      indices: '/api/v1/market/indices',
      sectors: '/api/v1/market/sectors',
      movers: '/api/v1/market/movers',
    },

    // Analysis
    analysis: {
      technical: (ticker: string) => `/api/v1/analysis/technical/${ticker}`,
      fundamental: (ticker: string) => `/api/v1/analysis/fundamental/${ticker}`,
      sentiment: (ticker: string) => `/api/v1/analysis/sentiment/${ticker}`,
      prediction: (ticker: string) => `/api/v1/analysis/prediction/${ticker}`,
    },

    // Recommendations
    recommendations: {
      list: '/api/v1/recommendations',
      detail: (id: string) => `/api/v1/recommendations/${id}`,
      active: '/api/v1/recommendations/active',
      history: '/api/v1/recommendations/history',
    },

    // Portfolio
    portfolio: {
      list: '/api/v1/portfolio',
      positions: '/api/v1/portfolio/positions',
      transactions: '/api/v1/portfolio/transactions',
      performance: '/api/v1/portfolio/performance',
      add: '/api/v1/portfolio/add',
      remove: '/api/v1/portfolio/remove',
    },

    // Watchlist - New API endpoints
    watchlist: {
      // Get all user watchlists
      list: '/api/v1/watchlists',
      // Create a new watchlist
      create: '/api/v1/watchlists',
      // Get specific watchlist with items
      get: (watchlistId: number) => `/api/v1/watchlists/${watchlistId}`,
      // Update a watchlist
      update: (watchlistId: number) => `/api/v1/watchlists/${watchlistId}`,
      // Delete a watchlist
      delete: (watchlistId: number) => `/api/v1/watchlists/${watchlistId}`,
      // Add item to watchlist
      addItem: (watchlistId: number) => `/api/v1/watchlists/${watchlistId}/items`,
      // Update watchlist item
      updateItem: (watchlistId: number, itemId: number) =>
        `/api/v1/watchlists/${watchlistId}/items/${itemId}`,
      // Remove watchlist item
      removeItem: (watchlistId: number, itemId: number) =>
        `/api/v1/watchlists/${watchlistId}/items/${itemId}`,
      // Default watchlist operations
      default: '/api/v1/watchlists/default',
      // Add symbol to default watchlist
      addToDefault: (symbol: string) => `/api/v1/watchlists/default/symbols/${symbol}`,
      // Remove symbol from default watchlist
      removeFromDefault: (symbol: string) => `/api/v1/watchlists/default/symbols/${symbol}`,
    },

    // News
    news: {
      latest: '/api/v1/news/latest',
      byTicker: (ticker: string) => `/api/v1/news/${ticker}`,
      market: '/api/v1/news/market',
    },

    // User Settings
    settings: {
      preferences: '/api/v1/settings/preferences',
      display: '/api/v1/settings/display',
      trading: '/api/v1/settings/trading',
      notifications: '/api/v1/settings/notifications',
      reset: '/api/v1/settings/reset',
      apiKeys: '/api/v1/settings/api-keys',
    },

    // Trading
    trading: {
      validateOrder: '/api/v1/trading/orders/validate',
      execute: (portfolioId: number) => `/api/v1/trading/orders/${portfolioId}`,
      impact: (portfolioId: number) => `/api/v1/trading/orders/${portfolioId}/impact`,
    },

    // ML
    ml: {
      predict: '/api/v1/ml/predictions',
      models: '/api/v1/ml/models',
      driftDetect: '/api/v1/ml/drift/detect',
      driftStatus: '/api/v1/ml/drift/status',
      versions: '/api/v1/ml/versions',
      promote: (modelName: string) => `/api/v1/ml/versions/${modelName}/promote`,
      rollback: (modelName: string) => `/api/v1/ml/versions/${modelName}/rollback`,
      backtest: '/api/v1/ml/backtest',
    },

    // Metrics
    metrics: {
      usage: '/api/v1/metrics/usage',
      costs: '/api/v1/metrics/costs',
      performance: '/api/v1/metrics/performance',
    },
  },
  
  // WebSocket events
  wsEvents: {
    // Market data events
    QUOTE_UPDATE: 'quote_update',
    TRADE_UPDATE: 'trade_update',
    ORDER_BOOK: 'order_book',
    
    // News events
    NEWS_UPDATE: 'news_update',
    
    // Recommendation events
    NEW_RECOMMENDATION: 'new_recommendation',
    RECOMMENDATION_UPDATE: 'recommendation_update',
    
    // System events
    SYSTEM_STATUS: 'system_status',
    RATE_LIMIT: 'rate_limit',
  },
};

// Helper function to build full URL
export const buildApiUrl = (endpoint: string): string => {
  return `${apiConfig.baseURL}${endpoint}`;
};

// Helper function to build WebSocket URL
export const buildWsUrl = (path: string = ''): string => {
  return `${apiConfig.wsURL}${path}`;
};

export default apiConfig;