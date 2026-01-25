/**
 * API Configuration
 * Centralized configuration for all API endpoints
 */

// Get environment variables with fallbacks
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

export const apiConfig = {
  baseURL: API_URL,
  wsURL: WS_URL,
  timeout: 30000, // 30 seconds
  
  endpoints: {
    // Authentication
    auth: {
      login: '/api/auth/login',
      logout: '/api/auth/logout',
      refresh: '/api/auth/refresh',
      register: '/api/auth/register',
      profile: '/api/auth/profile',
    },
    
    // Stocks
    stocks: {
      list: '/api/stocks',
      detail: (ticker: string) => `/api/stocks/${ticker}`,
      search: '/api/stocks/search',
      trending: '/api/stocks/trending',
    },
    
    // Market Data
    market: {
      overview: '/api/market/overview',
      indices: '/api/market/indices',
      sectors: '/api/market/sectors',
      movers: '/api/market/movers',
    },
    
    // Analysis
    analysis: {
      technical: (ticker: string) => `/api/analysis/technical/${ticker}`,
      fundamental: (ticker: string) => `/api/analysis/fundamental/${ticker}`,
      sentiment: (ticker: string) => `/api/analysis/sentiment/${ticker}`,
      prediction: (ticker: string) => `/api/analysis/prediction/${ticker}`,
    },
    
    // Recommendations
    recommendations: {
      list: '/api/recommendations',
      detail: (id: string) => `/api/recommendations/${id}`,
      active: '/api/recommendations/active',
      history: '/api/recommendations/history',
    },
    
    // Portfolio
    portfolio: {
      list: '/api/portfolio',
      positions: '/api/portfolio/positions',
      transactions: '/api/portfolio/transactions',
      performance: '/api/portfolio/performance',
      add: '/api/portfolio/add',
      remove: '/api/portfolio/remove',
    },
    
    // Watchlist - New API endpoints
    watchlist: {
      // Get all user watchlists
      list: '/api/watchlists',
      // Create a new watchlist
      create: '/api/watchlists',
      // Get specific watchlist with items
      get: (watchlistId: number) => `/api/watchlists/${watchlistId}`,
      // Update a watchlist
      update: (watchlistId: number) => `/api/watchlists/${watchlistId}`,
      // Delete a watchlist
      delete: (watchlistId: number) => `/api/watchlists/${watchlistId}`,
      // Add item to watchlist
      addItem: (watchlistId: number) => `/api/watchlists/${watchlistId}/items`,
      // Update watchlist item
      updateItem: (watchlistId: number, itemId: number) =>
        `/api/watchlists/${watchlistId}/items/${itemId}`,
      // Remove watchlist item
      removeItem: (watchlistId: number, itemId: number) =>
        `/api/watchlists/${watchlistId}/items/${itemId}`,
      // Default watchlist operations
      default: '/api/watchlists/default',
      // Add symbol to default watchlist
      addToDefault: (symbol: string) => `/api/watchlists/default/symbols/${symbol}`,
      // Remove symbol from default watchlist
      removeFromDefault: (symbol: string) => `/api/watchlists/default/symbols/${symbol}`,
    },
    
    // News
    news: {
      latest: '/api/news/latest',
      byTicker: (ticker: string) => `/api/news/${ticker}`,
      market: '/api/news/market',
    },
    
    // User Settings
    settings: {
      preferences: '/api/settings/preferences',
      notifications: '/api/settings/notifications',
      apiKeys: '/api/settings/api-keys',
    },
    
    // Metrics
    metrics: {
      usage: '/api/metrics/usage',
      costs: '/api/metrics/costs',
      performance: '/api/metrics/performance',
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