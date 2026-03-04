/**
 * API Service
 * Handles all HTTP requests to the backend
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosError } from 'axios';
import { apiConfig, buildApiUrl } from '../config/api.config';

export interface RegisterUserData {
  username: string;
  email: string;
  password: string;
  fullName?: string;
}

// Create axios instance with default config
const apiClient: AxiosInstance = axios.create({
  baseURL: apiConfig.baseURL,
  timeout: apiConfig.timeout,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle errors and token refresh
apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };
    
    // Handle 401 Unauthorized - try to refresh token
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        const refreshToken = localStorage.getItem('refresh_token');
        if (refreshToken) {
          const response = await axios.post(
            buildApiUrl(apiConfig.endpoints.auth.refresh),
            { refresh_token: refreshToken }
          );
          
          const { access_token } = response.data;
          localStorage.setItem('access_token', access_token);
          
          // Retry original request with new token
          if (originalRequest.headers) {
            originalRequest.headers.Authorization = `Bearer ${access_token}`;
          }
          return apiClient(originalRequest);
        }
      } catch (refreshError) {
        // Refresh failed - redirect to login
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        window.location.href = '/login';
      }
    }
    
    // Handle rate limiting (error propagated to caller via reject below)
    
    return Promise.reject(error);
  }
);

// API methods
export const api = {
  // Generic methods
  get: <T = unknown>(url: string, config?: AxiosRequestConfig) =>
    apiClient.get<T>(url, config),
  
  post: <T = unknown>(url: string, data?: unknown, config?: AxiosRequestConfig) =>
    apiClient.post<T>(url, data, config),

  put: <T = unknown>(url: string, data?: unknown, config?: AxiosRequestConfig) =>
    apiClient.put<T>(url, data, config),
  
  delete: <T = unknown>(url: string, config?: AxiosRequestConfig) =>
    apiClient.delete<T>(url, config),
  
  // Authentication
  auth: {
    login: (credentials: { username: string; password: string }) =>
      api.post(apiConfig.endpoints.auth.login, credentials),
    
    logout: () =>
      api.post(apiConfig.endpoints.auth.logout),
    
    register: (userData: RegisterUserData) =>
      api.post(apiConfig.endpoints.auth.register, userData),
    
    getProfile: () =>
      api.get(apiConfig.endpoints.auth.profile),
  },
  
  // Stocks
  stocks: {
    getList: (params?: { page?: number; limit?: number; sector?: string }) =>
      api.get(apiConfig.endpoints.stocks.list, { params }),
    
    getDetail: (ticker: string) =>
      api.get(apiConfig.endpoints.stocks.detail(ticker)),
    
    search: (query: string) =>
      api.get(apiConfig.endpoints.stocks.search, { params: { q: query } }),
    
    getTrending: () =>
      api.get(apiConfig.endpoints.stocks.trending),
  },
  
  // Analysis
  analysis: {
    getTechnical: (ticker: string) =>
      api.get(apiConfig.endpoints.analysis.technical(ticker)),
    
    getFundamental: (ticker: string) =>
      api.get(apiConfig.endpoints.analysis.fundamental(ticker)),
    
    getSentiment: (ticker: string) =>
      api.get(apiConfig.endpoints.analysis.sentiment(ticker)),
    
    getPrediction: (ticker: string) =>
      api.get(apiConfig.endpoints.analysis.prediction(ticker)),
  },
  
  // Recommendations
  recommendations: {
    getList: (params?: { page?: number; limit?: number }) =>
      api.get(apiConfig.endpoints.recommendations.list, { params }),
    
    getActive: () =>
      api.get(apiConfig.endpoints.recommendations.active),
    
    getDetail: (id: string) =>
      api.get(apiConfig.endpoints.recommendations.detail(id)),
  },
  
  // Portfolio
  portfolio: {
    getPositions: () =>
      api.get(apiConfig.endpoints.portfolio.positions),
    
    getPerformance: () =>
      api.get(apiConfig.endpoints.portfolio.performance),
    
    addPosition: (data: { ticker: string; quantity: number; price: number }) =>
      api.post(apiConfig.endpoints.portfolio.add, data),
    
    removePosition: (ticker: string) =>
      api.delete(apiConfig.endpoints.portfolio.remove, { data: { ticker } }),
  },
  
  // News
  news: {
    getLatest: () =>
      api.get(apiConfig.endpoints.news.latest),
    
    getByTicker: (ticker: string) =>
      api.get(apiConfig.endpoints.news.byTicker(ticker)),
    
    getMarketNews: () =>
      api.get(apiConfig.endpoints.news.market),
  },
  
  // Watchlist
  watchlist: {
    getAll: () =>
      api.get(apiConfig.endpoints.watchlist.list),

    create: (data: { name: string; description?: string }) =>
      api.post(apiConfig.endpoints.watchlist.create, data),

    getDefault: () =>
      api.get(apiConfig.endpoints.watchlist.default),

    get: (watchlistId: number) =>
      api.get(apiConfig.endpoints.watchlist.get(watchlistId)),

    update: (watchlistId: number, data: { name?: string; description?: string }) =>
      api.put(apiConfig.endpoints.watchlist.update(watchlistId), data),

    remove: (watchlistId: number) =>
      api.delete(apiConfig.endpoints.watchlist.delete(watchlistId)),

    addItem: (watchlistId: number, data: { symbol: string; notes?: string }) =>
      api.post(apiConfig.endpoints.watchlist.addItem(watchlistId), data),

    updateItem: (watchlistId: number, itemId: number, data: { notes?: string }) =>
      api.put(apiConfig.endpoints.watchlist.updateItem(watchlistId, itemId), data),

    removeItem: (watchlistId: number, itemId: number) =>
      api.delete(apiConfig.endpoints.watchlist.removeItem(watchlistId, itemId)),

    addToDefault: (symbol: string) =>
      api.post(apiConfig.endpoints.watchlist.addToDefault(symbol)),

    removeFromDefault: (symbol: string) =>
      api.delete(apiConfig.endpoints.watchlist.removeFromDefault(symbol)),
  },

  // Settings
  settings: {
    getPreferences: () =>
      api.get(apiConfig.endpoints.settings.preferences),

    updatePreferences: (data: Record<string, unknown>) =>
      api.put(apiConfig.endpoints.settings.preferences, data),

    getDisplay: () =>
      api.get(apiConfig.endpoints.settings.display),

    updateDisplay: (data: Record<string, unknown>) =>
      api.put(apiConfig.endpoints.settings.display, data),

    getTradingPrefs: () =>
      api.get(apiConfig.endpoints.settings.trading),

    updateTradingPrefs: (data: Record<string, unknown>) =>
      api.put(apiConfig.endpoints.settings.trading, data),

    getNotifications: () =>
      api.get(apiConfig.endpoints.settings.notifications),

    updateNotifications: (data: Record<string, unknown>) =>
      api.put(apiConfig.endpoints.settings.notifications, data),

    reset: () =>
      api.post(apiConfig.endpoints.settings.reset),
  },

  // Trading
  trading: {
    validateOrder: (portfolioId: number, order: {
      symbol: string;
      side: 'buy' | 'sell';
      order_type: 'market' | 'limit' | 'stop' | 'stop_limit';
      quantity: number;
      price?: number;
      stop_price?: number;
    }) =>
      api.post(apiConfig.endpoints.trading.validateOrder, order, {
        params: { portfolio_id: portfolioId },
      }),

    executeTrade: (portfolioId: number, trade: {
      symbol: string;
      side: 'buy' | 'sell';
      order_type: 'market' | 'limit' | 'stop' | 'stop_limit';
      quantity: number;
      price: number;
    }) =>
      api.post(apiConfig.endpoints.trading.execute(portfolioId), trade),

    calculateImpact: (portfolioId: number, data: {
      symbol: string;
      side: 'buy' | 'sell';
      quantity: number;
      price: number;
    }) =>
      api.post(apiConfig.endpoints.trading.impact(portfolioId), data),
  },

  // ML
  ml: {
    predict: (data: { tickers: string[]; model?: string }) =>
      api.post(apiConfig.endpoints.ml.predict, data),

    getModels: () =>
      api.get(apiConfig.endpoints.ml.models),

    detectDrift: (modelName: string) =>
      api.post(apiConfig.endpoints.ml.driftDetect, null, {
        params: { model_name: modelName },
      }),

    getDriftStatus: () =>
      api.get(apiConfig.endpoints.ml.driftStatus),

    getVersions: () =>
      api.get(apiConfig.endpoints.ml.versions),

    promoteModel: (modelName: string, version: string, targetStage?: string) =>
      api.post(apiConfig.endpoints.ml.promote(modelName), null, {
        params: { version, target_stage: targetStage || 'production' },
      }),

    rollbackModel: (modelName: string, targetVersion: string) =>
      api.post(apiConfig.endpoints.ml.rollback(modelName), null, {
        params: { target_version: targetVersion },
      }),

    backtest: (data: {
      tickers: string[];
      start_date: string;
      end_date: string;
      initial_capital?: number;
      benchmark?: string;
    }) =>
      api.post(apiConfig.endpoints.ml.backtest, data),
  },

  // Metrics
  metrics: {
    getUsage: () =>
      api.get(apiConfig.endpoints.metrics.usage),

    getCosts: () =>
      api.get(apiConfig.endpoints.metrics.costs),
  },
};

export default api;
export const apiService = api;