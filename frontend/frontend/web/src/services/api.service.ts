/**
 * API Service
 * Handles all HTTP requests to the backend
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosError } from 'axios';
import { apiConfig, buildApiUrl } from '../config/api.config';

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
    
    // Handle rate limiting
    if (error.response?.status === 429) {
      const retryAfter = error.response.headers['retry-after'];
      console.error(`Rate limited. Retry after ${retryAfter} seconds`);
    }
    
    return Promise.reject(error);
  }
);

// API methods
export const api = {
  // Generic methods
  get: <T = any>(url: string, config?: AxiosRequestConfig) => 
    apiClient.get<T>(url, config),
  
  post: <T = any>(url: string, data?: any, config?: AxiosRequestConfig) => 
    apiClient.post<T>(url, data, config),
  
  put: <T = any>(url: string, data?: any, config?: AxiosRequestConfig) => 
    apiClient.put<T>(url, data, config),
  
  delete: <T = any>(url: string, config?: AxiosRequestConfig) => 
    apiClient.delete<T>(url, config),
  
  // Authentication
  auth: {
    login: (credentials: { username: string; password: string }) =>
      api.post(apiConfig.endpoints.auth.login, credentials),
    
    logout: () =>
      api.post(apiConfig.endpoints.auth.logout),
    
    register: (userData: any) =>
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