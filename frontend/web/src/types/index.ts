// Common types used across the application

export interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  preferences?: UserPreferences;
  createdAt: string;
  updatedAt: string;
}

export interface UserPreferences {
  theme?: 'light' | 'dark';
  defaultView?: string;
  notifications?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
  currency?: string;
  timezone?: string;
  language?: string;
}

export interface Stock {
  ticker: string;
  companyName: string;
  exchange: string;
  sector: string;
  industry: string;
  marketCap: number;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  avgVolume: number;
  high: number;
  low: number;
  open: number;
  previousClose: number;
  week52High: number;
  week52Low: number;
  peRatio?: number;
  eps?: number;
  dividendYield?: number;
  beta?: number;
  lastUpdated: string;
}

export interface ChartData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  adjustedClose?: number;
}

export interface Recommendation {
  id: string;
  ticker: string;
  companyName: string;
  sector: string;
  price: number;
  targetPrice: number;
  recommendation: RecommendationType;
  confidence: number;
  signals: RecommendationSignals;
  reasons: string[];
  risk: RiskLevel;
  timeHorizon: TimeHorizon;
  expectedReturn: number;
  createdAt: string;
  updatedAt: string;
}

export type RecommendationType = 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH';
export type TimeHorizon = 'SHORT' | 'MEDIUM' | 'LONG';

export interface RecommendationSignals {
  technical: number;
  fundamental: number;
  sentiment: number;
  ml_prediction: number;
}

export interface Portfolio {
  id: string;
  userId: string;
  positions: Position[];
  transactions: Transaction[];
  metrics: PortfolioMetrics;
  createdAt: string;
  updatedAt: string;
}

export interface Position {
  id: string;
  ticker: string;
  companyName: string;
  quantity: number;
  averagePrice: number;
  currentPrice: number;
  marketValue: number;
  totalGain: number;
  totalGainPercent: number;
  dayGain: number;
  dayGainPercent: number;
  sector: string;
  lastUpdated: string;
}

export interface Transaction {
  id: string;
  ticker: string;
  type: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  totalAmount: number;
  date: string;
  notes?: string;
}

export interface PortfolioMetrics {
  totalValue: number;
  totalCost: number;
  totalGain: number;
  totalGainPercent: number;
  dayGain: number;
  dayGainPercent: number;
  cashBalance: number;
  buyingPower: number;
  diversification: {
    sector: Record<string, number>;
    asset: Record<string, number>;
  };
  performance: {
    daily: PerformancePoint[];
    monthly: PerformancePoint[];
    yearly: PerformancePoint[];
  };
  riskMetrics: RiskMetrics;
}

export interface PerformancePoint {
  date: string;
  value: number;
  gain?: number;
  gainPercent?: number;
}

export interface RiskMetrics {
  sharpeRatio: number;
  beta: number;
  alpha: number;
  standardDeviation: number;
  maxDrawdown: number;
  valueAtRisk?: number;
  sortino?: number;
}

export interface Alert {
  id: string;
  userId: string;
  ticker: string;
  type: 'price' | 'percent' | 'volume' | 'news';
  condition: 'above' | 'below' | 'equals';
  value: number;
  active: boolean;
  triggered: boolean;
  triggeredAt?: string;
  createdAt: string;
}

export interface NewsArticle {
  id: string;
  title: string;
  summary: string;
  content?: string;
  url: string;
  source: string;
  author?: string;
  publishedAt: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  sentimentScore?: number;
  relevanceScore?: number;
  relatedTickers: string[];
  categories?: string[];
  image?: string;
}

export interface MarketIndex {
  symbol: string;
  name: string;
  value: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  previousClose: number;
  timestamp: string;
}

export interface SectorPerformance {
  sector: string;
  changePercent: number;
  marketCap: number;
  volume: number;
  gainers: number;
  losers: number;
  topStock: {
    ticker: string;
    changePercent: number;
  };
  worstStock?: {
    ticker: string;
    changePercent: number;
  };
}

export interface EconomicEvent {
  date: string;
  time: string;
  event: string;
  country?: string;
  importance: 'high' | 'medium' | 'low';
  actual?: number;
  forecast?: number;
  previous?: number;
  impact?: string;
}

export interface TechnicalIndicator {
  name: string;
  value: number;
  signal?: 'buy' | 'sell' | 'neutral';
  timestamp: string;
}

export interface OptionsContract {
  ticker: string;
  type: 'call' | 'put';
  strike: number;
  expiration: string;
  bid: number;
  ask: number;
  last?: number;
  volume: number;
  openInterest: number;
  impliedVolatility: number;
  inTheMoney: boolean;
  delta?: number;
  gamma?: number;
  theta?: number;
  vega?: number;
}

export interface Screener {
  id: string;
  name: string;
  description?: string;
  criteria: ScreenerCriteria[];
  results?: Stock[];
  createdAt: string;
  updatedAt: string;
}

export interface ScreenerCriteria {
  field: string;
  operator: 'gt' | 'gte' | 'lt' | 'lte' | 'eq' | 'neq' | 'in' | 'nin';
  value: any;
}

export interface ApiError {
  message: string;
  code?: string;
  status?: number;
  details?: any;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

export interface WebSocketMessage {
  event: string;
  data: any;
  timestamp: string;
}

export interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  timestamp: number;
  read?: boolean;
  action?: {
    label: string;
    handler: () => void;
  };
}