/**
 * Shared types for recommendation card sub-components
 */

export interface Recommendation {
  ticker: string;
  company_name?: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  target_price?: number;
  current_price: number;
  potential_return?: number;
  risk_level?: 'LOW' | 'MEDIUM' | 'HIGH';
  reasoning?: string;
  technical_score?: number;
  fundamental_score?: number;
  sentiment_score?: number;
  ml_prediction?: number;
  time_horizon?: string;
  sector?: string;
  market_cap?: number;
  volume?: number;
  pe_ratio?: number;
  dividend_yield?: number;
  price_history?: Array<{ date: string; price: number }>;
  analyst_ratings?: { buy: number; hold: number; sell: number };
  esg_score?: number;
}

export interface NotificationState {
  open: boolean;
  message: string;
  severity: 'success' | 'error' | 'warning' | 'info';
}
