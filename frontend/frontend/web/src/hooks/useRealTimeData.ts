import { useEffect, useCallback, useRef } from 'react';
import { useAppDispatch, useAppSelector } from './redux';
import wsService from '../services/websocket.service';
import { updateQuote, updateTechnicalIndicators } from '../store/slices/stockSlice';
import { updatePosition, updatePortfolioMetrics } from '../store/slices/portfolioSlice';
import { updateMarketIndex } from '../store/slices/marketSlice';
import { addNotification } from '../store/slices/appSlice';

interface UseRealTimeDataOptions {
  subscribeTo?: string[]; // Array of tickers to subscribe to
  enableMarketData?: boolean;
  enablePortfolioUpdates?: boolean;
  enablePriceAlerts?: boolean;
}

export const useRealTimeData = (options: UseRealTimeDataOptions = {}) => {
  const dispatch = useAppDispatch();
  const { isAuthenticated } = useAppSelector(state => state.app);
  const { selectedTicker } = useAppSelector(state => state.stock);
  const { positions } = useAppSelector(state => state.portfolio);
  
  const subscribedTickers = useRef<Set<string>>(new Set());
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const {
    subscribeTo = [],
    enableMarketData = true,
    enablePortfolioUpdates = true,
    enablePriceAlerts = true,
  } = options;

  // Handle connection establishment
  const initializeConnection = useCallback(() => {
    if (isAuthenticated && !wsService.isConnected()) {
      try {
        wsService.connect();
        reconnectAttempts.current = 0;
        
        dispatch(addNotification({
          type: 'success',
          message: 'Real-time data connection established',
        }));
      } catch (error) {
        console.error('Failed to initialize WebSocket connection:', error);
        
        // Attempt reconnection with exponential backoff
        if (reconnectAttempts.current < maxReconnectAttempts) {
          const delay = Math.pow(2, reconnectAttempts.current) * 1000;
          setTimeout(() => {
            reconnectAttempts.current++;
            initializeConnection();
          }, delay);
        } else {
          dispatch(addNotification({
            type: 'error',
            message: 'Unable to establish real-time connection',
          }));
        }
      }
    }
  }, [isAuthenticated, dispatch]);

  // Subscribe to specific tickers
  const subscribeToTickers = useCallback((tickers: string[]) => {
    if (!wsService.isConnected()) return;

    tickers.forEach(ticker => {
      if (!subscribedTickers.current.has(ticker)) {
        wsService.subscribeToStock(ticker);
        subscribedTickers.current.add(ticker);
      }
    });
  }, []);

  // Unsubscribe from tickers
  const unsubscribeFromTickers = useCallback((tickers: string[]) => {
    if (!wsService.isConnected()) return;

    tickers.forEach(ticker => {
      if (subscribedTickers.current.has(ticker)) {
        wsService.unsubscribeFromStock(ticker);
        subscribedTickers.current.delete(ticker);
      }
    });
  }, []);

  // Subscribe to portfolio tickers
  useEffect(() => {
    if (enablePortfolioUpdates && positions.length > 0) {
      const portfolioTickers = positions.map(p => p.ticker);
      subscribeToTickers(portfolioTickers);

      return () => unsubscribeFromTickers(portfolioTickers);
    }
  }, [positions, enablePortfolioUpdates, subscribeToTickers, unsubscribeFromTickers]);

  // Subscribe to selected ticker
  useEffect(() => {
    if (selectedTicker) {
      subscribeToTickers([selectedTicker]);

      return () => unsubscribeFromTickers([selectedTicker]);
    }
  }, [selectedTicker, subscribeToTickers, unsubscribeFromTickers]);

  // Subscribe to specific tickers from options
  useEffect(() => {
    if (subscribeTo.length > 0) {
      subscribeToTickers(subscribeTo);

      return () => unsubscribeFromTickers(subscribeTo);
    }
  }, [subscribeTo, subscribeToTickers, unsubscribeFromTickers]);

  // Initialize connection on mount
  useEffect(() => {
    initializeConnection();
  }, [initializeConnection]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsService.isConnected()) {
        // Unsubscribe from all tickers
        Array.from(subscribedTickers.current).forEach(ticker => {
          wsService.unsubscribeFromStock(ticker);
        });
        subscribedTickers.current.clear();
      }
    };
  }, []);

  return {
    isConnected: wsService.isConnected(),
    subscribeToStock: (ticker: string) => subscribeToTickers([ticker]),
    unsubscribeFromStock: (ticker: string) => unsubscribeFromTickers([ticker]),
    subscribedTickers: Array.from(subscribedTickers.current),
  };
};

// Custom hook for portfolio real-time updates
export const usePortfolioRealTimeData = () => {
  const { positions } = useAppSelector(state => state.portfolio);
  
  return useRealTimeData({
    enablePortfolioUpdates: true,
    enablePriceAlerts: true,
  });
};

// Custom hook for single stock real-time data
export const useStockRealTimeData = (ticker?: string) => {
  return useRealTimeData({
    subscribeTo: ticker ? [ticker] : [],
    enableMarketData: true,
  });
};

// Custom hook for market data
export const useMarketRealTimeData = () => {
  return useRealTimeData({
    subscribeTo: ['SPY', 'QQQ', 'DIA', 'IWM'], // Major market indices
    enableMarketData: true,
    enablePortfolioUpdates: false,
  });
};

export default useRealTimeData;