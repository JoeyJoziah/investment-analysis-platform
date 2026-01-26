import { useEffect, useRef, useCallback, useState } from 'react';
import { useDispatch } from 'react-redux';

interface PriceUpdate {
  ticker: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: string;
}

interface UseRealTimePricesOptions {
  tickers: string[];
  onPriceUpdate?: (update: PriceUpdate) => void;
  throttleMs?: number;
  autoReconnect?: boolean;
  maxReconnectAttempts?: number;
}

interface UseRealTimePricesReturn {
  isConnected: boolean;
  lastUpdate: Date | null;
  subscribe: (tickers: string[]) => void;
  unsubscribe: (tickers: string[]) => void;
  reconnect: () => void;
}

/**
 * useRealTimePrices - WebSocket hook for real-time price updates
 *
 * Features:
 * - Automatic reconnection with exponential backoff
 * - Throttled updates to prevent UI thrashing
 * - Subscription management
 * - Connection status tracking
 *
 * Usage:
 * ```tsx
 * const { isConnected, lastUpdate } = useRealTimePrices({
 *   tickers: ['AAPL', 'MSFT', 'GOOGL'],
 *   onPriceUpdate: (update) => dispatch(updatePrice(update)),
 *   throttleMs: 1000,
 * });
 * ```
 */
export function useRealTimePrices({
  tickers,
  onPriceUpdate,
  throttleMs = 1000,
  autoReconnect = true,
  maxReconnectAttempts = 10,
}: UseRealTimePricesOptions): UseRealTimePricesReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastUpdateTimeRef = useRef<Record<string, number>>({});

  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // WebSocket URL from environment
  const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';

  // Throttle check
  const shouldUpdate = useCallback(
    (ticker: string): boolean => {
      const now = Date.now();
      const lastTime = lastUpdateTimeRef.current[ticker] || 0;
      if (now - lastTime >= throttleMs) {
        lastUpdateTimeRef.current[ticker] = now;
        return true;
      }
      return false;
    },
    [throttleMs]
  );

  // Handle incoming message
  const handleMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'price_update' && data.ticker) {
          if (shouldUpdate(data.ticker)) {
            const update: PriceUpdate = {
              ticker: data.ticker,
              price: data.price,
              change: data.change,
              changePercent: data.changePercent,
              volume: data.volume,
              timestamp: data.timestamp || new Date().toISOString(),
            };

            setLastUpdate(new Date());
            onPriceUpdate?.(update);
          }
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    },
    [shouldUpdate, onPriceUpdate]
  );

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        setIsConnected(true);
        reconnectAttemptsRef.current = 0;

        // Subscribe to tickers
        if (tickers.length > 0) {
          wsRef.current?.send(
            JSON.stringify({
              action: 'subscribe',
              tickers,
            })
          );
        }
      };

      wsRef.current.onmessage = handleMessage;

      wsRef.current.onclose = () => {
        setIsConnected(false);

        if (autoReconnect && reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = Math.min(
            1000 * Math.pow(2, reconnectAttemptsRef.current),
            30000
          );
          reconnectAttemptsRef.current += 1;

          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
    }
  }, [wsUrl, tickers, handleMessage, autoReconnect, maxReconnectAttempts]);

  // Subscribe to additional tickers
  const subscribe = useCallback((newTickers: string[]) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          action: 'subscribe',
          tickers: newTickers,
        })
      );
    }
  }, []);

  // Unsubscribe from tickers
  const unsubscribe = useCallback((tickersToRemove: string[]) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          action: 'unsubscribe',
          tickers: tickersToRemove,
        })
      );
    }
  }, []);

  // Manual reconnect
  const reconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    reconnectAttemptsRef.current = 0;
    wsRef.current?.close();
    connect();
  }, [connect]);

  // Connect on mount
  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      wsRef.current?.close();
    };
  }, [connect]);

  // Update subscriptions when tickers change
  useEffect(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN && tickers.length > 0) {
      wsRef.current.send(
        JSON.stringify({
          action: 'subscribe',
          tickers,
        })
      );
    }
  }, [tickers]);

  return {
    isConnected,
    lastUpdate,
    subscribe,
    unsubscribe,
    reconnect,
  };
}

export default useRealTimePrices;
