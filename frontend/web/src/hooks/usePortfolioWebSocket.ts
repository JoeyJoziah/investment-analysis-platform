import { useEffect, useRef, useState, useCallback } from 'react';
import { useAppDispatch } from './redux';
import { addNotification } from '../store/slices/appSlice';

interface WebSocketMessage {
  type: string;
  symbol?: string;
  price?: number;
  bid?: number;
  ask?: number;
  bid_size?: number;
  ask_size?: number;
  timestamp?: string;
  volume?: number;
  change?: number;
  change_percent?: number;
  portfolio_id?: string;
  total_value?: number;
  day_change?: number;
  day_change_percent?: number;
  positions?: any[];
}

interface PriceUpdate {
  symbol: string;
  price: number;
  change?: number;
  change_percent?: number;
  bid?: number;
  ask?: number;
  timestamp: string;
}

export const usePortfolioWebSocket = (
  portfolioId: string,
  symbols: string[],
  enabled: boolean = true
) => {
  const dispatch = useAppDispatch();
  const websocketRef = useRef<WebSocket | null>(null);
  const clientIdRef = useRef<string>('');
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttemptsRef = useRef(5);
  const reconnectDelayRef = useRef(1000);
  const subscriptionTimerRef = useRef<NodeJS.Timeout | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [priceUpdates, setPriceUpdates] = useState<Map<string, PriceUpdate>>(new Map());
  const [latency, setLatency] = useState(0);
  const lastPingRef = useRef<number>(0);

  // Generate unique client ID
  useEffect(() => {
    clientIdRef.current = `portfolio-${portfolioId}-${Date.now()}`;
  }, [portfolioId]);

  // Handle incoming WebSocket messages
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);

      if (message.type === 'price_update' && message.symbol) {
        const update: PriceUpdate = {
          symbol: message.symbol,
          price: message.price || 0,
          change: message.change,
          change_percent: message.change_percent,
          bid: message.bid,
          ask: message.ask,
          timestamp: message.timestamp || new Date().toISOString(),
        };

        setPriceUpdates((prev) => {
          const newUpdates = new Map(prev);
          newUpdates.set(message.symbol!, update);
          return newUpdates;
        });
      } else if (message.type === 'portfolio_update') {
        // Handle portfolio-level updates
        dispatch(
          addNotification({
            type: 'info',
            message: 'Portfolio updated',
          })
        );
      } else if (message.type === 'heartbeat') {
        // Calculate latency from heartbeat
        const now = Date.now();
        setLatency(now - lastPingRef.current);
      } else if (message.type === 'system') {
        // Log system messages
        console.log('WebSocket system:', message);
      } else if (message.type === 'error') {
        dispatch(
          addNotification({
            type: 'error',
            message: 'WebSocket error occurred',
          })
        );
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }, [dispatch]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (!enabled) return;

    try {
      // Determine WebSocket URL
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      const wsUrl = `${protocol}//${host}/api/ws/stream?client_id=${clientIdRef.current}`;

      websocketRef.current = new WebSocket(wsUrl);

      websocketRef.current.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        reconnectAttemptsRef.current = 0;
        reconnectDelayRef.current = 1000;

        dispatch(
          addNotification({
            type: 'success',
            message: 'Real-time updates connected',
          })
        );

        // Subscribe to portfolio symbols
        if (symbols.length > 0) {
          const subscribeMessage = {
            type: 'subscribe',
            symbols: symbols,
          };
          websocketRef.current?.send(JSON.stringify(subscribeMessage));
        }

        // Start heartbeat
        startHeartbeat();
      };

      websocketRef.current.onmessage = handleMessage;

      websocketRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };

      websocketRef.current.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        attemptReconnect();
      };
    } catch (error) {
      console.error('Error connecting to WebSocket:', error);
      attemptReconnect();
    }
  }, [enabled, symbols, dispatch, handleMessage]);

  // Attempt reconnection with exponential backoff
  const attemptReconnect = useCallback(() => {
    if (reconnectAttemptsRef.current < maxReconnectAttemptsRef.current) {
      reconnectAttemptsRef.current++;
      const delay = reconnectDelayRef.current * (2 ** (reconnectAttemptsRef.current - 1));
      const maxDelay = 30000; // Max 30 seconds

      reconnectDelayRef.current = Math.min(delay, maxDelay);

      console.log(
        `Attempting to reconnect in ${reconnectDelayRef.current}ms (attempt ${reconnectAttemptsRef.current})`
      );

      subscriptionTimerRef.current = setTimeout(() => {
        connect();
      }, reconnectDelayRef.current);
    } else {
      dispatch(
        addNotification({
          type: 'error',
          message: 'Failed to connect to real-time updates. Using polling instead.',
        })
      );
    }
  }, [connect, dispatch]);

  // Send heartbeat to keep connection alive
  const startHeartbeat = useCallback(() => {
    const heartbeatInterval = setInterval(() => {
      if (websocketRef.current && isConnected) {
        try {
          lastPingRef.current = Date.now();
          websocketRef.current.send(
            JSON.stringify({
              type: 'heartbeat',
              timestamp: new Date().toISOString(),
            })
          );
        } catch (error) {
          console.error('Error sending heartbeat:', error);
        }
      }
    }, 30000); // Send heartbeat every 30 seconds

    return () => clearInterval(heartbeatInterval);
  }, [isConnected]);

  // Subscribe to additional symbols
  const subscribe = useCallback((newSymbols: string[]) => {
    if (websocketRef.current && isConnected) {
      const subscribeMessage = {
        type: 'subscribe',
        symbols: newSymbols,
      };
      websocketRef.current.send(JSON.stringify(subscribeMessage));
    }
  }, [isConnected]);

  // Unsubscribe from symbols
  const unsubscribe = useCallback((removeSymbols: string[]) => {
    if (websocketRef.current && isConnected) {
      const unsubscribeMessage = {
        type: 'unsubscribe',
        symbols: removeSymbols,
      };
      websocketRef.current.send(JSON.stringify(unsubscribeMessage));
    }
  }, [isConnected]);

  // Disconnect WebSocket
  const disconnect = useCallback(() => {
    if (subscriptionTimerRef.current) {
      clearTimeout(subscriptionTimerRef.current);
    }

    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }

    setIsConnected(false);
  }, []);

  // Initialize WebSocket on mount
  useEffect(() => {
    if (enabled) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [enabled, connect, disconnect]);

  return {
    isConnected,
    priceUpdates,
    latency,
    subscribe,
    unsubscribe,
    disconnect,
  };
};
