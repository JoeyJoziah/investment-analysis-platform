/**
 * usePortfolioWebSocket
 *
 * React hook that subscribes to real-time portfolio and price updates using
 * Socket.IO (via the shared WebSocketService singleton).  This replaces the
 * previous native-WebSocket implementation so that the entire frontend speaks
 * a single, consistent protocol.
 *
 * Usage:
 *   const { isConnected, priceUpdates, latency, subscribe, unsubscribe } =
 *     usePortfolioWebSocket(portfolioId, symbols);
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { useAppDispatch } from './redux';
import { addNotification } from '../store/slices/appSlice';
import wsService from '../services/websocket.service';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PriceUpdate {
  symbol: string;
  price: number;
  change?: number;
  change_percent?: number;
  bid?: number;
  ask?: number;
  bid_size?: number;
  ask_size?: number;
  volume?: number;
  timestamp: string;
}

interface PortfolioUpdate {
  portfolio_id: string;
  total_value?: number;
  day_change?: number;
  day_change_percent?: number;
  positions?: unknown[];
  timestamp: string;
}

interface AlertNotification {
  alert_type?: string;
  message: string;
  severity?: string;
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export const usePortfolioWebSocket = (
  portfolioId: string,
  symbols: string[],
  enabled: boolean = true,
) => {
  const dispatch = useAppDispatch();

  const [isConnected, setIsConnected] = useState<boolean>(wsService.isConnected());
  const [priceUpdates, setPriceUpdates] = useState<Map<string, PriceUpdate>>(new Map());
  const [latency, setLatency] = useState<number>(0);

  // Track the symbols we have subscribed so we can unsubscribe on cleanup.
  const subscribedSymbolsRef = useRef<string[]>([]);
  const pingTimestampRef = useRef<number>(0);

  // ------------------------------------------------------------------
  // Connection status sync – poll the service's connected state because
  // Socket.IO fires events on the singleton, not on this hook instance.
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!enabled) return;

    const interval = setInterval(() => {
      const connected = wsService.isConnected();
      setIsConnected(connected);
    }, 1000);

    return () => clearInterval(interval);
  }, [enabled]);

  // ------------------------------------------------------------------
  // Subscribe to Socket.IO events that carry data for this hook.
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!enabled) return;

    // Ensure the Socket.IO connection is live.
    if (!wsService.isConnected()) {
      wsService.connect();
    }

    // -- price_update --
    const handlePriceUpdate = (data: PriceUpdate) => {
      if (!data?.symbol) return;

      // Only process symbols that belong to this hook instance.
      if (!subscribedSymbolsRef.current.includes(data.symbol)) return;

      setPriceUpdates((prev) => {
        const next = new Map(prev);
        next.set(data.symbol, data);
        return next;
      });
    };

    // -- portfolio_update --
    const handlePortfolioUpdate = (data: PortfolioUpdate) => {
      if (data.portfolio_id !== portfolioId) return;

      dispatch(
        addNotification({
          type: 'info',
          message: `Portfolio ${portfolioId} updated`,
        }),
      );
    };

    // -- alert_notification --
    const handleAlertNotification = (data: AlertNotification) => {
      dispatch(
        addNotification({
          type: (data.severity as 'error' | 'warning' | 'info' | 'success') ?? 'info',
          message: data.message,
        }),
      );
    };

    // -- heartbeat (latency measurement via system messages) --
    const handleSystem = (_data: unknown) => {
      const now = Date.now();
      if (pingTimestampRef.current > 0) {
        setLatency(now - pingTimestampRef.current);
      }
      pingTimestampRef.current = now;
    };

    // Register listeners on the underlying Socket.IO socket.
    // The WebSocketService exposes `sendMessage` but not `on`, so we access
    // the internal socket via the service's `sendMessage` + a lightweight
    // event registration helper below.
    _addSocketListener('price_update', handlePriceUpdate);
    _addSocketListener('portfolio_update', handlePortfolioUpdate);
    _addSocketListener('alert_notification', handleAlertNotification);
    _addSocketListener('system', handleSystem);

    return () => {
      _removeSocketListener('price_update', handlePriceUpdate);
      _removeSocketListener('portfolio_update', handlePortfolioUpdate);
      _removeSocketListener('alert_notification', handleAlertNotification);
      _removeSocketListener('system', handleSystem);
    };
  }, [enabled, portfolioId, dispatch]);

  // ------------------------------------------------------------------
  // Subscribe to initial symbols and portfolio once connected.
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!enabled || symbols.length === 0) return;

    // Wait until the socket is connected before emitting subscriptions.
    const trySubscribe = () => {
      if (!wsService.isConnected()) return;

      wsService.sendMessage('subscribe_prices', { symbols });
      subscribedSymbolsRef.current = symbols;

      wsService.sendMessage('subscribe_portfolio', { portfolio_id: portfolioId });

      dispatch(
        addNotification({
          type: 'success',
          message: 'Real-time updates connected',
        }),
      );
    };

    // Attempt immediately, then retry every 500 ms until connected.
    trySubscribe();
    const poll = setInterval(() => {
      if (wsService.isConnected()) {
        trySubscribe();
        clearInterval(poll);
      }
    }, 500);

    return () => clearInterval(poll);
  }, [enabled, portfolioId, symbols, dispatch]);

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  const subscribe = useCallback(
    (newSymbols: string[]) => {
      if (!wsService.isConnected() || newSymbols.length === 0) return;

      wsService.sendMessage('subscribe_prices', { symbols: newSymbols });

      // Merge into the tracked set (immutable update).
      const merged = Array.from(new Set([...subscribedSymbolsRef.current, ...newSymbols]));
      subscribedSymbolsRef.current = merged;
    },
    [],
  );

  const unsubscribe = useCallback(
    (removeSymbols: string[]) => {
      // Socket.IO does not have a built-in "unsubscribe from room" message –
      // the backend handles room membership.  We send a custom event and
      // remove the symbols from our local tracking set.
      if (wsService.isConnected()) {
        wsService.sendMessage('unsubscribe_prices', { symbols: removeSymbols });
      }

      subscribedSymbolsRef.current = subscribedSymbolsRef.current.filter(
        (s) => !removeSymbols.includes(s),
      );

      // Remove stale entries from the price map (immutable update).
      setPriceUpdates((prev) => {
        const next = new Map(prev);
        removeSymbols.forEach((s) => next.delete(s));
        return next;
      });
    },
    [],
  );

  const disconnect = useCallback(() => {
    wsService.disconnect();
    setIsConnected(false);
  }, []);

  return {
    isConnected,
    priceUpdates,
    latency,
    subscribe,
    unsubscribe,
    disconnect,
  };
};

// ---------------------------------------------------------------------------
// Internal helpers – thin wrappers around the socket instance
// ---------------------------------------------------------------------------

/**
 * Access the raw socket.io-client Socket instance held inside the service.
 * The service does not expose it publicly, so we reach for it via a known
 * property name.  Type-cast through `unknown` avoids `any` lint issues.
 */
function _getRawSocket() {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (wsService as unknown as { socket: import('socket.io-client').Socket | null }).socket;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function _addSocketListener(event: string, handler: (...args: any[]) => void) {
  const sock = _getRawSocket();
  if (sock) {
    sock.on(event, handler as Parameters<typeof sock.on>[1]);
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function _removeSocketListener(event: string, handler: (...args: any[]) => void) {
  const sock = _getRawSocket();
  if (sock) {
    sock.off(event, handler as Parameters<typeof sock.off>[1]);
  }
}
