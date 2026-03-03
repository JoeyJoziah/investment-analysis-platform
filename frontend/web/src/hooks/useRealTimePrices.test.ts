import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

// Mock env module
vi.mock('../utils/env', () => ({
  env: { WS_URL: 'ws://test:8000' },
}));

// ---------------------------------------------------------------------------
// WebSocket mock
// ---------------------------------------------------------------------------
type WSHandler = (event: any) => void;

class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  url: string;

  onopen: WSHandler | null = null;
  onclose: WSHandler | null = null;
  onmessage: WSHandler | null = null;
  onerror: WSHandler | null = null;

  sent: string[] = [];

  constructor(url: string) {
    this.url = url;
    MockWebSocket._instances.push(this);
  }

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.({} as CloseEvent);
  }

  // Test helpers
  simulateOpen() {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.({} as Event);
  }

  simulateMessage(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent);
  }

  simulateClose() {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.({} as CloseEvent);
  }

  simulateError() {
    this.onerror?.({} as Event);
  }

  static _instances: MockWebSocket[] = [];
  static reset() {
    MockWebSocket._instances = [];
  }
  static get latest() {
    return MockWebSocket._instances[MockWebSocket._instances.length - 1];
  }
}

// Assign to global before importing the hook
(globalThis as any).WebSocket = MockWebSocket;

import { useRealTimePrices } from './useRealTimePrices';

describe('useRealTimePrices', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    MockWebSocket.reset();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  const defaultProps = {
    tickers: ['AAPL', 'MSFT'],
    throttleMs: 0, // disable throttle for tests
  };

  it('creates a WebSocket connection on mount', () => {
    renderHook(() => useRealTimePrices(defaultProps));

    expect(MockWebSocket._instances.length).toBeGreaterThanOrEqual(1);
    expect(MockWebSocket.latest.url).toBe('ws://test:8000');
  });

  it('starts with isConnected false', () => {
    const { result } = renderHook(() => useRealTimePrices(defaultProps));
    expect(result.current.isConnected).toBe(false);
  });

  it('sets isConnected true on open', () => {
    const { result } = renderHook(() => useRealTimePrices(defaultProps));

    act(() => {
      MockWebSocket.latest.simulateOpen();
    });

    expect(result.current.isConnected).toBe(true);
  });

  it('sends subscribe message on open', () => {
    renderHook(() => useRealTimePrices(defaultProps));

    act(() => {
      MockWebSocket.latest.simulateOpen();
    });

    const messages = MockWebSocket.latest.sent.map((m) => JSON.parse(m));
    const subscribes = messages.filter((m: any) => m.action === 'subscribe');
    expect(subscribes.length).toBeGreaterThanOrEqual(1);
    expect(subscribes[0].tickers).toEqual(['AAPL', 'MSFT']);
  });

  it('calls onPriceUpdate when price_update message arrives', () => {
    const onPriceUpdate = vi.fn();
    renderHook(() =>
      useRealTimePrices({ ...defaultProps, onPriceUpdate })
    );

    act(() => {
      MockWebSocket.latest.simulateOpen();
    });

    act(() => {
      MockWebSocket.latest.simulateMessage({
        type: 'price_update',
        ticker: 'AAPL',
        price: 180.5,
        change: 1.2,
        changePercent: 0.67,
        volume: 50000000,
      });
    });

    expect(onPriceUpdate).toHaveBeenCalledTimes(1);
    expect(onPriceUpdate).toHaveBeenCalledWith(
      expect.objectContaining({
        ticker: 'AAPL',
        price: 180.5,
      })
    );
  });

  it('ignores non-price_update messages', () => {
    const onPriceUpdate = vi.fn();
    renderHook(() =>
      useRealTimePrices({ ...defaultProps, onPriceUpdate })
    );

    act(() => {
      MockWebSocket.latest.simulateOpen();
      MockWebSocket.latest.simulateMessage({ type: 'heartbeat' });
    });

    expect(onPriceUpdate).not.toHaveBeenCalled();
  });

  it('ignores malformed messages', () => {
    const onPriceUpdate = vi.fn();
    renderHook(() =>
      useRealTimePrices({ ...defaultProps, onPriceUpdate })
    );

    act(() => {
      MockWebSocket.latest.simulateOpen();
      // Send raw string that will fail JSON.parse
      MockWebSocket.latest.onmessage?.({ data: 'not-json' } as MessageEvent);
    });

    expect(onPriceUpdate).not.toHaveBeenCalled();
  });

  it('throttles rapid updates for same ticker', () => {
    const onPriceUpdate = vi.fn();
    renderHook(() =>
      useRealTimePrices({ ...defaultProps, throttleMs: 1000, onPriceUpdate })
    );

    act(() => {
      MockWebSocket.latest.simulateOpen();
    });

    act(() => {
      MockWebSocket.latest.simulateMessage({
        type: 'price_update', ticker: 'AAPL', price: 180,
        change: 0, changePercent: 0, volume: 0,
      });
    });
    expect(onPriceUpdate).toHaveBeenCalledTimes(1);

    // Second update within throttle window
    act(() => {
      MockWebSocket.latest.simulateMessage({
        type: 'price_update', ticker: 'AAPL', price: 181,
        change: 0, changePercent: 0, volume: 0,
      });
    });
    // Should be throttled
    expect(onPriceUpdate).toHaveBeenCalledTimes(1);

    // Advance past throttle window
    act(() => {
      vi.advanceTimersByTime(1001);
    });

    act(() => {
      MockWebSocket.latest.simulateMessage({
        type: 'price_update', ticker: 'AAPL', price: 182,
        change: 0, changePercent: 0, volume: 0,
      });
    });
    expect(onPriceUpdate).toHaveBeenCalledTimes(2);
  });

  it('sets isConnected false on close', () => {
    const { result } = renderHook(() => useRealTimePrices(defaultProps));

    act(() => {
      MockWebSocket.latest.simulateOpen();
    });
    expect(result.current.isConnected).toBe(true);

    act(() => {
      MockWebSocket.latest.simulateClose();
    });
    expect(result.current.isConnected).toBe(false);
  });

  it('attempts reconnection on close with autoReconnect', () => {
    renderHook(() =>
      useRealTimePrices({ ...defaultProps, autoReconnect: true, maxReconnectAttempts: 3 })
    );

    const initialCount = MockWebSocket._instances.length;

    act(() => {
      MockWebSocket.latest.simulateOpen();
    });
    act(() => {
      MockWebSocket.latest.simulateClose();
    });

    // Advance past first reconnect delay (1000ms * 2^0 = 1000ms)
    act(() => {
      vi.advanceTimersByTime(1100);
    });

    expect(MockWebSocket._instances.length).toBeGreaterThan(initialCount);
  });

  it('does not reconnect when autoReconnect is false', () => {
    renderHook(() =>
      useRealTimePrices({ ...defaultProps, autoReconnect: false })
    );

    const countAfterMount = MockWebSocket._instances.length;

    act(() => {
      MockWebSocket.latest.simulateOpen();
    });
    act(() => {
      MockWebSocket.latest.simulateClose();
    });
    act(() => {
      vi.advanceTimersByTime(5000);
    });

    expect(MockWebSocket._instances.length).toBe(countAfterMount);
  });

  it('subscribe sends message when connected', () => {
    const { result } = renderHook(() => useRealTimePrices(defaultProps));

    act(() => {
      MockWebSocket.latest.simulateOpen();
    });

    const sentBefore = MockWebSocket.latest.sent.length;

    act(() => {
      result.current.subscribe(['GOOGL']);
    });

    const newMessages = MockWebSocket.latest.sent.slice(sentBefore);
    const parsed = newMessages.map((m) => JSON.parse(m));
    expect(parsed).toContainEqual(
      expect.objectContaining({ action: 'subscribe', tickers: ['GOOGL'] })
    );
  });

  it('unsubscribe sends message when connected', () => {
    const { result } = renderHook(() => useRealTimePrices(defaultProps));

    act(() => {
      MockWebSocket.latest.simulateOpen();
    });

    const sentBefore = MockWebSocket.latest.sent.length;

    act(() => {
      result.current.unsubscribe(['AAPL']);
    });

    const newMessages = MockWebSocket.latest.sent.slice(sentBefore);
    const parsed = newMessages.map((m) => JSON.parse(m));
    expect(parsed).toContainEqual(
      expect.objectContaining({ action: 'unsubscribe', tickers: ['AAPL'] })
    );
  });

  it('reconnect resets attempts and creates new connection', () => {
    const { result } = renderHook(() => useRealTimePrices(defaultProps));

    act(() => {
      MockWebSocket.latest.simulateOpen();
    });

    const countBefore = MockWebSocket._instances.length;

    act(() => {
      result.current.reconnect();
    });

    // Should have created a new WebSocket after closing the old one
    expect(MockWebSocket._instances.length).toBeGreaterThan(countBefore);
  });

  it('updates lastUpdate on price_update', () => {
    const { result } = renderHook(() =>
      useRealTimePrices({ ...defaultProps })
    );

    expect(result.current.lastUpdate).toBeNull();

    act(() => {
      MockWebSocket.latest.simulateOpen();
    });

    act(() => {
      MockWebSocket.latest.simulateMessage({
        type: 'price_update', ticker: 'AAPL', price: 180,
        change: 0, changePercent: 0, volume: 0,
      });
    });

    expect(result.current.lastUpdate).toBeInstanceOf(Date);
  });

  it('cleans up WebSocket on unmount', () => {
    const { unmount } = renderHook(() => useRealTimePrices(defaultProps));
    const ws = MockWebSocket.latest;

    act(() => {
      ws.simulateOpen();
    });

    unmount();

    expect(ws.readyState).toBe(MockWebSocket.CLOSED);
  });
});
