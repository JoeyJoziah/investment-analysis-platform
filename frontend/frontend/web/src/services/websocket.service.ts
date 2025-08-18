import { io, Socket } from 'socket.io-client';
import { store } from '../store';
import { setWebSocketConnected, addNotification } from '../store/slices/appSlice';
import { updateQuote } from '../store/slices/stockSlice';
import { updateMarketIndex, updateMarketBreadth } from '../store/slices/marketSlice';
import { updatePosition } from '../store/slices/portfolioSlice';

class WebSocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 5000;

  connect(token?: string) {
    if (this.socket?.connected) {
      return;
    }

    const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';
    
    this.socket = io(wsUrl, {
      auth: {
        token: token || localStorage.getItem('authToken'),
      },
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: this.reconnectDelay,
      reconnectionAttempts: this.maxReconnectAttempts,
    });

    this.setupEventListeners();
  }

  private setupEventListeners() {
    if (!this.socket) return;

    // Connection events
    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      store.dispatch(setWebSocketConnected(true));
      this.reconnectAttempts = 0;
      
      // Subscribe to relevant channels
      this.subscribeToChannels();
    });

    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      store.dispatch(setWebSocketConnected(false));
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.reconnectAttempts++;
      
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        store.dispatch(
          addNotification({
            type: 'error',
            message: 'Unable to establish real-time connection. Some features may be limited.',
          })
        );
      }
    });

    // Market data events
    this.socket.on('market:index:update', (data) => {
      store.dispatch(updateMarketIndex(data));
    });

    this.socket.on('market:breadth:update', (data) => {
      store.dispatch(updateMarketBreadth(data));
    });

    // Stock data events
    this.socket.on('stock:quote:update', (data) => {
      const currentState = store.getState();
      if (currentState.stock.selectedTicker === data.ticker) {
        store.dispatch(updateQuote(data));
      }
    });

    this.socket.on('stock:trade', (data) => {
      // Handle real-time trade data
      this.handleTradeUpdate(data);
    });

    // Portfolio events
    this.socket.on('portfolio:position:update', (data) => {
      store.dispatch(updatePosition(data));
    });

    this.socket.on('portfolio:alert', (data) => {
      store.dispatch(
        addNotification({
          type: data.type || 'info',
          message: data.message,
        })
      );
    });

    // News events
    this.socket.on('news:breaking', (data) => {
      store.dispatch(
        addNotification({
          type: 'info',
          message: `Breaking News: ${data.title}`,
        })
      );
    });

    // System events
    this.socket.on('system:maintenance', (data) => {
      store.dispatch(
        addNotification({
          type: 'warning',
          message: data.message || 'System maintenance scheduled',
        })
      );
    });

    this.socket.on('system:announcement', (data) => {
      store.dispatch(
        addNotification({
          type: 'info',
          message: data.message,
        })
      );
    });
  }

  private subscribeToChannels() {
    if (!this.socket) return;

    const state = store.getState();

    // Subscribe to user's portfolio
    if (state.app.user?.id) {
      this.socket.emit('subscribe', {
        channel: 'portfolio',
        userId: state.app.user.id,
      });
    }

    // Subscribe to watchlist stocks
    if (state.portfolio.watchlist.length > 0) {
      this.socket.emit('subscribe', {
        channel: 'stocks',
        tickers: state.portfolio.watchlist,
      });
    }

    // Subscribe to market indices
    this.socket.emit('subscribe', {
      channel: 'market',
      indices: ['SPY', 'QQQ', 'DIA', 'IWM'],
    });

    // Subscribe to selected stock if any
    if (state.stock.selectedTicker) {
      this.subscribeToStock(state.stock.selectedTicker);
    }
  }

  subscribeToStock(ticker: string) {
    if (!this.socket) return;
    
    this.socket.emit('subscribe', {
      channel: 'stock',
      ticker,
    });
  }

  unsubscribeFromStock(ticker: string) {
    if (!this.socket) return;
    
    this.socket.emit('unsubscribe', {
      channel: 'stock',
      ticker,
    });
  }

  subscribeToWatchlist(tickers: string[]) {
    if (!this.socket) return;
    
    this.socket.emit('subscribe', {
      channel: 'stocks',
      tickers,
    });
  }

  private handleTradeUpdate(data: any) {
    // Process real-time trade data
    const { ticker, price, volume, timestamp } = data;
    
    // Update relevant parts of the application
    const currentState = store.getState();
    
    if (currentState.stock.selectedTicker === ticker) {
      store.dispatch(
        updateQuote({
          price,
          volume,
          timestamp,
        })
      );
    }

    // Check for price alerts
    this.checkPriceAlerts(ticker, price);
  }

  private checkPriceAlerts(ticker: string, price: number) {
    // This would normally check against user's configured alerts
    // For now, just a placeholder
    const alerts = []; // Get from state or local storage
    
    alerts.forEach((alert: any) => {
      if (alert.ticker === ticker && alert.active) {
        const triggered = 
          (alert.condition === 'above' && price > alert.value) ||
          (alert.condition === 'below' && price < alert.value);
        
        if (triggered) {
          store.dispatch(
            addNotification({
              type: 'warning',
              message: `Price Alert: ${ticker} is ${alert.condition} ${alert.value}`,
            })
          );
        }
      }
    });
  }

  sendMessage(event: string, data: any) {
    if (!this.socket?.connected) {
      console.warn('WebSocket not connected');
      return;
    }
    
    this.socket.emit(event, data);
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      store.dispatch(setWebSocketConnected(false));
    }
  }

  isConnected(): boolean {
    return this.socket?.connected || false;
  }
}

// Create singleton instance
const wsService = new WebSocketService();

// Auto-connect when user is authenticated
store.subscribe(() => {
  const state = store.getState();
  if (state.app.isAuthenticated && !wsService.isConnected()) {
    wsService.connect();
  } else if (!state.app.isAuthenticated && wsService.isConnected()) {
    wsService.disconnect();
  }
});

export default wsService;