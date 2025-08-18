import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { apiService } from '../../services/api.service';

export type ThemeMode = 'light' | 'dark';

interface User {
  id: string;
  email: string;
  name: string;
  preferences?: {
    theme?: ThemeMode;
    defaultView?: string;
    notifications?: boolean;
  };
}

interface AppState {
  isInitialized: boolean;
  isAuthenticated: boolean;
  user: User | null;
  themeMode: ThemeMode;
  sidebarOpen: boolean;
  searchOpen: boolean;
  notifications: Array<{
    id: string;
    type: 'success' | 'error' | 'warning' | 'info';
    message: string;
    timestamp: number;
  }>;
  webSocketConnected: boolean;
}

const initialState: AppState = {
  isInitialized: false,
  isAuthenticated: false,
  user: null,
  themeMode: (localStorage.getItem('themeMode') as ThemeMode) || 'dark',
  sidebarOpen: true,
  searchOpen: false,
  notifications: [],
  webSocketConnected: false,
};

// Async thunks
export const initializeApp = createAsyncThunk(
  'app/initialize',
  async () => {
    try {
      // Check for stored auth token
      const token = localStorage.getItem('authToken');
      if (token) {
        // Verify token and get user info
        const response = await apiService.get('/auth/me');
        return { isAuthenticated: true, user: response.data };
      }
      return { isAuthenticated: false, user: null };
    } catch (error) {
      return { isAuthenticated: false, user: null };
    }
  }
);

export const login = createAsyncThunk(
  'app/login',
  async (credentials: { email: string; password: string }) => {
    const response = await apiService.post('/auth/login', credentials);
    localStorage.setItem('authToken', response.data.token);
    return response.data.user;
  }
);

export const logout = createAsyncThunk(
  'app/logout',
  async () => {
    await apiService.post('/auth/logout');
    localStorage.removeItem('authToken');
  }
);

const appSlice = createSlice({
  name: 'app',
  initialState,
  reducers: {
    setThemeMode: (state, action: PayloadAction<ThemeMode>) => {
      state.themeMode = action.payload;
      localStorage.setItem('themeMode', action.payload);
    },
    toggleTheme: (state) => {
      state.themeMode = state.themeMode === 'light' ? 'dark' : 'light';
      localStorage.setItem('themeMode', state.themeMode);
    },
    toggleSidebar: (state) => {
      state.sidebarOpen = !state.sidebarOpen;
    },
    setSidebarOpen: (state, action: PayloadAction<boolean>) => {
      state.sidebarOpen = action.payload;
    },
    toggleSearch: (state) => {
      state.searchOpen = !state.searchOpen;
    },
    setSearchOpen: (state, action: PayloadAction<boolean>) => {
      state.searchOpen = action.payload;
    },
    addNotification: (state, action: PayloadAction<{
      type: 'success' | 'error' | 'warning' | 'info';
      message: string;
    }>) => {
      state.notifications.push({
        id: Date.now().toString(),
        timestamp: Date.now(),
        ...action.payload,
      });
      // Keep only last 10 notifications
      if (state.notifications.length > 10) {
        state.notifications.shift();
      }
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },
    clearNotifications: (state) => {
      state.notifications = [];
    },
    setWebSocketConnected: (state, action: PayloadAction<boolean>) => {
      state.webSocketConnected = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(initializeApp.fulfilled, (state, action) => {
        state.isInitialized = true;
        state.isAuthenticated = action.payload.isAuthenticated;
        state.user = action.payload.user;
        if (action.payload.user?.preferences?.theme) {
          state.themeMode = action.payload.user.preferences.theme;
        }
      })
      .addCase(initializeApp.rejected, (state) => {
        state.isInitialized = true;
        state.isAuthenticated = false;
      })
      .addCase(login.fulfilled, (state, action) => {
        state.isAuthenticated = true;
        state.user = action.payload;
      })
      .addCase(logout.fulfilled, (state) => {
        state.isAuthenticated = false;
        state.user = null;
      });
  },
});

export const {
  setThemeMode,
  toggleTheme,
  toggleSidebar,
  setSidebarOpen,
  toggleSearch,
  setSearchOpen,
  addNotification,
  removeNotification,
  clearNotifications,
  setWebSocketConnected,
} = appSlice.actions;

export default appSlice.reducer;