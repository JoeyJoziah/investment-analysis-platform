import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { apiService } from '../../services/api.service';
import { apiConfig } from '../../config/api.config';

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

// The backend `/api/v1/auth/me` payload uses snake_case `full_name` and an
// integer `id`. Normalize it into the camelCase `User` shape the UI renders so
// the Profile form (and anything reading state.app.user) gets `name`/`email`.
interface BackendUser {
  id?: string | number;
  email?: string;
  full_name?: string;
  name?: string;
  preferences?: User['preferences'];
}

const normalizeUser = (raw: unknown): User | null => {
  if (!raw || typeof raw !== 'object') {
    return null;
  }
  const u = raw as BackendUser;
  if (u.email == null && u.full_name == null && u.name == null) {
    return null;
  }
  return {
    id: u.id != null ? String(u.id) : '',
    email: u.email ?? '',
    name: u.full_name ?? u.name ?? '',
    preferences: u.preferences,
  };
};

// Axios response bodies default to `unknown`; describe the ApiResponse envelope
// ({ success, data: T }) so reading `.data?.data` type-checks. `Partial<T>`
// tolerates an un-enveloped payload landing directly on `.data`.
type Envelope<T> = { data?: T } & Partial<T>;

interface LoginData {
  access_token?: string;
  user?: BackendUser;
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
      const token = localStorage.getItem('access_token');
      if (token) {
        // Verify token and get user info using centralized endpoint config.
        // Backend returns the ApiResponse envelope { success, data: {...user} }.
        const response = await apiService.get<Envelope<BackendUser>>(apiConfig.endpoints.auth.profile);
        const raw = response.data?.data ?? response.data;
        return { isAuthenticated: true, user: normalizeUser(raw) };
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
    // Backend returns ApiResponse envelope: { success, data: { access_token, ... } }
    // (see backend/api/routers/auth.py and backend/models/api_response.py).
    const response = await apiService.post<Envelope<LoginData>>(apiConfig.endpoints.auth.login, credentials);
    const payload = response.data?.data ?? response.data;
    const token = payload?.access_token;
    if (!token) {
      throw new Error('Login response did not include access_token');
    }
    localStorage.setItem('access_token', token);

    // The backend login response only carries the token (no user object), so
    // fetch the profile to populate state.app.user. The request interceptor
    // attaches the token we just stored. Keep login successful even if /me
    // hiccups — initializeApp will repopulate the user on the next mount.
    try {
      const profileResp = await apiService.get<Envelope<BackendUser>>(apiConfig.endpoints.auth.profile);
      const raw = profileResp.data?.data ?? profileResp.data;
      return normalizeUser(raw);
    } catch {
      return normalizeUser(payload?.user);
    }
  }
);

export const logout = createAsyncThunk(
  'app/logout',
  async () => {
    await apiService.post(apiConfig.endpoints.auth.logout);
    localStorage.removeItem('access_token');
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
