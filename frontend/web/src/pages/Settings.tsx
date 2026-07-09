import React, { useEffect, useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Tabs,
  Tab,
} from '@mui/material';
import {
  Person,
  Security,
  Notifications,
  Palette,
  Api,
  DataUsage,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import { setThemeMode, addNotification } from '../store/slices/appSlice';
import {
  TabPanel,
  ProfileTab,
  AppearanceTab,
  NotificationsTab,
  SecurityTab,
  DataPrivacyTab,
} from '../components/settings/SettingsTabs';
import type { AlertItem, NewAlertInput } from '../components/settings/SettingsTabs';
import { ApiKeysForm } from '../components/settings/SettingsForm';
import type { ApiKeysState, ApiKeysConfiguredStatus } from '../components/settings/SettingsForm';
import { apiService } from '../services/api.service';
import { apiConfig } from '../config/api.config';

// state.app.user is normalized to the camelCase `User` shape (name/email) in
// appSlice. Read `full_name` too as a defensive fallback in case an
// un-normalized payload ever reaches here.
const resolveUserName = (user: { name?: string; full_name?: string } | null | undefined): string =>
  user?.name ?? user?.full_name ?? '';

const Settings: React.FC = () => {
  const dispatch = useAppDispatch();
  const { themeMode, user } = useAppSelector((state) => state.app);

  // Pre-fill the Profile form from the logged-in user. `user` may briefly be
  // null on first render (before GET /api/v1/auth/me resolves), so we seed from
  // whatever is present and sync via useEffect once the user loads.
  const [profile, setProfile] = useState({
    name: resolveUserName(user),
    email: user?.email ?? '',
  });

  useEffect(() => {
    setProfile({
      name: resolveUserName(user),
      email: user?.email ?? '',
    });
  }, [user]);

  const [tabValue, setTabValue] = useState(0);
  const [showPassword, setShowPassword] = useState(false);
  const [apiKeys, setApiKeys] = useState<ApiKeysState>({
    alphaVantage: '',
    finnhub: '',
    polygon: '',
    newsApi: '',
  });
  // Masked configured-status from the backend so the user can see which keys
  // are already saved (and not blindly re-enter, which can overwrite a good
  // key). The GET endpoint never returns full secrets — only masked previews.
  const [apiKeyStatus, setApiKeyStatus] = useState<ApiKeysConfiguredStatus>({});

  const loadApiKeyStatus = React.useCallback(async () => {
    type KeyStatus = { configured?: boolean; masked?: string | null };
    type StatusMap = Record<string, KeyStatus>;
    try {
      const resp = await apiService.get<{ data?: StatusMap } & StatusMap>(
        apiConfig.endpoints.settings.apiKeys
      );
      const data: StatusMap | undefined = resp.data?.data ?? resp.data;
      if (!data) return;
      setApiKeyStatus({
        alphaVantage: data.alpha_vantage && {
          configured: !!data.alpha_vantage.configured,
          masked: data.alpha_vantage.masked ?? null,
        },
        finnhub: data.finnhub && {
          configured: !!data.finnhub.configured,
          masked: data.finnhub.masked ?? null,
        },
        polygon: data.polygon && {
          configured: !!data.polygon.configured,
          masked: data.polygon.masked ?? null,
        },
        newsApi: data.news_api && {
          configured: !!data.news_api.configured,
          masked: data.news_api.masked ?? null,
        },
      });
    } catch {
      // Status is a convenience; ignore failures (form still works).
    }
  }, []);

  useEffect(() => {
    loadApiKeyStatus();
  }, [loadApiKeyStatus]);
  const [preferences, setPreferences] = useState({
    defaultView: 'dashboard',
    autoRefresh: true,
    refreshInterval: 60,
    showNotifications: true,
    emailAlerts: false,
    pushNotifications: false,
    language: 'en',
    timezone: 'America/New_York',
    currency: 'USD',
  });
  const [alerts, setAlerts] = useState<AlertItem[]>([
    { id: 1, ticker: 'AAPL', type: 'price', condition: 'above', value: 150, active: true },
    { id: 2, ticker: 'GOOGL', type: 'percent', condition: 'below', value: -5, active: true },
  ]);
  const [newAlert, setNewAlert] = useState<NewAlertInput>({
    ticker: '',
    type: 'price',
    condition: 'above',
    value: 0,
  });

  const handleSaveProfile = () => {
    dispatch(
      addNotification({
        type: 'success',
        message: 'Profile settings saved successfully',
      })
    );
  };

  const handleSavePreferences = () => {
    dispatch(
      addNotification({
        type: 'success',
        message: 'Preferences saved successfully',
      })
    );
  };

  const handleSaveApiKeys = async () => {
    // Map the UI's camelCase fields to the backend snake_case contract. Blank fields
    // are sent as empty strings and ignored server-side (so they aren't overwritten).
    const payload = {
      alpha_vantage: apiKeys.alphaVantage,
      finnhub: apiKeys.finnhub,
      polygon: apiKeys.polygon,
      news_api: apiKeys.newsApi,
    };
    try {
      await apiService.put(apiConfig.endpoints.settings.apiKeys, payload);
      // Clear the typed values and refresh the masked status so the saved
      // (masked) keys are reflected and aren't accidentally re-submitted.
      setApiKeys({ alphaVantage: '', finnhub: '', polygon: '', newsApi: '' });
      await loadApiKeyStatus();
      dispatch(
        addNotification({
          type: 'success',
          message: 'API keys saved. Restart the backend for all data providers to pick them up.',
        })
      );
    } catch (err: unknown) {
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        (err as { message?: string })?.message ||
        'Failed to save API keys';
      dispatch(addNotification({ type: 'error', message: detail }));
    }
  };

  const handleAddAlert = () => {
    if (newAlert.ticker) {
      setAlerts([
        ...alerts,
        {
          ...newAlert,
          id: Date.now(),
          active: true,
        },
      ]);
      setNewAlert({
        ticker: '',
        type: 'price',
        condition: 'above',
        value: 0,
      });
      dispatch(
        addNotification({
          type: 'success',
          message: 'Alert added successfully',
        })
      );
    }
  };

  const handleDeleteAlert = (id: number) => {
    setAlerts(alerts.filter((alert) => alert.id !== id));
    dispatch(
      addNotification({
        type: 'info',
        message: 'Alert removed',
      })
    );
  };

  const handleToggleAlert = (id: number) => {
    setAlerts(
      alerts.map((alert) =>
        alert.id === id ? { ...alert, active: !alert.active } : alert
      )
    );
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="bold">
          Settings
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Manage your account settings and preferences
        </Typography>
      </Box>

      <Paper>
        <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
          <Tab icon={<Person />} label="Profile" />
          <Tab icon={<Palette />} label="Appearance" />
          <Tab icon={<Notifications />} label="Notifications" />
          <Tab icon={<Api />} label="API Keys" />
          <Tab icon={<Security />} label="Security" />
          <Tab icon={<DataUsage />} label="Data & Privacy" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <ProfileTab
            key={`${profile.name}|${profile.email}`}
            userName={profile.name}
            userEmail={profile.email}
            timezone={preferences.timezone}
            onTimezoneChange={(timezone) =>
              setPreferences({ ...preferences, timezone })
            }
            onSave={handleSaveProfile}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <AppearanceTab
            themeMode={themeMode}
            onThemeModeChange={(mode) => dispatch(setThemeMode(mode))}
            defaultView={preferences.defaultView}
            onDefaultViewChange={(defaultView) =>
              setPreferences({ ...preferences, defaultView })
            }
            currency={preferences.currency}
            onCurrencyChange={(currency) =>
              setPreferences({ ...preferences, currency })
            }
            autoRefresh={preferences.autoRefresh}
            onAutoRefreshChange={(autoRefresh) =>
              setPreferences({ ...preferences, autoRefresh })
            }
            refreshInterval={preferences.refreshInterval}
            onRefreshIntervalChange={(refreshInterval) =>
              setPreferences({ ...preferences, refreshInterval })
            }
            onSave={handleSavePreferences}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <NotificationsTab
            showNotifications={preferences.showNotifications}
            onShowNotificationsChange={(showNotifications) =>
              setPreferences({ ...preferences, showNotifications })
            }
            emailAlerts={preferences.emailAlerts}
            onEmailAlertsChange={(emailAlerts) =>
              setPreferences({ ...preferences, emailAlerts })
            }
            pushNotifications={preferences.pushNotifications}
            onPushNotificationsChange={(pushNotifications) =>
              setPreferences({ ...preferences, pushNotifications })
            }
            alerts={alerts}
            newAlert={newAlert}
            onNewAlertChange={setNewAlert}
            onAddAlert={handleAddAlert}
            onDeleteAlert={handleDeleteAlert}
            onToggleAlert={handleToggleAlert}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <ApiKeysForm
            apiKeys={apiKeys}
            showPassword={showPassword}
            onApiKeysChange={setApiKeys}
            onToggleShowPassword={() => setShowPassword(!showPassword)}
            onSave={handleSaveApiKeys}
            configuredStatus={apiKeyStatus}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={4}>
          <SecurityTab />
        </TabPanel>

        <TabPanel value={tabValue} index={5}>
          <DataPrivacyTab />
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default Settings;
