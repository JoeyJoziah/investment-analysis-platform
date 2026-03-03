import React, { useState } from 'react';
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
import type { ApiKeysState } from '../components/settings/SettingsForm';

const Settings: React.FC = () => {
  const dispatch = useAppDispatch();
  const { themeMode, user } = useAppSelector((state) => state.app);

  const [tabValue, setTabValue] = useState(0);
  const [showPassword, setShowPassword] = useState(false);
  const [apiKeys, setApiKeys] = useState<ApiKeysState>({
    alphaVantage: '',
    finnhub: '',
    polygon: '',
    newsApi: '',
  });
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

  const handleSaveApiKeys = () => {
    dispatch(
      addNotification({
        type: 'success',
        message: 'API keys saved successfully',
      })
    );
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
            userName={user?.name || ''}
            userEmail={user?.email || ''}
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
