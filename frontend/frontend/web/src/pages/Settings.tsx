import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Grid,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  Tabs,
  Tab,
  Slider,
  Chip,
  IconButton,
  InputAdornment,
} from '@mui/material';
import {
  Person,
  Security,
  Notifications,
  Palette,
  Api,
  DataUsage,
  Save,
  Visibility,
  VisibilityOff,
  Add,
  Delete,
  Info,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import { setThemeMode, addNotification } from '../store/slices/appSlice';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const Settings: React.FC = () => {
  const dispatch = useAppDispatch();
  const { themeMode, user } = useAppSelector((state) => state.app);
  
  const [tabValue, setTabValue] = useState(0);
  const [showPassword, setShowPassword] = useState(false);
  const [apiKeys, setApiKeys] = useState({
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
  const [alerts, setAlerts] = useState([
    { id: 1, ticker: 'AAPL', type: 'price', condition: 'above', value: 150, active: true },
    { id: 2, ticker: 'GOOGL', type: 'percent', condition: 'below', value: -5, active: true },
  ]);
  const [newAlert, setNewAlert] = useState({
    ticker: '',
    type: 'price',
    condition: 'above',
    value: 0,
  });

  const handleSaveProfile = () => {
    // Save profile logic here
    dispatch(
      addNotification({
        type: 'success',
        message: 'Profile settings saved successfully',
      })
    );
  };

  const handleSavePreferences = () => {
    // Save preferences logic here
    dispatch(
      addNotification({
        type: 'success',
        message: 'Preferences saved successfully',
      })
    );
  };

  const handleSaveApiKeys = () => {
    // Save API keys logic here
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

        {/* Profile Tab */}
        <TabPanel value={tabValue} index={0}>
          <Typography variant="h6" gutterBottom>
            Profile Information
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Full Name"
                defaultValue={user?.name || ''}
                margin="normal"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Email"
                type="email"
                defaultValue={user?.email || ''}
                margin="normal"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Phone Number"
                margin="normal"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth margin="normal">
                <InputLabel>Timezone</InputLabel>
                <Select
                  value={preferences.timezone}
                  label="Timezone"
                  onChange={(e) =>
                    setPreferences({ ...preferences, timezone: e.target.value })
                  }
                >
                  <MenuItem value="America/New_York">Eastern Time</MenuItem>
                  <MenuItem value="America/Chicago">Central Time</MenuItem>
                  <MenuItem value="America/Denver">Mountain Time</MenuItem>
                  <MenuItem value="America/Los_Angeles">Pacific Time</MenuItem>
                  <MenuItem value="Europe/London">London</MenuItem>
                  <MenuItem value="Asia/Tokyo">Tokyo</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Bio"
                multiline
                rows={4}
                margin="normal"
              />
            </Grid>
            <Grid item xs={12}>
              <Button variant="contained" startIcon={<Save />} onClick={handleSaveProfile}>
                Save Profile
              </Button>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Appearance Tab */}
        <TabPanel value={tabValue} index={1}>
          <Typography variant="h6" gutterBottom>
            Appearance Settings
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Theme Mode</InputLabel>
                <Select
                  value={themeMode}
                  label="Theme Mode"
                  onChange={(e) => dispatch(setThemeMode(e.target.value as 'light' | 'dark'))}
                >
                  <MenuItem value="light">Light</MenuItem>
                  <MenuItem value="dark">Dark</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Default View</InputLabel>
                <Select
                  value={preferences.defaultView}
                  label="Default View"
                  onChange={(e) =>
                    setPreferences({ ...preferences, defaultView: e.target.value })
                  }
                >
                  <MenuItem value="dashboard">Dashboard</MenuItem>
                  <MenuItem value="portfolio">Portfolio</MenuItem>
                  <MenuItem value="recommendations">Recommendations</MenuItem>
                  <MenuItem value="market">Market Overview</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Currency Display</InputLabel>
                <Select
                  value={preferences.currency}
                  label="Currency Display"
                  onChange={(e) =>
                    setPreferences({ ...preferences, currency: e.target.value })
                  }
                >
                  <MenuItem value="USD">USD ($)</MenuItem>
                  <MenuItem value="EUR">EUR (€)</MenuItem>
                  <MenuItem value="GBP">GBP (£)</MenuItem>
                  <MenuItem value="JPY">JPY (¥)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={preferences.autoRefresh}
                    onChange={(e) =>
                      setPreferences({ ...preferences, autoRefresh: e.target.checked })
                    }
                  />
                }
                label="Auto-refresh data"
              />
            </Grid>
            {preferences.autoRefresh && (
              <Grid item xs={12}>
                <Typography gutterBottom>
                  Refresh Interval: {preferences.refreshInterval} seconds
                </Typography>
                <Slider
                  value={preferences.refreshInterval}
                  onChange={(_, value) =>
                    setPreferences({ ...preferences, refreshInterval: value as number })
                  }
                  min={30}
                  max={300}
                  step={30}
                  marks={[
                    { value: 30, label: '30s' },
                    { value: 60, label: '1m' },
                    { value: 120, label: '2m' },
                    { value: 180, label: '3m' },
                    { value: 300, label: '5m' },
                  ]}
                />
              </Grid>
            )}
            <Grid item xs={12}>
              <Button variant="contained" startIcon={<Save />} onClick={handleSavePreferences}>
                Save Preferences
              </Button>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Notifications Tab */}
        <TabPanel value={tabValue} index={2}>
          <Typography variant="h6" gutterBottom>
            Notification Settings
          </Typography>
          <List>
            <ListItem>
              <ListItemText
                primary="Show Notifications"
                secondary="Display in-app notifications for important events"
              />
              <ListItemSecondaryAction>
                <Switch
                  checked={preferences.showNotifications}
                  onChange={(e) =>
                    setPreferences({ ...preferences, showNotifications: e.target.checked })
                  }
                />
              </ListItemSecondaryAction>
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Email Alerts"
                secondary="Receive email notifications for price alerts and recommendations"
              />
              <ListItemSecondaryAction>
                <Switch
                  checked={preferences.emailAlerts}
                  onChange={(e) =>
                    setPreferences({ ...preferences, emailAlerts: e.target.checked })
                  }
                />
              </ListItemSecondaryAction>
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Push Notifications"
                secondary="Receive push notifications on your mobile device"
              />
              <ListItemSecondaryAction>
                <Switch
                  checked={preferences.pushNotifications}
                  onChange={(e) =>
                    setPreferences({ ...preferences, pushNotifications: e.target.checked })
                  }
                />
              </ListItemSecondaryAction>
            </ListItem>
          </List>

          <Divider sx={{ my: 3 }} />

          <Typography variant="h6" gutterBottom>
            Price Alerts
          </Typography>
          <Box sx={{ mb: 3 }}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} sm={3}>
                <TextField
                  fullWidth
                  label="Ticker"
                  value={newAlert.ticker}
                  onChange={(e) =>
                    setNewAlert({ ...newAlert, ticker: e.target.value.toUpperCase() })
                  }
                />
              </Grid>
              <Grid item xs={12} sm={2}>
                <FormControl fullWidth>
                  <InputLabel>Type</InputLabel>
                  <Select
                    value={newAlert.type}
                    label="Type"
                    onChange={(e) =>
                      setNewAlert({ ...newAlert, type: e.target.value })
                    }
                  >
                    <MenuItem value="price">Price</MenuItem>
                    <MenuItem value="percent">Percent</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={2}>
                <FormControl fullWidth>
                  <InputLabel>Condition</InputLabel>
                  <Select
                    value={newAlert.condition}
                    label="Condition"
                    onChange={(e) =>
                      setNewAlert({ ...newAlert, condition: e.target.value })
                    }
                  >
                    <MenuItem value="above">Above</MenuItem>
                    <MenuItem value="below">Below</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={3}>
                <TextField
                  fullWidth
                  label="Value"
                  type="number"
                  value={newAlert.value}
                  onChange={(e) =>
                    setNewAlert({ ...newAlert, value: Number(e.target.value) })
                  }
                  InputProps={{
                    startAdornment: newAlert.type === 'price' ? '$' : undefined,
                    endAdornment: newAlert.type === 'percent' ? '%' : undefined,
                  }}
                />
              </Grid>
              <Grid item xs={12} sm={2}>
                <Button
                  fullWidth
                  variant="contained"
                  startIcon={<Add />}
                  onClick={handleAddAlert}
                >
                  Add Alert
                </Button>
              </Grid>
            </Grid>
          </Box>

          <List>
            {alerts.map((alert) => (
              <ListItem key={alert.id}>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Chip label={alert.ticker} size="small" color="primary" />
                      <Typography>
                        {alert.condition} {alert.type === 'price' ? '$' : ''}
                        {alert.value}
                        {alert.type === 'percent' ? '%' : ''}
                      </Typography>
                    </Box>
                  }
                  secondary={`Alert when ${alert.ticker} ${alert.type} is ${alert.condition} ${
                    alert.type === 'price' ? '$' : ''
                  }${alert.value}${alert.type === 'percent' ? '%' : ''}`}
                />
                <ListItemSecondaryAction>
                  <Switch
                    checked={alert.active}
                    onChange={() => handleToggleAlert(alert.id)}
                  />
                  <IconButton edge="end" onClick={() => handleDeleteAlert(alert.id)}>
                    <Delete />
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        </TabPanel>

        {/* API Keys Tab */}
        <TabPanel value={tabValue} index={3}>
          <Alert severity="warning" sx={{ mb: 3 }}>
            Keep your API keys secure. Never share them publicly or commit them to version control.
          </Alert>
          
          <Typography variant="h6" gutterBottom>
            API Configuration
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Alpha Vantage API Key"
                type={showPassword ? 'text' : 'password'}
                value={apiKeys.alphaVantage}
                onChange={(e) =>
                  setApiKeys({ ...apiKeys, alphaVantage: e.target.value })
                }
                margin="normal"
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        onClick={() => setShowPassword(!showPassword)}
                        edge="end"
                      >
                        {showPassword ? <VisibilityOff /> : <Visibility />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
                helperText="Free tier: 25 API calls/day, 5 calls/minute"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Finnhub API Key"
                type={showPassword ? 'text' : 'password'}
                value={apiKeys.finnhub}
                onChange={(e) =>
                  setApiKeys({ ...apiKeys, finnhub: e.target.value })
                }
                margin="normal"
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        onClick={() => setShowPassword(!showPassword)}
                        edge="end"
                      >
                        {showPassword ? <VisibilityOff /> : <Visibility />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
                helperText="Free tier: 60 calls/minute"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Polygon.io API Key"
                type={showPassword ? 'text' : 'password'}
                value={apiKeys.polygon}
                onChange={(e) =>
                  setApiKeys({ ...apiKeys, polygon: e.target.value })
                }
                margin="normal"
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        onClick={() => setShowPassword(!showPassword)}
                        edge="end"
                      >
                        {showPassword ? <VisibilityOff /> : <Visibility />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
                helperText="Free tier: 5 API calls/minute"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="News API Key"
                type={showPassword ? 'text' : 'password'}
                value={apiKeys.newsApi}
                onChange={(e) =>
                  setApiKeys({ ...apiKeys, newsApi: e.target.value })
                }
                margin="normal"
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        onClick={() => setShowPassword(!showPassword)}
                        edge="end"
                      >
                        {showPassword ? <VisibilityOff /> : <Visibility />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
                helperText="For news and sentiment analysis"
              />
            </Grid>
            <Grid item xs={12}>
              <Button variant="contained" startIcon={<Save />} onClick={handleSaveApiKeys}>
                Save API Keys
              </Button>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Security Tab */}
        <TabPanel value={tabValue} index={4}>
          <Typography variant="h6" gutterBottom>
            Security Settings
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Change Password
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Current Password"
                type="password"
                margin="normal"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="New Password"
                type="password"
                margin="normal"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Confirm New Password"
                type="password"
                margin="normal"
              />
            </Grid>
            <Grid item xs={12}>
              <Button variant="contained" color="primary">
                Update Password
              </Button>
            </Grid>
            
            <Grid item xs={12}>
              <Divider sx={{ my: 2 }} />
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Two-Factor Authentication
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Add an extra layer of security to your account
              </Typography>
              <Button variant="outlined">
                Enable 2FA
              </Button>
            </Grid>
            
            <Grid item xs={12}>
              <Divider sx={{ my: 2 }} />
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Active Sessions
              </Typography>
              <List>
                <ListItem>
                  <ListItemText
                    primary="Current Session"
                    secondary="Chrome on Windows • New York, US"
                  />
                  <ListItemSecondaryAction>
                    <Chip label="Active" color="success" size="small" />
                  </ListItemSecondaryAction>
                </ListItem>
              </List>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Data & Privacy Tab */}
        <TabPanel value={tabValue} index={5}>
          <Typography variant="h6" gutterBottom>
            Data & Privacy Settings
          </Typography>
          <List>
            <ListItem>
              <ListItemText
                primary="Data Collection"
                secondary="Allow collection of usage data to improve the service"
              />
              <ListItemSecondaryAction>
                <Switch defaultChecked />
              </ListItemSecondaryAction>
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Personalized Recommendations"
                secondary="Use your trading history to provide personalized stock recommendations"
              />
              <ListItemSecondaryAction>
                <Switch defaultChecked />
              </ListItemSecondaryAction>
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Share Data with Partners"
                secondary="Share anonymized data with our partners for research"
              />
              <ListItemSecondaryAction>
                <Switch />
              </ListItemSecondaryAction>
            </ListItem>
          </List>
          
          <Divider sx={{ my: 3 }} />
          
          <Typography variant="h6" gutterBottom>
            Data Management
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Button variant="outlined" fullWidth>
                Download My Data
              </Button>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Button variant="outlined" color="error" fullWidth>
                Delete Account
              </Button>
            </Grid>
          </Grid>
          
          <Alert severity="info" sx={{ mt: 3 }}>
            Your data is encrypted and stored securely. We comply with GDPR and SEC regulations.
          </Alert>
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default Settings;