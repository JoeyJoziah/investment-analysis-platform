import React from 'react';
import {
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
  Slider,
  Chip,
  IconButton,
  Typography,
} from '@mui/material';
import {
  Save,
  Add,
  Delete,
} from '@mui/icons-material';

// --- Shared types ---

export interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

export function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

// --- Profile Tab ---

export interface ProfileTabProps {
  userName: string;
  userEmail: string;
  timezone: string;
  onTimezoneChange: (timezone: string) => void;
  onSave: () => void;
}

export const ProfileTab: React.FC<ProfileTabProps> = ({
  userName,
  userEmail,
  timezone,
  onTimezoneChange,
  onSave,
}) => (
  <>
    <Typography variant="h6" gutterBottom>
      Profile Information
    </Typography>
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          label="Full Name"
          defaultValue={userName}
          margin="normal"
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          label="Email"
          type="email"
          defaultValue={userEmail}
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
            value={timezone}
            label="Timezone"
            onChange={(e) => onTimezoneChange(e.target.value)}
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
        <Button variant="contained" startIcon={<Save />} onClick={onSave}>
          Save Profile
        </Button>
      </Grid>
    </Grid>
  </>
);

// --- Appearance Tab ---

export interface AppearanceTabProps {
  themeMode: 'light' | 'dark';
  onThemeModeChange: (mode: 'light' | 'dark') => void;
  defaultView: string;
  onDefaultViewChange: (view: string) => void;
  currency: string;
  onCurrencyChange: (currency: string) => void;
  autoRefresh: boolean;
  onAutoRefreshChange: (enabled: boolean) => void;
  refreshInterval: number;
  onRefreshIntervalChange: (interval: number) => void;
  onSave: () => void;
}

export const AppearanceTab: React.FC<AppearanceTabProps> = ({
  themeMode,
  onThemeModeChange,
  defaultView,
  onDefaultViewChange,
  currency,
  onCurrencyChange,
  autoRefresh,
  onAutoRefreshChange,
  refreshInterval,
  onRefreshIntervalChange,
  onSave,
}) => (
  <>
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
            onChange={(e) => onThemeModeChange(e.target.value as 'light' | 'dark')}
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
            value={defaultView}
            label="Default View"
            onChange={(e) => onDefaultViewChange(e.target.value)}
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
            value={currency}
            label="Currency Display"
            onChange={(e) => onCurrencyChange(e.target.value)}
          >
            <MenuItem value="USD">USD ($)</MenuItem>
            <MenuItem value="EUR">EUR</MenuItem>
            <MenuItem value="GBP">GBP</MenuItem>
            <MenuItem value="JPY">JPY</MenuItem>
          </Select>
        </FormControl>
      </Grid>
      <Grid item xs={12}>
        <FormControlLabel
          control={
            <Switch
              checked={autoRefresh}
              onChange={(e) => onAutoRefreshChange(e.target.checked)}
            />
          }
          label="Auto-refresh data"
        />
      </Grid>
      {autoRefresh && (
        <Grid item xs={12}>
          <Typography gutterBottom>
            Refresh Interval: {refreshInterval} seconds
          </Typography>
          <Slider
            value={refreshInterval}
            onChange={(_, value) => onRefreshIntervalChange(value as number)}
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
        <Button variant="contained" startIcon={<Save />} onClick={onSave}>
          Save Preferences
        </Button>
      </Grid>
    </Grid>
  </>
);

// --- Notifications Tab ---

export interface AlertItem {
  id: number;
  ticker: string;
  type: string;
  condition: string;
  value: number;
  active: boolean;
}

export interface NewAlertInput {
  ticker: string;
  type: string;
  condition: string;
  value: number;
}

export interface NotificationsTabProps {
  showNotifications: boolean;
  onShowNotificationsChange: (enabled: boolean) => void;
  emailAlerts: boolean;
  onEmailAlertsChange: (enabled: boolean) => void;
  pushNotifications: boolean;
  onPushNotificationsChange: (enabled: boolean) => void;
  alerts: AlertItem[];
  newAlert: NewAlertInput;
  onNewAlertChange: (alert: NewAlertInput) => void;
  onAddAlert: () => void;
  onDeleteAlert: (id: number) => void;
  onToggleAlert: (id: number) => void;
}

export const NotificationsTab: React.FC<NotificationsTabProps> = ({
  showNotifications,
  onShowNotificationsChange,
  emailAlerts,
  onEmailAlertsChange,
  pushNotifications,
  onPushNotificationsChange,
  alerts,
  newAlert,
  onNewAlertChange,
  onAddAlert,
  onDeleteAlert,
  onToggleAlert,
}) => (
  <>
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
            checked={showNotifications}
            onChange={(e) => onShowNotificationsChange(e.target.checked)}
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
            checked={emailAlerts}
            onChange={(e) => onEmailAlertsChange(e.target.checked)}
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
            checked={pushNotifications}
            onChange={(e) => onPushNotificationsChange(e.target.checked)}
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
              onNewAlertChange({ ...newAlert, ticker: e.target.value.toUpperCase() })
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
                onNewAlertChange({ ...newAlert, type: e.target.value })
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
                onNewAlertChange({ ...newAlert, condition: e.target.value })
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
              onNewAlertChange({ ...newAlert, value: Number(e.target.value) })
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
            onClick={onAddAlert}
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
              onChange={() => onToggleAlert(alert.id)}
            />
            <IconButton edge="end" onClick={() => onDeleteAlert(alert.id)}>
              <Delete />
            </IconButton>
          </ListItemSecondaryAction>
        </ListItem>
      ))}
    </List>
  </>
);

// --- Security Tab ---

export const SecurityTab: React.FC = () => (
  <>
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
              secondary="Chrome on Windows - New York, US"
            />
            <ListItemSecondaryAction>
              <Chip label="Active" color="success" size="small" />
            </ListItemSecondaryAction>
          </ListItem>
        </List>
      </Grid>
    </Grid>
  </>
);

// --- Data & Privacy Tab ---

export const DataPrivacyTab: React.FC = () => (
  <>
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
  </>
);
