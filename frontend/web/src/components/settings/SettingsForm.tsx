import React from 'react';
import {
  Grid,
  TextField,
  Button,
  Alert,
  IconButton,
  InputAdornment,
  Typography,
} from '@mui/material';
import {
  Save,
  Visibility,
  VisibilityOff,
} from '@mui/icons-material';

// --- API Keys Form ---

export interface ApiKeysState {
  alphaVantage: string;
  finnhub: string;
  polygon: string;
  newsApi: string;
}

export interface ApiKeyConfiguredStatus {
  configured: boolean;
  masked: string | null;
}

export type ApiKeysConfiguredStatus = Partial<
  Record<keyof ApiKeysState, ApiKeyConfiguredStatus>
>;

export interface ApiKeysFormProps {
  apiKeys: ApiKeysState;
  showPassword: boolean;
  onApiKeysChange: (keys: ApiKeysState) => void;
  onToggleShowPassword: () => void;
  onSave: () => void;
  // Masked configured-status from GET /settings/api-keys, so the user can see
  // which keys are already saved and avoid blindly re-entering (which can
  // overwrite a good key with a wrong one).
  configuredStatus?: ApiKeysConfiguredStatus;
}

interface ApiKeyFieldConfig {
  key: keyof ApiKeysState;
  label: string;
  helperText: string;
}

const API_KEY_FIELDS: ApiKeyFieldConfig[] = [
  {
    key: 'alphaVantage',
    label: 'Alpha Vantage API Key',
    helperText: 'Free tier: 25 API calls/day, 5 calls/minute',
  },
  {
    key: 'finnhub',
    label: 'Finnhub API Key',
    helperText: 'Free tier: 60 calls/minute',
  },
  {
    key: 'polygon',
    label: 'Polygon.io API Key',
    helperText: 'Free tier: 5 API calls/minute',
  },
  {
    key: 'newsApi',
    label: 'News API Key',
    helperText: 'For news and sentiment analysis',
  },
];

export const ApiKeysForm: React.FC<ApiKeysFormProps> = ({
  apiKeys,
  showPassword,
  onApiKeysChange,
  onToggleShowPassword,
  onSave,
  configuredStatus,
}) => (
  <>
    <Alert severity="warning" sx={{ mb: 3 }}>
      Keep your API keys secure. Never share them publicly or commit them to version control.
    </Alert>

    <Typography variant="h6" gutterBottom>
      API Configuration
    </Typography>
    <Grid container spacing={3}>
      {API_KEY_FIELDS.map((field) => {
        const status = configuredStatus?.[field.key];
        const isConfigured = Boolean(status?.configured);
        const isEmpty = !apiKeys[field.key];
        return (
        <Grid item xs={12} key={field.key}>
          <TextField
            fullWidth
            label={field.label}
            type={showPassword ? 'text' : 'password'}
            value={apiKeys[field.key]}
            onChange={(e) =>
              onApiKeysChange({ ...apiKeys, [field.key]: e.target.value })
            }
            margin="normal"
            placeholder={
              isConfigured && isEmpty
                ? `${status?.masked ?? 'Configured'} — leave blank to keep`
                : undefined
            }
            helperText={
              isConfigured
                ? `Configured${status?.masked ? ` (${status.masked})` : ''} — enter a new value only to replace it`
                : field.helperText
            }
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    onClick={onToggleShowPassword}
                    edge="end"
                  >
                    {showPassword ? <VisibilityOff /> : <Visibility />}
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
        </Grid>
        );
      })}
      <Grid item xs={12}>
        <Button variant="contained" startIcon={<Save />} onClick={onSave}>
          Save API Keys
        </Button>
      </Grid>
    </Grid>
  </>
);
