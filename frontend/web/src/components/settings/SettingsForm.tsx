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

export interface ApiKeysFormProps {
  apiKeys: ApiKeysState;
  showPassword: boolean;
  onApiKeysChange: (keys: ApiKeysState) => void;
  onToggleShowPassword: () => void;
  onSave: () => void;
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
}) => (
  <>
    <Alert severity="warning" sx={{ mb: 3 }}>
      Keep your API keys secure. Never share them publicly or commit them to version control.
    </Alert>

    <Typography variant="h6" gutterBottom>
      API Configuration
    </Typography>
    <Grid container spacing={3}>
      {API_KEY_FIELDS.map((field) => (
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
            helperText={field.helperText}
          />
        </Grid>
      ))}
      <Grid item xs={12}>
        <Button variant="contained" startIcon={<Save />} onClick={onSave}>
          Save API Keys
        </Button>
      </Grid>
    </Grid>
  </>
);
