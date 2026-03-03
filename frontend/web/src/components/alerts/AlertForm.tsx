import React, { memo, useMemo } from 'react';
import {
  Box,
  Typography,
  TextField,
  MenuItem,
  Switch,
  FormControlLabel,
  FormControl,
  InputLabel,
  Select,
  InputAdornment,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
} from '@mui/material';

export type AlertType = 'price' | 'volume' | 'percent_change' | 'news' | 'portfolio_drift';
export type AlertCondition = 'above' | 'below' | 'equals';

export interface AlertFormData {
  ticker: string;
  type: AlertType;
  condition: AlertCondition;
  value: string;
  notifyInApp: boolean;
  notifyEmail: boolean;
}

export const EMPTY_FORM: AlertFormData = {
  ticker: '',
  type: 'price',
  condition: 'above',
  value: '',
  notifyInApp: true,
  notifyEmail: false,
};

interface AlertFormProps {
  open: boolean;
  form: AlertFormData;
  mode: 'create' | 'edit';
  editTicker?: string;
  onFormChange: (form: AlertFormData) => void;
  onSubmit: () => void;
  onClose: () => void;
}

/**
 * AlertForm - Dialog for creating or editing an alert.
 *
 * Renders ticker, type, condition, value, and notification fields inside a
 * MUI Dialog. The parent owns the form state; this component calls
 * onFormChange for every field update and onSubmit when the user confirms.
 */
const AlertForm: React.FC<AlertFormProps> = ({
  open,
  form,
  mode,
  editTicker,
  onFormChange,
  onSubmit,
  onClose,
}) => {
  const adornment = useMemo(() => {
    if (form.type === 'price') return { start: '$' };
    if (form.type === 'percent_change' || form.type === 'portfolio_drift') return { end: '%' };
    return {};
  }, [form.type]);

  const formDisabled = !form.ticker.trim() || (form.type !== 'news' && !form.value);

  const title = mode === 'create'
    ? 'Create Alert'
    : `Edit Alert - ${editTicker ?? ''}`;

  const submitLabel = mode === 'create' ? 'Create' : 'Save Changes';

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>{title}</DialogTitle>
      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: 1 }}>
          <TextField
            autoFocus
            label="Ticker Symbol"
            fullWidth
            value={form.ticker}
            onChange={(e) =>
              onFormChange({ ...form, ticker: e.target.value.toUpperCase() })
            }
            disabled={form.type === 'portfolio_drift'}
            helperText={
              form.type === 'portfolio_drift' ? 'Portfolio-level alert' : undefined
            }
          />
          <FormControl fullWidth>
            <InputLabel>Alert Type</InputLabel>
            <Select
              value={form.type}
              label="Alert Type"
              onChange={(e) => {
                const t = e.target.value as AlertType;
                onFormChange({
                  ...form,
                  type: t,
                  ticker: t === 'portfolio_drift' ? 'Portfolio' : form.ticker,
                  condition: t === 'news' ? 'equals' : form.condition,
                });
              }}
            >
              <MenuItem value="price">Price</MenuItem>
              <MenuItem value="volume">Volume</MenuItem>
              <MenuItem value="percent_change">Percentage Change</MenuItem>
              <MenuItem value="news">News</MenuItem>
              <MenuItem value="portfolio_drift">Portfolio Drift</MenuItem>
            </Select>
          </FormControl>
          {form.type !== 'news' && (
            <>
              <FormControl fullWidth>
                <InputLabel>Condition</InputLabel>
                <Select
                  value={form.condition}
                  label="Condition"
                  onChange={(e) =>
                    onFormChange({
                      ...form,
                      condition: e.target.value as AlertCondition,
                    })
                  }
                >
                  <MenuItem value="above">Above</MenuItem>
                  <MenuItem value="below">Below</MenuItem>
                  <MenuItem value="equals">Equals</MenuItem>
                </Select>
              </FormControl>
              <TextField
                label="Value"
                type="number"
                fullWidth
                value={form.value}
                onChange={(e) =>
                  onFormChange({ ...form, value: e.target.value })
                }
                InputProps={{
                  startAdornment: adornment.start ? (
                    <InputAdornment position="start">
                      {adornment.start}
                    </InputAdornment>
                  ) : undefined,
                  endAdornment: adornment.end ? (
                    <InputAdornment position="end">
                      {adornment.end}
                    </InputAdornment>
                  ) : undefined,
                }}
              />
            </>
          )}
          <Typography variant="subtitle2" sx={{ mt: 1 }}>
            Notification Method
          </Typography>
          <Box sx={{ display: 'flex', gap: 3 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={form.notifyInApp}
                  onChange={(e) =>
                    onFormChange({ ...form, notifyInApp: e.target.checked })
                  }
                />
              }
              label="In-App"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={form.notifyEmail}
                  onChange={(e) =>
                    onFormChange({ ...form, notifyEmail: e.target.checked })
                  }
                />
              }
              label="Email"
            />
          </Box>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={onSubmit} variant="contained" disabled={formDisabled}>
          {submitLabel}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default memo(AlertForm);
