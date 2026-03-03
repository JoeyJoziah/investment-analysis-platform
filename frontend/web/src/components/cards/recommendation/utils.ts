/**
 * Shared utility functions for recommendation card sub-components
 */

import { Theme } from '@mui/material';

export const getActionColor = (action: string, theme: Theme): string => {
  switch (action) {
    case 'BUY':
      return theme.palette.success.main;
    case 'SELL':
      return theme.palette.error.main;
    case 'HOLD':
      return theme.palette.warning.main;
    default:
      return theme.palette.text.secondary;
  }
};

/**
 * Maps a recommendation action to a valid MUI Button color prop.
 * Use this instead of getActionColor when assigning the `color` prop on MUI components.
 */
export const getActionMuiColor = (
  action: string
): 'success' | 'error' | 'warning' | 'inherit' => {
  switch (action) {
    case 'BUY':
      return 'success';
    case 'SELL':
      return 'error';
    case 'HOLD':
      return 'warning';
    default:
      return 'inherit';
  }
};

export const formatValue = (
  value: number | undefined | null,
  type: 'currency' | 'percent' | 'number' = 'number'
): string => {
  if (value === undefined || value === null) return 'N/A';

  switch (type) {
    case 'currency':
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
      }).format(value);
    case 'percent':
      return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
    default:
      return value.toLocaleString();
  }
};
