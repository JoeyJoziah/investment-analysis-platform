import React, { memo } from 'react';
import {
  Box,
  Typography,
  Skeleton,
  useTheme,
  alpha,
  Grid,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material';
import { MarketIndex } from '../../types';

interface MarketOverviewPanelProps {
  indices: MarketIndex[];
  isLoading?: boolean;
}

/**
 * MarketOverviewPanel - Market indices display
 *
 * Features:
 * - Major indices (S&P 500, NASDAQ, DOW)
 * - Real-time updates
 * - Color-coded changes
 *
 * WCAG 2.1 AA Compliant
 */
const MarketOverviewPanel: React.FC<MarketOverviewPanelProps> = ({
  indices = [],
  isLoading = false,
}) => {
  const theme = useTheme();

  // Default indices if none provided
  const defaultIndices: MarketIndex[] = [
    {
      symbol: 'SPY',
      name: 'S&P 500',
      value: 4892.5,
      change: 22.3,
      changePercent: 0.45,
      volume: 0,
      high: 0,
      low: 0,
      previousClose: 0,
      timestamp: new Date().toISOString(),
    },
    {
      symbol: 'QQQ',
      name: 'NASDAQ',
      value: 15628.4,
      change: 97.2,
      changePercent: 0.62,
      volume: 0,
      high: 0,
      low: 0,
      previousClose: 0,
      timestamp: new Date().toISOString(),
    },
    {
      symbol: 'DIA',
      name: 'DOW',
      value: 38245.8,
      change: 107.1,
      changePercent: 0.28,
      volume: 0,
      high: 0,
      low: 0,
      previousClose: 0,
      timestamp: new Date().toISOString(),
    },
  ];

  const displayIndices = indices.length > 0 ? indices : defaultIndices;

  // Loading state
  if (isLoading) {
    return (
      <Box>
        <Skeleton variant="text" width={140} height={28} sx={{ mb: 2 }} />
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {[...Array(3)].map((_, i) => (
            <Skeleton
              key={i}
              variant="rectangular"
              height={60}
              sx={{ borderRadius: 1 }}
            />
          ))}
        </Box>
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Typography variant="h6" component="h2" gutterBottom>
        Market Overview
      </Typography>

      {/* Indices list */}
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          gap: 1.5,
        }}
        role="list"
        aria-label="Market indices"
      >
        {displayIndices.map((index) => {
          const isPositive = index.changePercent >= 0;

          return (
            <Box
              key={index.symbol}
              role="listitem"
              sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                p: 1.5,
                borderRadius: 1,
                backgroundColor: alpha(
                  isPositive
                    ? theme.palette.success.main
                    : theme.palette.error.main,
                  0.05
                ),
                border: `1px solid ${alpha(
                  isPositive
                    ? theme.palette.success.main
                    : theme.palette.error.main,
                  0.1
                )}`,
                transition: 'all 0.2s ease-out',
                '&:hover': {
                  backgroundColor: alpha(
                    isPositive
                      ? theme.palette.success.main
                      : theme.palette.error.main,
                    0.1
                  ),
                },
              }}
              aria-label={`${index.name}: ${index.value.toLocaleString()}, ${isPositive ? 'up' : 'down'} ${Math.abs(index.changePercent).toFixed(2)} percent`}
            >
              {/* Index info */}
              <Box>
                <Typography
                  variant="subtitle2"
                  sx={{
                    fontWeight: 700,
                    color: 'text.primary',
                  }}
                >
                  {index.name}
                </Typography>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{
                    fontFamily: '"SF Mono", Monaco, monospace',
                  }}
                >
                  {index.symbol}
                </Typography>
              </Box>

              {/* Value and change */}
              <Box sx={{ textAlign: 'right' }}>
                <Typography
                  variant="body2"
                  sx={{
                    fontWeight: 600,
                    fontFamily: '"SF Mono", Monaco, monospace',
                  }}
                >
                  {index.value.toLocaleString(undefined, {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </Typography>
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'flex-end',
                    gap: 0.5,
                  }}
                >
                  {isPositive ? (
                    <TrendingUpIcon
                      sx={{
                        fontSize: 14,
                        color: theme.palette.success.main,
                      }}
                      aria-hidden="true"
                    />
                  ) : (
                    <TrendingDownIcon
                      sx={{
                        fontSize: 14,
                        color: theme.palette.error.main,
                      }}
                      aria-hidden="true"
                    />
                  )}
                  <Typography
                    variant="caption"
                    sx={{
                      fontWeight: 600,
                      fontFamily: '"SF Mono", Monaco, monospace',
                      color: isPositive
                        ? theme.palette.success.main
                        : theme.palette.error.main,
                    }}
                  >
                    {isPositive ? '+' : ''}
                    {index.changePercent.toFixed(2)}%
                  </Typography>
                </Box>
              </Box>
            </Box>
          );
        })}
      </Box>

      {/* Market status */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          gap: 1,
          mt: 2,
          pt: 2,
          borderTop: `1px solid ${theme.palette.divider}`,
        }}
      >
        <Box
          sx={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            backgroundColor: isMarketOpen()
              ? theme.palette.success.main
              : theme.palette.error.main,
            animation: isMarketOpen()
              ? 'pulse 2s infinite'
              : 'none',
            '@keyframes pulse': {
              '0%': { opacity: 1 },
              '50%': { opacity: 0.5 },
              '100%': { opacity: 1 },
            },
          }}
          aria-hidden="true"
        />
        <Typography variant="caption" color="text.secondary">
          Market {isMarketOpen() ? 'Open' : 'Closed'}
        </Typography>
      </Box>
    </Box>
  );
};

/**
 * Check if US market is currently open
 * Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
 */
function isMarketOpen(): boolean {
  const now = new Date();
  const etTime = new Date(
    now.toLocaleString('en-US', { timeZone: 'America/New_York' })
  );

  const day = etTime.getDay();
  const hours = etTime.getHours();
  const minutes = etTime.getMinutes();
  const time = hours * 60 + minutes;

  // Weekend check
  if (day === 0 || day === 6) return false;

  // Market hours: 9:30 AM (570 minutes) to 4:00 PM (960 minutes)
  return time >= 570 && time < 960;
}

export default memo(MarketOverviewPanel);
