/**
 * Market Summary Cards - Displays market indices and market breadth panels
 */

import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
} from '@mui/material';
import {
  ArrowUpward,
  ArrowDownward,
  CloudOff,
} from '@mui/icons-material';
import type { MarketIndex, MarketBreadth } from '../../store/slices/marketSlice';

const EmptyStateBox: React.FC<{
  icon: React.ReactNode;
  message: string;
  submessage?: string;
  minHeight?: number;
}> = ({ icon, message, submessage, minHeight = 200 }) => (
  <Box
    sx={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight,
      py: 4,
      px: 2,
      color: 'text.secondary',
    }}
  >
    <Box sx={{ mb: 1.5, opacity: 0.5 }}>{icon}</Box>
    <Typography variant="body1" color="text.secondary" textAlign="center">
      {message}
    </Typography>
    {submessage && (
      <Typography variant="body2" color="text.disabled" textAlign="center" sx={{ mt: 0.5 }}>
        {submessage}
      </Typography>
    )}
  </Box>
);

export { EmptyStateBox };

interface MarketSummaryProps {
  indices: MarketIndex[];
  marketBreadth: MarketBreadth | null;
  formatPercent: (value: number) => string;
  formatLargeNumber: (value: number) => string;
}

const MarketSummary: React.FC<MarketSummaryProps> = ({
  indices = [],
  marketBreadth,
  formatPercent,
  formatLargeNumber,
}) => {
  const safeIndices = indices ?? [];

  return (
    <>
      {/* Market Indices */}
      {safeIndices.length > 0 ? (
        <Grid container spacing={2} sx={{ mb: 3 }}>
          {safeIndices.map((index) => (
            <Grid item xs={12} sm={6} md={3} key={index.symbol}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    {index.name}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1 }}>
                    <Typography variant="h5" fontWeight="bold">
                      {index.value.toLocaleString()}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      {index.change >= 0 ? (
                        <ArrowUpward sx={{ fontSize: 16, color: 'success.main' }} />
                      ) : (
                        <ArrowDownward sx={{ fontSize: 16, color: 'error.main' }} />
                      )}
                      <Typography
                        variant="body2"
                        color={index.change >= 0 ? 'success.main' : 'error.main'}
                      >
                        {index.change.toFixed(2)} ({formatPercent(index.changePercent)})
                      </Typography>
                    </Box>
                  </Box>
                  <Box sx={{ mt: 1, display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="caption" color="text.secondary">
                      Vol: {formatLargeNumber(index.volume)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {index.high.toFixed(2)} / {index.low.toFixed(2)}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      ) : (
        <Paper sx={{ p: 2, mb: 3 }}>
          <EmptyStateBox
            icon={<CloudOff sx={{ fontSize: 48 }} />}
            message="No market index data available"
            submessage="Market indices will appear here once connected to a data provider."
            minHeight={120}
          />
        </Paper>
      )}

      {/* Market Breadth */}
      {marketBreadth && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Market Breadth
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center' }}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="success.main">
                    {marketBreadth.advancers}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Advancers
                  </Typography>
                </Box>
                <Typography variant="h6" color="text.secondary">
                  vs
                </Typography>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="error.main">
                    {marketBreadth.decliners}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Decliners
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="text.secondary">
                    {marketBreadth.unchanged}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Unchanged
                  </Typography>
                </Box>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', justifyContent: 'space-around' }}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h6">
                    {marketBreadth.advanceDeclineRatio.toFixed(2)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    A/D Ratio
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" color="primary.main">
                    {marketBreadth.newHighs}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    New Highs
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" color="warning.main">
                    {marketBreadth.newLows}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    New Lows
                  </Typography>
                </Box>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      )}
    </>
  );
};

export default MarketSummary;
