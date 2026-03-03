import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableRow,
  Divider,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import {
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import StockChart from '../charts/StockChart';
import type { StockChart as StockChartData } from '../../store/slices/stockSlice';
import type { TechnicalIndicators } from '../../store/slices/stockSlice';

interface RadarDataPoint {
  signal: string;
  value: number;
}

export interface AnalysisChartsProps {
  chartData: StockChartData | null;
  chartType: 'line' | 'candle';
  chartInterval: string;
  onChartIntervalChange: (interval: string) => void;
  onChartTypeChange: (type: 'line' | 'candle') => void;
  technicalIndicators: TechnicalIndicators | null;
  radarData: RadarDataPoint[];
  formatCurrency: (value: number) => string;
}

export const ChartTabContent: React.FC<
  Pick<AnalysisChartsProps, 'chartData' | 'chartType' | 'chartInterval' | 'onChartIntervalChange' | 'onChartTypeChange'>
> = ({ chartData, chartType, chartInterval, onChartIntervalChange, onChartTypeChange }) => {
  return (
    <>
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between' }}>
        <ToggleButtonGroup
          value={chartInterval}
          exclusive
          onChange={(_, newInterval) => newInterval && onChartIntervalChange(newInterval)}
          size="small"
        >
          <ToggleButton value="1d">1D</ToggleButton>
          <ToggleButton value="1w">1W</ToggleButton>
          <ToggleButton value="1m">1M</ToggleButton>
          <ToggleButton value="3m">3M</ToggleButton>
          <ToggleButton value="6m">6M</ToggleButton>
          <ToggleButton value="1y">1Y</ToggleButton>
          <ToggleButton value="5y">5Y</ToggleButton>
          <ToggleButton value="max">MAX</ToggleButton>
        </ToggleButtonGroup>
        <ToggleButtonGroup
          value={chartType}
          exclusive
          onChange={(_, newType) => newType && onChartTypeChange(newType)}
          size="small"
        >
          <ToggleButton value="line">Line</ToggleButton>
          <ToggleButton value="candle">Candlestick</ToggleButton>
        </ToggleButtonGroup>
      </Box>
      <Box sx={{ height: 500 }}>
        {chartData && <StockChart data={chartData.data} chartType={chartType === 'candle' ? 'area' : chartType} />}
      </Box>
    </>
  );
};

export const TechnicalTabContent: React.FC<
  Pick<AnalysisChartsProps, 'technicalIndicators' | 'radarData' | 'formatCurrency'>
> = ({ technicalIndicators, radarData, formatCurrency }) => {
  if (!technicalIndicators) return null;

  const signals = technicalIndicators.signals;

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Typography variant="h6" gutterBottom>
          Technical Indicators
        </Typography>
        <TableContainer>
          <Table size="small">
            <TableBody>
              <TableRow>
                <TableCell>RSI (14)</TableCell>
                <TableCell align="right">
                  <Chip
                    label={(technicalIndicators.rsi ?? 0).toFixed(2)}
                    color={
                      (technicalIndicators.rsi ?? 50) > 70
                        ? 'error'
                        : (technicalIndicators.rsi ?? 50) < 30
                        ? 'success'
                        : 'default'
                    }
                    size="small"
                  />
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell>MACD</TableCell>
                <TableCell align="right">
                  {(technicalIndicators.macd?.macd ?? 0).toFixed(2)}
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell>MACD Signal</TableCell>
                <TableCell align="right">
                  {(technicalIndicators.macd?.signal ?? 0).toFixed(2)}
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell>MACD Histogram</TableCell>
                <TableCell align="right">
                  <Typography
                    color={(technicalIndicators.macd?.histogram ?? 0) > 0 ? 'success.main' : 'error.main'}
                  >
                    {(technicalIndicators.macd?.histogram ?? 0).toFixed(2)}
                  </Typography>
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell>SMA 20</TableCell>
                <TableCell align="right">
                  {formatCurrency(technicalIndicators.sma?.sma20 ?? 0)}
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell>SMA 50</TableCell>
                <TableCell align="right">
                  {formatCurrency(technicalIndicators.sma?.sma50 ?? 0)}
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell>SMA 200</TableCell>
                <TableCell align="right">
                  {formatCurrency(technicalIndicators.sma?.sma200 ?? 0)}
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Bollinger Upper</TableCell>
                <TableCell align="right">
                  {formatCurrency(technicalIndicators.bollingerBands?.upper ?? 0)}
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Bollinger Middle</TableCell>
                <TableCell align="right">
                  {formatCurrency(technicalIndicators.bollingerBands?.middle ?? 0)}
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Bollinger Lower</TableCell>
                <TableCell align="right">
                  {formatCurrency(technicalIndicators.bollingerBands?.lower ?? 0)}
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell>ATR</TableCell>
                <TableCell align="right">
                  {(technicalIndicators.atr ?? 0).toFixed(2)}
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell>ADX</TableCell>
                <TableCell align="right">
                  {(technicalIndicators.adx ?? 0).toFixed(2)}
                </TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </TableContainer>
      </Grid>
      <Grid item xs={12} md={6}>
        <Typography variant="h6" gutterBottom>
          Analysis Signals
        </Typography>
        <Box sx={{ mb: 3 }}>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="signal" />
              <PolarRadiusAxis angle={90} domain={[0, 100]} />
              <Radar name="Signal Strength" dataKey="value" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
            </RadarChart>
          </ResponsiveContainer>
        </Box>
        {signals && (
        <Paper sx={{ p: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            Trading Signals
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography>Trend</Typography>
              <Chip
                label={signals.trend.toUpperCase()}
                color={
                  signals.trend === 'bullish'
                    ? 'success'
                    : signals.trend === 'bearish'
                    ? 'error'
                    : 'default'
                }
                size="small"
              />
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography>Momentum</Typography>
              <Chip
                label={signals.momentum.toUpperCase()}
                color={
                  signals.momentum === 'strong'
                    ? 'success'
                    : signals.momentum === 'weak'
                    ? 'error'
                    : 'warning'
                }
                size="small"
              />
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography>Volatility</Typography>
              <Chip
                label={signals.volatility.toUpperCase()}
                color={
                  signals.volatility === 'high'
                    ? 'error'
                    : signals.volatility === 'low'
                    ? 'success'
                    : 'warning'
                }
                size="small"
              />
            </Box>
            <Divider />
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography fontWeight="bold">Recommendation</Typography>
              <Chip
                label={signals.recommendation.replace('_', ' ').toUpperCase()}
                color={
                  signals.recommendation.includes('buy')
                    ? 'success'
                    : signals.recommendation.includes('sell')
                    ? 'error'
                    : 'warning'
                }
              />
            </Box>
          </Box>
        </Paper>
        )}
      </Grid>
    </Grid>
  );
};

export default { ChartTabContent, TechnicalTabContent };
