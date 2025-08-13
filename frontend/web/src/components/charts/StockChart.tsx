import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Brush,
  ComposedChart
} from 'recharts';
import { Box, Paper, Typography, ToggleButton, ToggleButtonGroup, Chip, useTheme } from '@mui/material';
import { format, parseISO } from 'date-fns';

interface StockChartProps {
  data?: Array<{
    date: string;
    open?: number;
    high?: number;
    low?: number;
    close: number;
    volume?: number;
    ma20?: number;
    ma50?: number;
    ma200?: number;
    rsi?: number;
    macd?: number;
    signal?: number;
  }>;
  height?: number;
  showVolume?: boolean;
  showIndicators?: boolean;
  chartType?: 'line' | 'area' | 'candlestick';
  ticker?: string;
}

const StockChart: React.FC<StockChartProps> = ({
  data = [],
  height = 400,
  showVolume = true,
  showIndicators = false,
  chartType = 'area',
  ticker
}) => {
  const theme = useTheme();
  const [selectedPeriod, setSelectedPeriod] = React.useState('1M');
  const [selectedIndicators, setSelectedIndicators] = React.useState<string[]>([]);

  const periodOptions = ['1D', '1W', '1M', '3M', '6M', '1Y', 'ALL'];
  const indicatorOptions = ['MA20', 'MA50', 'MA200', 'RSI', 'MACD'];

  const filteredData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    let days = data.length;
    switch (selectedPeriod) {
      case '1D': days = 1; break;
      case '1W': days = 7; break;
      case '1M': days = 30; break;
      case '3M': days = 90; break;
      case '6M': days = 180; break;
      case '1Y': days = 365; break;
      case 'ALL': days = data.length; break;
    }
    
    return data.slice(-days);
  }, [data, selectedPeriod]);

  const formatXAxisTick = (tickItem: string) => {
    try {
      const date = parseISO(tickItem);
      if (selectedPeriod === '1D') {
        return format(date, 'HH:mm');
      } else if (selectedPeriod === '1W') {
        return format(date, 'EEE');
      } else if (selectedPeriod === '1M') {
        return format(date, 'MMM dd');
      } else {
        return format(date, 'MMM yy');
      }
    } catch {
      return tickItem;
    }
  };

  const formatTooltipLabel = (value: any, name: string) => {
    if (name === 'volume') {
      return `$${(value / 1000000).toFixed(2)}M`;
    }
    return `$${value?.toFixed(2) || 0}`;
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Paper sx={{ p: 2 }}>
          <Typography variant="body2" gutterBottom>
            {label && format(parseISO(label), 'MMM dd, yyyy')}
          </Typography>
          {payload.map((entry: any, index: number) => (
            <Typography
              key={index}
              variant="body2"
              sx={{ color: entry.color }}
            >
              {entry.name}: {formatTooltipLabel(entry.value, entry.name)}
            </Typography>
          ))}
        </Paper>
      );
    }
    return null;
  };

  const latestPrice = filteredData[filteredData.length - 1]?.close || 0;
  const firstPrice = filteredData[0]?.close || 0;
  const priceChange = latestPrice - firstPrice;
  const priceChangePercent = firstPrice ? ((priceChange / firstPrice) * 100) : 0;

  return (
    <Box>
      {ticker && (
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Box display="flex" alignItems="center" gap={2}>
            <Typography variant="h5">{ticker}</Typography>
            <Typography variant="h4">${latestPrice.toFixed(2)}</Typography>
            <Chip
              label={`${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)} (${priceChangePercent.toFixed(2)}%)`}
              color={priceChange >= 0 ? 'success' : 'error'}
              size="small"
            />
          </Box>
          <ToggleButtonGroup
            value={selectedPeriod}
            exclusive
            onChange={(e, newPeriod) => newPeriod && setSelectedPeriod(newPeriod)}
            size="small"
          >
            {periodOptions.map(period => (
              <ToggleButton key={period} value={period}>
                {period}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        </Box>
      )}

      <ResponsiveContainer width="100%" height={height}>
        {chartType === 'line' ? (
          <LineChart data={filteredData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
            <XAxis 
              dataKey="date" 
              tickFormatter={formatXAxisTick}
              stroke={theme.palette.text.secondary}
            />
            <YAxis 
              domain={['auto', 'auto']}
              stroke={theme.palette.text.secondary}
              tickFormatter={(value) => `$${value}`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line
              type="monotone"
              dataKey="close"
              stroke={theme.palette.primary.main}
              strokeWidth={2}
              dot={false}
              name="Price"
            />
            {selectedIndicators.includes('MA20') && (
              <Line
                type="monotone"
                dataKey="ma20"
                stroke={theme.palette.info.main}
                strokeWidth={1}
                dot={false}
                name="MA20"
                strokeDasharray="5 5"
              />
            )}
            {selectedIndicators.includes('MA50') && (
              <Line
                type="monotone"
                dataKey="ma50"
                stroke={theme.palette.warning.main}
                strokeWidth={1}
                dot={false}
                name="MA50"
                strokeDasharray="5 5"
              />
            )}
            {selectedIndicators.includes('MA200') && (
              <Line
                type="monotone"
                dataKey="ma200"
                stroke={theme.palette.error.main}
                strokeWidth={1}
                dot={false}
                name="MA200"
                strokeDasharray="5 5"
              />
            )}
          </LineChart>
        ) : chartType === 'area' ? (
          <ComposedChart data={filteredData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <defs>
              <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={theme.palette.primary.main} stopOpacity={0.3}/>
                <stop offset="95%" stopColor={theme.palette.primary.main} stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="colorVolume" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={theme.palette.secondary.main} stopOpacity={0.3}/>
                <stop offset="95%" stopColor={theme.palette.secondary.main} stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
            <XAxis 
              dataKey="date" 
              tickFormatter={formatXAxisTick}
              stroke={theme.palette.text.secondary}
            />
            <YAxis 
              yAxisId="price"
              orientation="left"
              domain={['auto', 'auto']}
              stroke={theme.palette.text.secondary}
              tickFormatter={(value) => `$${value}`}
            />
            {showVolume && (
              <YAxis
                yAxisId="volume"
                orientation="right"
                stroke={theme.palette.text.secondary}
                tickFormatter={(value) => `${(value / 1000000).toFixed(0)}M`}
              />
            )}
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Area
              yAxisId="price"
              type="monotone"
              dataKey="close"
              stroke={theme.palette.primary.main}
              strokeWidth={2}
              fill="url(#colorPrice)"
              name="Price"
            />
            {showVolume && (
              <Bar
                yAxisId="volume"
                dataKey="volume"
                fill={theme.palette.secondary.main}
                opacity={0.3}
                name="Volume"
              />
            )}
            {selectedIndicators.includes('MA20') && (
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="ma20"
                stroke={theme.palette.info.main}
                strokeWidth={1}
                dot={false}
                name="MA20"
                strokeDasharray="5 5"
              />
            )}
            {selectedIndicators.includes('MA50') && (
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="ma50"
                stroke={theme.palette.warning.main}
                strokeWidth={1}
                dot={false}
                name="MA50"
                strokeDasharray="5 5"
              />
            )}
            <Brush
              dataKey="date"
              height={30}
              stroke={theme.palette.primary.main}
              tickFormatter={formatXAxisTick}
            />
          </ComposedChart>
        ) : null}
      </ResponsiveContainer>

      {showIndicators && (
        <Box mt={2}>
          <Typography variant="body2" gutterBottom>
            Technical Indicators
          </Typography>
          <ToggleButtonGroup
            value={selectedIndicators}
            onChange={(e, newIndicators) => setSelectedIndicators(newIndicators)}
            size="small"
          >
            {indicatorOptions.map(indicator => (
              <ToggleButton key={indicator} value={indicator}>
                {indicator}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        </Box>
      )}
    </Box>
  );
};

export default StockChart;