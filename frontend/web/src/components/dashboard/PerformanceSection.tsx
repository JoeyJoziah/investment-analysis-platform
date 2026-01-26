import React, { memo, useMemo, useState, useCallback } from 'react';
import {
  Box,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  Skeleton,
  useTheme,
  useMediaQuery,
  Chip,
  alpha,
} from '@mui/material';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ComposedChart,
  Bar,
} from 'recharts';
import { format, parseISO, subDays, subMonths, subYears } from 'date-fns';

interface PerformancePoint {
  date: string;
  value: number;
  volume?: number;
  benchmark?: number;
}

interface PerformanceSectionProps {
  data: PerformancePoint[];
  isLoading?: boolean;
  showBenchmark?: boolean;
  showVolume?: boolean;
}

type Period = '1D' | '1W' | '1M' | '3M' | 'YTD' | '1Y' | 'ALL';

/**
 * PerformanceSection - Portfolio performance chart
 *
 * Features:
 * - Area chart with gradient fill
 * - Period selection (1D to ALL)
 * - Optional volume bars
 * - Optional S&P 500 benchmark
 * - Responsive height adjustment
 * - WCAG 2.1 AA compliant
 */
const PerformanceSection: React.FC<PerformanceSectionProps> = ({
  data = [],
  isLoading = false,
  showBenchmark = false,
  showVolume = false,
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const [selectedPeriod, setSelectedPeriod] = useState<Period>('1M');

  // Period options
  const periodOptions: Period[] = ['1D', '1W', '1M', '3M', 'YTD', '1Y', 'ALL'];

  // Filter data based on selected period
  const filteredData = useMemo(() => {
    if (!data || data.length === 0) return [];

    const now = new Date();
    let startDate: Date;

    switch (selectedPeriod) {
      case '1D':
        startDate = subDays(now, 1);
        break;
      case '1W':
        startDate = subDays(now, 7);
        break;
      case '1M':
        startDate = subMonths(now, 1);
        break;
      case '3M':
        startDate = subMonths(now, 3);
        break;
      case 'YTD':
        startDate = new Date(now.getFullYear(), 0, 1);
        break;
      case '1Y':
        startDate = subYears(now, 1);
        break;
      case 'ALL':
      default:
        return data;
    }

    return data.filter((point) => {
      const pointDate = parseISO(point.date);
      return pointDate >= startDate;
    });
  }, [data, selectedPeriod]);

  // Calculate performance metrics
  const metrics = useMemo(() => {
    if (filteredData.length < 2) {
      return { startValue: 0, endValue: 0, change: 0, changePercent: 0 };
    }

    const startValue = filteredData[0].value;
    const endValue = filteredData[filteredData.length - 1].value;
    const change = endValue - startValue;
    const changePercent = startValue > 0 ? (change / startValue) * 100 : 0;

    return { startValue, endValue, change, changePercent };
  }, [filteredData]);

  const isPositive = metrics.changePercent >= 0;

  // Handle period change
  const handlePeriodChange = useCallback(
    (_: React.MouseEvent<HTMLElement>, newPeriod: Period | null) => {
      if (newPeriod) {
        setSelectedPeriod(newPeriod);
      }
    },
    []
  );

  // Format X-axis tick based on period
  const formatXAxisTick = useCallback(
    (tickItem: string) => {
      try {
        const date = parseISO(tickItem);
        if (selectedPeriod === '1D') {
          return format(date, 'HH:mm');
        } else if (selectedPeriod === '1W') {
          return format(date, 'EEE');
        } else if (selectedPeriod === '1M' || selectedPeriod === '3M') {
          return format(date, 'MMM dd');
        } else {
          return format(date, 'MMM yy');
        }
      } catch {
        return tickItem;
      }
    },
    [selectedPeriod]
  );

  // Format Y-axis tick
  const formatYAxisTick = useCallback((value: number) => {
    if (value >= 1000000) {
      return `$${(value / 1000000).toFixed(1)}M`;
    }
    if (value >= 1000) {
      return `$${(value / 1000).toFixed(0)}K`;
    }
    return `$${value.toFixed(0)}`;
  }, []);

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const date = label ? parseISO(label) : new Date();
      const value = payload[0]?.value ?? 0;
      const benchmark = payload.find((p: any) => p.dataKey === 'benchmark')?.value;

      return (
        <Box
          sx={{
            backgroundColor: theme.palette.background.paper,
            border: `1px solid ${theme.palette.divider}`,
            borderRadius: 1,
            p: 1.5,
            boxShadow: theme.shadows[4],
          }}
        >
          <Typography variant="caption" color="text.secondary" display="block">
            {format(date, 'MMM dd, yyyy')}
            {selectedPeriod === '1D' && ` ${format(date, 'HH:mm')}`}
          </Typography>
          <Typography
            variant="body2"
            sx={{
              fontWeight: 600,
              fontFamily: '"SF Mono", Monaco, monospace',
            }}
          >
            Portfolio: ${value.toLocaleString()}
          </Typography>
          {benchmark !== undefined && (
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{
                fontFamily: '"SF Mono", Monaco, monospace',
              }}
            >
              S&P 500: ${benchmark.toLocaleString()}
            </Typography>
          )}
        </Box>
      );
    }
    return null;
  };

  // Chart height based on screen size
  const chartHeight = isMobile ? 250 : 300;

  // Loading state
  if (isLoading) {
    return (
      <Box>
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mb: 2,
          }}
        >
          <Skeleton variant="text" width={180} height={32} />
          <Skeleton variant="rectangular" width={280} height={32} sx={{ borderRadius: 1 }} />
        </Box>
        <Skeleton
          variant="rectangular"
          width="100%"
          height={chartHeight}
          sx={{ borderRadius: 1 }}
        />
      </Box>
    );
  }

  // Empty state
  if (data.length === 0) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: chartHeight,
          color: 'text.secondary',
        }}
      >
        <Typography variant="body1">No performance data available</Typography>
        <Typography variant="caption">
          Data will appear once you add positions to your portfolio
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      {/* Header with metrics and period selector */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          mb: 2,
          flexWrap: 'wrap',
          gap: 2,
        }}
      >
        <Box>
          <Typography variant="h6" component="h2" gutterBottom>
            Portfolio Performance
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography
              variant="h5"
              sx={{
                fontWeight: 700,
                fontFamily: '"SF Mono", Monaco, monospace',
              }}
            >
              ${metrics.endValue.toLocaleString()}
            </Typography>
            <Chip
              label={`${isPositive ? '+' : ''}${metrics.changePercent.toFixed(2)}%`}
              size="small"
              sx={{
                backgroundColor: alpha(
                  isPositive ? theme.palette.success.main : theme.palette.error.main,
                  0.1
                ),
                color: isPositive ? theme.palette.success.main : theme.palette.error.main,
                fontWeight: 600,
                fontFamily: '"SF Mono", Monaco, monospace',
              }}
              aria-label={`${isPositive ? 'Gain' : 'Loss'} of ${Math.abs(metrics.changePercent).toFixed(2)} percent for selected period`}
            />
          </Box>
        </Box>

        <ToggleButtonGroup
          value={selectedPeriod}
          exclusive
          onChange={handlePeriodChange}
          size="small"
          aria-label="Select time period"
          sx={{
            '& .MuiToggleButton-root': {
              px: { xs: 1, sm: 1.5 },
              py: 0.5,
              fontSize: { xs: '0.75rem', sm: '0.8125rem' },
            },
          }}
        >
          {periodOptions.map((period) => (
            <ToggleButton
              key={period}
              value={period}
              aria-label={`Show ${period} performance`}
            >
              {period}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
      </Box>

      {/* Chart */}
      <Box
        role="img"
        aria-label={`Portfolio performance chart showing ${isPositive ? 'gain' : 'loss'} of ${Math.abs(metrics.changePercent).toFixed(2)} percent over ${selectedPeriod}. Starting value $${metrics.startValue.toLocaleString()}, ending value $${metrics.endValue.toLocaleString()}.`}
      >
        <ResponsiveContainer width="100%" height={chartHeight}>
          <ComposedChart
            data={filteredData}
            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
          >
            <defs>
              <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor={isPositive ? theme.palette.success.main : theme.palette.error.main}
                  stopOpacity={0.3}
                />
                <stop
                  offset="95%"
                  stopColor={isPositive ? theme.palette.success.main : theme.palette.error.main}
                  stopOpacity={0}
                />
              </linearGradient>
            </defs>

            <CartesianGrid
              strokeDasharray="3 3"
              stroke={theme.palette.divider}
              vertical={false}
            />

            <XAxis
              dataKey="date"
              tickFormatter={formatXAxisTick}
              stroke={theme.palette.text.secondary}
              tick={{ fontSize: 12 }}
              axisLine={false}
              tickLine={false}
              minTickGap={30}
            />

            <YAxis
              tickFormatter={formatYAxisTick}
              stroke={theme.palette.text.secondary}
              tick={{ fontSize: 12 }}
              axisLine={false}
              tickLine={false}
              domain={['auto', 'auto']}
              width={60}
            />

            <Tooltip content={<CustomTooltip />} />

            {/* Reference line at start value */}
            <ReferenceLine
              y={metrics.startValue}
              stroke={theme.palette.grey[400]}
              strokeDasharray="3 3"
              strokeWidth={1}
            />

            {/* Volume bars (optional) */}
            {showVolume && (
              <Bar
                dataKey="volume"
                fill={theme.palette.grey[400]}
                opacity={0.3}
                yAxisId="volume"
              />
            )}

            {/* Benchmark line (optional) */}
            {showBenchmark && (
              <Area
                type="monotone"
                dataKey="benchmark"
                stroke={theme.palette.grey[500]}
                strokeWidth={1}
                strokeDasharray="4 4"
                fill="none"
              />
            )}

            {/* Main portfolio value area */}
            <Area
              type="monotone"
              dataKey="value"
              stroke={isPositive ? theme.palette.success.main : theme.palette.error.main}
              strokeWidth={2}
              fill="url(#colorValue)"
              dot={false}
              activeDot={{
                r: 6,
                fill: theme.palette.background.paper,
                stroke: isPositive ? theme.palette.success.main : theme.palette.error.main,
                strokeWidth: 2,
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </Box>

      {/* Legend (if showing benchmark) */}
      {showBenchmark && (
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            gap: 3,
            mt: 2,
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Box
              sx={{
                width: 24,
                height: 3,
                borderRadius: 1,
                backgroundColor: isPositive
                  ? theme.palette.success.main
                  : theme.palette.error.main,
              }}
            />
            <Typography variant="caption" color="text.secondary">
              Portfolio
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Box
              sx={{
                width: 24,
                height: 3,
                borderRadius: 1,
                backgroundColor: theme.palette.grey[500],
              }}
            />
            <Typography variant="caption" color="text.secondary">
              S&P 500
            </Typography>
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default memo(PerformanceSection);
