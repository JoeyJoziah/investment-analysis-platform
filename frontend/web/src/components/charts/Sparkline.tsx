import React, { memo, useMemo } from 'react';
import { Box, useTheme } from '@mui/material';
import { LineChart, Line, ResponsiveContainer } from 'recharts';

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: 'auto' | 'gain' | 'loss' | 'neutral';
  strokeWidth?: number;
}

/**
 * Sparkline - Compact trend visualization
 *
 * Features:
 * - Auto color based on trend
 * - Minimal footprint (no axes)
 * - Smooth curve interpolation
 *
 * WCAG 2.1 AA Compliant:
 * - Provides aria-label with trend description
 */
const Sparkline: React.FC<SparklineProps> = ({
  data = [],
  width = 60,
  height = 24,
  color = 'auto',
  strokeWidth = 1.5,
}) => {
  const theme = useTheme();

  // Calculate trend
  const trend = useMemo(() => {
    if (data.length < 2) return 0;
    const first = data[0];
    const last = data[data.length - 1];
    return last - first;
  }, [data]);

  // Determine line color
  const lineColor = useMemo(() => {
    if (color === 'auto') {
      if (trend > 0) return theme.palette.success.main;
      if (trend < 0) return theme.palette.error.main;
      return theme.palette.grey[500];
    }
    switch (color) {
      case 'gain':
        return theme.palette.success.main;
      case 'loss':
        return theme.palette.error.main;
      default:
        return theme.palette.grey[500];
    }
  }, [color, trend, theme.palette]);

  // Format data for recharts
  const chartData = useMemo(
    () => data.map((value, index) => ({ index, value })),
    [data]
  );

  // Calculate trend percentage for aria-label
  const trendPercent = useMemo(() => {
    if (data.length < 2 || data[0] === 0) return 0;
    return ((data[data.length - 1] - data[0]) / data[0]) * 100;
  }, [data]);

  if (data.length < 2) {
    return (
      <Box
        sx={{
          width,
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Box
          sx={{
            width: '100%',
            height: 1,
            backgroundColor: theme.palette.grey[300],
          }}
        />
      </Box>
    );
  }

  return (
    <Box
      sx={{
        width,
        height,
        display: 'flex',
        alignItems: 'center',
      }}
      role="img"
      aria-label={`Price trend: ${trend >= 0 ? 'up' : 'down'} ${Math.abs(trendPercent).toFixed(1)}% over period`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
          <Line
            type="monotone"
            dataKey="value"
            stroke={lineColor}
            strokeWidth={strokeWidth}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default memo(Sparkline);
